import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 21

def construct():
    start_time = time.time()
    time_limit = 145.0

    def get_wh(a):
        h = 2.0 / (1.0 + a)
        w = 2.0 * a / (1.0 + a)
        return w, h

    def _softplus(x, beta):
        return np.log1p(np.exp(-beta * x)) / beta

    def _softplus_grad(x, beta):
        return 1.0 / (1.0 + np.exp(beta * x))

    def joint_objective_and_grad(params, w, h, beta=30.0):
        """
        Joint optimization of positions (x,y) and radii (r).
        params: [x_0,...,x_{n-1}, y_0,...,y_{n-1}, r_0,...,r_{n-1}]
        Maximize sum(r) subject to:
          - r_i >= 0
          - r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
          - r_i + r_j <= dist(i,j) for all pairs
        Using smooth penalty: objective = sum(r) - penalty
        """
        n = N
        xs = params[0:n]
        ys = params[n:2*n]
        rs = params[2*n:3*n]
        
        f = -rs.sum()  # minimize negative sum of radii
        g = np.zeros(3*n)
        
        # Gradient of -sum(r)
        g[2*n:3*n] = -1.0
        
        # Boundary violations: r_i - x_i > 0, r_i - (w-x_i) > 0, etc.
        # Penalty: softplus(r_i - x_i) + softplus(r_i - (w-x_i)) + ...
        viol_x0 = rs - xs
        sp = _softplus(viol_x0, beta)
        f += sp.sum()
        g[2*n:3*n] += _softplus_grad(viol_x0, beta)
        g[0:n] -= _softplus_grad(viol_x0, beta)
        
        viol_x1 = rs - (w - xs)
        sp = _softplus(viol_x1, beta)
        f += sp.sum()
        g[2*n:3*n] += _softplus_grad(viol_x1, beta)
        g[0:n] += _softplus_grad(viol_x1, beta)
        
        viol_y0 = rs - ys
        sp = _softplus(viol_y0, beta)
        f += sp.sum()
        g[2*n:3*n] += _softplus_grad(viol_y0, beta)
        g[n:2*n] -= _softplus_grad(viol_y0, beta)
        
        viol_y1 = rs - (h - ys)
        sp = _softplus(viol_y1, beta)
        f += sp.sum()
        g[2*n:3*n] += _softplus_grad(viol_y1, beta)
        g[n:2*n] += _softplus_grad(viol_y1, beta)
        
        # Negative radius violation: -r_i > 0
        viol_neg = -rs
        sp = _softplus(viol_neg, beta)
        f += sp.sum()
        g[2*n:3*n] -= _softplus_grad(viol_neg, beta)
        
        # Pairwise overlap: r_i + r_j - dist(i,j) > 0
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist_sq = dx*dx + dy*dy
        np.fill_diagonal(dist_sq, 1e-12)
        dist = np.sqrt(dist_sq)
        
        # Only penalize close pairs for efficiency
        mask = dist < 2.5 * (rs.max() if rs.max() > 0 else 0.2)
        np.fill_diagonal(mask, False)
        
        viol = rs[:, None] + rs[None, :] - dist
        sp_vals = _softplus(viol, beta)
        sp_vals[~mask] = 0.0
        f += sp_vals.sum()
        
        grad_sp = _softplus_grad(viol, beta)
        grad_sp[~mask] = 0.0
        
        # Gradient w.r.t. r_i: sum_j grad_sp[i,j]
        g_r = grad_sp.sum(axis=1)
        g[2*n:3*n] += g_r
        
        # Gradient w.r.t. x_i: sum_j grad_sp[i,j] * (-dx[i,j]/dist[i,j])
        # Note: d(dist)/d(x_i) = (x_i - x_j)/dist = dx[i,j]/dist
        # d(viol)/d(x_i) = -dx[i,j]/dist
        # d(sp)/d(x_i) = grad_sp[i,j] * (-dx[i,j]/dist[i,j])
        inv_dist = np.where(dist > 1e-10, 1.0/dist, 0.0)
        g_x = (grad_sp * (-dx) * inv_dist).sum(axis=1)
        g_y = (grad_sp * (-dy) * inv_dist).sum(axis=1)
        
        g[0:n] += g_x
        g[n:2*n] += g_y
        
        return f, g

    def is_valid(params, w, h, tol=1e-5):
        xs = params[0:N]
        ys = params[N:2*N]
        rs = params[2*N:3*N]
        
        if np.any(rs < -tol): return False
        if np.any(xs + rs < -tol) or np.any(w - xs + rs < -tol): return False
        if np.any(ys + rs < -tol) or np.any(h - ys + rs < -tol): return False
        
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        np.fill_diagonal(dist, np.inf)
        
        if np.any(dist < rs[:, None] + rs[None, :] - tol):
            return False
        return True

    def make_grid_init(w, h, rng, noise_scale=0.02):
        best_init = None
        best_sum = -1
        
        for rows in range(3, 8):
            cols = int(np.ceil(N / rows))
            if cols < 2:
                continue
            
            xs_init = np.zeros(N)
            ys_init = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                xs_init[i] = (col + 0.5) * (w / cols)
                ys_init[i] = (row + 0.5) * (h / rows)
            
            xs_init += rng.normal(0, noise_scale, N)
            ys_init += rng.normal(0, noise_scale, N)
            
            xs_init = np.clip(xs_init, 0.05, w - 0.05)
            ys_init = np.clip(ys_init, 0.05, h - 0.05)
            
            r_init = np.minimum(np.minimum(xs_init, w - xs_init),
                                np.minimum(ys_init, h - ys_init)) * 0.4
            r_init = np.maximum(r_init, 0.01)
            
            params = np.concatenate([xs_init, ys_init, r_init])
            
            s = np.sum(r_init)
            if s > best_sum:
                best_sum = s
                best_init = params
        
        return best_init

    def make_hex_init(w, h, rng):
        patterns = [
            [5, 4, 5, 4, 3],
            [4, 5, 4, 5, 3],
            [5, 5, 5, 4, 2],
            [3, 4, 5, 4, 5],
            [4, 4, 5, 4, 4],
            [7, 7, 7],
            [4, 4, 4, 4, 5],
        ]
        
        best_init = None
        best_sum = -1
        
        for pattern in patterns:
            if sum(pattern) != N:
                continue
            
            xs_init = np.zeros(N)
            ys_init = np.zeros(N)
            
            idx = 0
            n_rows = len(pattern)
            row_h = h / n_rows
            
            for row_idx, n_cols in enumerate(pattern):
                col_w = w / n_cols
                offset = 0.0 if row_idx % 2 == 0 else col_w / 2
                for col_idx in range(n_cols):
                    if idx >= N:
                        break
                    x = (col_idx + 0.5) * col_w + offset
                    y = (row_idx + 0.5) * row_h
                    xs_init[idx] = x
                    ys_init[idx] = y
                    idx += 1
            
            if idx != N:
                continue
            
            xs_init += rng.normal(0, 0.01, N)
            ys_init += rng.normal(0, 0.01, N)
            
            xs_init = np.clip(xs_init, 0.05, w - 0.05)
            ys_init = np.clip(ys_init, 0.05, h - 0.05)
            
            r_init = np.minimum(np.minimum(xs_init, w - xs_init),
                                np.minimum(ys_init, h - ys_init)) * 0.4
            r_init = np.maximum(r_init, 0.01)
            
            params = np.concatenate([xs_init, ys_init, r_init])
            s = np.sum(r_init)
            if s > best_sum:
                best_sum = s
                best_init = params
        
        return best_init

    def make_random_init(w, h, rng):
        xs_init = rng.uniform(0.1, w - 0.1, N)
        ys_init = rng.uniform(0.1, h - 0.1, N)
        r_init = np.minimum(np.minimum(xs_init, w - xs_init),
                            np.minimum(ys_init, h - ys_init)) * 0.3
        r_init = np.maximum(r_init, 0.01)
        return np.concatenate([xs_init, ys_init, r_init])

    def polish(params, w, h, start_beta=20.0, end_beta=100.0, n_phases=5):
        """Multi-phase polishing with increasing beta."""
        current = params.copy()
        betas = np.linspace(start_beta, end_beta, n_phases)
        
        for beta in betas:
            if time.time() - start_time > time_limit - 15:
                break
            
            try:
                res = minimize(joint_objective_and_grad, current, 
                               args=(w, h, beta),
                               jac=True, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                
                if res.fun < joint_objective_and_grad(current, w, h, beta)[0]:
                    current = res.x
            except Exception:
                pass
        
        return current

    best_val = 0.0
    best_params = None
    best_a = 1.0
    
    rng = np.random.default_rng(42)
    
    # Simulated annealing over aspect ratio
    a_current = 1.0
    a_best = 1.0
    val_best = 0.0
    T = 0.05  # initial temperature
    
    # Pre-compute a grid of aspect ratios for exploration
    a_grid = np.linspace(0.80, 1.25, 18)
    
    # Phase 1: Coarse search over aspect ratios with multiple restarts
    for a in a_grid:
        if time.time() - start_time > time_limit - 40:
            break
        
        w, h = get_wh(a)
        
        n_restarts = 8
        for k in range(n_restarts):
            if time.time() - start_time > time_limit - 30:
                break
            
            if k % 3 == 0:
                p0 = make_grid_init(w, h, rng, noise_scale=0.02)
            elif k % 3 == 1:
                p0 = make_hex_init(w, h, rng)
                if p0 is None:
                    p0 = make_grid_init(w, h, rng, noise_scale=0.03)
            else:
                p0 = make_random_init(w, h, rng)
            
            if p0 is None:
                continue
            
            # Multi-phase optimization
            try:
                p_opt = polish(p0, w, h, start_beta=15.0, end_beta=60.0, n_phases=4)
                
                # Validate and compute value
                if is_valid(p_opt, w, h):
                    val = np.sum(p_opt[2*N:3*N])
                    if val > val_best:
                        val_best = val
                        best_val = val
                        best_params = p_opt.copy()
                        best_a = a
                        a_best = a
            except Exception:
                pass
    
    # Phase 2: Fine-tune around best aspect ratio
    if best_params is not None:
        if time.time() - start_time < time_limit - 20:
            a_fine = np.linspace(best_a - 0.05, best_a + 0.05, 9)
            
            for a in a_fine:
                if time.time() - start_time > time_limit - 10:
                    break
                
                w, h = get_wh(a)
                
                for k in range(6):
                    if time.time() - start_time > time_limit - 8:
                        break
                    
                    # Perturb best solution
                    p0 = best_params.copy()
                    p0[0:N] += rng.normal(0, 0.005, N)
                    p0[N:2*N] += rng.normal(0, 0.005, N)
                    p0 = np.clip(p0, 0.01, None)
                    
                    try:
                        p_opt = polish(p0, w, h, start_beta=30.0, end_beta=100.0, n_phases=3)
                        
                        if is_valid(p_opt, w, h):
                            val = np.sum(p_opt[2*N:3*N])
                            if val > best_val:
                                best_val = val
                                best_params = p_opt.copy()
                                best_a = a
                    except Exception:
                        pass
    
    # Final high-precision polish
    if best_params is not None and time.time() - start_time < time_limit - 5:
        w, h = get_wh(best_a)
        try:
            p_final = polish(best_params, w, h, start_beta=50.0, end_beta=200.0, n_phases=3)
            if is_valid(p_final, w, h):
                val = np.sum(p_final[2*N:3*N])
                if val > best_val:
                    best_val = val
                    best_params = p_final
        except Exception:
            pass
    
    # Format output
    if best_params is not None:
        out = best_params.reshape(N, 3)
        out[:, 2] = np.maximum(0, out[:, 2])
        return out
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
