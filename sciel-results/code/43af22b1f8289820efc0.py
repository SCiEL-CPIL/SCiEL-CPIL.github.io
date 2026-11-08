import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_result = None
    best_score = -np.inf
    best_w = 1.0

    def validate(xs, ys, rs, w, h):
        """Check that all circles are within bounds and non-overlapping."""
        if len(xs) != N or len(ys) != N or len(rs) != N:
            return False
        if np.any(rs < 1e-9):
            return False
        if np.any(xs < -1e-7) or np.any(xs > w + 1e-7):
            return False
        if np.any(ys < -1e-7) or np.any(ys > h + 1e-7):
            return False
        # Check pairwise non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if rs[i] + rs[j] > d + 1e-6:
                    return False
        return True

    def compute_score(xs, ys, rs, w, h):
        """Compute sum of radii if valid, else -inf."""
        if not validate(xs, ys, rs, w, h):
            return -np.inf
        return float(np.sum(rs))

    def max_feasible_radii(xs, ys, w, h):
        """Compute the maximum feasible radius for each circle given positions."""
        rs = np.zeros(N)
        for i in range(N):
            # Wall constraint
            r_max = min(xs[i], w - xs[i], ys[i], h - ys[i])
            # Pairwise constraint
            for j in range(N):
                if j == i:
                    continue
                d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                r_max = min(r_max, d / 2.0)
            rs[i] = max(r_max * 0.9999, 1e-4)
        return rs

    def grid_init(w, h, seed):
        """Grid-like initialization with better spacing."""
        np.random.seed(seed)
        cols = 5
        rows = 5
        xs = []
        ys = []
        for i in range(cols):
            for j in range(rows):
                if len(xs) < N:
                    xs.append((i + 0.5) / cols * w + np.random.uniform(-0.03, 0.03) * w)
                    ys.append((j + 0.5) / rows * h + np.random.uniform(-0.03, 0.03) * h)
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        rs = max_feasible_radii(xs, ys, w, h)
        return xs, ys, rs

    def random_init(w, h, seed):
        """Random initialization with jitter."""
        np.random.seed(seed)
        xs = np.random.uniform(0.05 * w, 0.95 * w, N)
        ys = np.random.uniform(0.05 * h, 0.95 * h, N)
        rs = max_feasible_radii(xs, ys, w, h)
        return xs, ys, rs

    def hex_init(w, h, seed):
        """Hexagonal packing initialization."""
        np.random.seed(seed)
        xs = []
        ys = []
        # Try to fit hexagonal pattern
        for row in range(6):
            for col in range(5):
                if len(xs) < N:
                    x = (col + 0.5) / 5 * w
                    y = (row + 0.5) / 6 * h
                    if row % 2 == 1:
                        x += 0.5 / 5 * w
                    xs.append(x + np.random.uniform(-0.02, 0.02) * w)
                    ys.append(y + np.random.uniform(-0.02, 0.02) * h)
        xs = np.array(xs[:N])
        ys = np.array(ys[:N])
        # Clip to bounds
        xs = np.clip(xs, 0, w)
        ys = np.clip(ys, 0, h)
        rs = max_feasible_radii(xs, ys, w, h)
        return xs, ys, rs

    def soft_potential_energy(xs, ys, rs, w, h, softness=0.1):
        """
        Compute soft potential energy: sum of overlaps with soft penalty.
        Returns (energy, forces) where forces are gradients w.r.t. positions.
        """
        energy = 0.0
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d_sq = dx * dx + dy * dy
                d = np.sqrt(d_sq) if d_sq > 1e-14 else 1e-7
                r_sum = rs[i] + rs[j]
                if d < r_sum:
                    overlap = r_sum - d
                    # Soft quadratic penalty
                    f_mag = 2.0 * overlap
                    inv_d = 1.0 / d
                    fx[i] += f_mag * dx * inv_d
                    fy[i] += f_mag * dy * inv_d
                    fx[j] -= f_mag * dx * inv_d
                    fy[j] -= f_mag * dy * inv_d
                    energy += 0.5 * overlap * overlap
        
        # Wall repulsion
        for i in range(N):
            # Left wall
            o = rs[i] - xs[i]
            if o > 0:
                fx[i] += 2.0 * o
                energy += 0.5 * o * o
            # Right wall
            o = rs[i] - (w - xs[i])
            if o > 0:
                fx[i] -= 2.0 * o
                energy += 0.5 * o * o
            # Bottom wall
            o = rs[i] - ys[i]
            if o > 0:
                fy[i] += 2.0 * o
                energy += 0.5 * o * o
            # Top wall
            o = rs[i] - (h - ys[i])
            if o > 0:
                fy[i] -= 2.0 * o
                energy += 0.5 * o * o
        
        return energy, fx, fy

    def simulated_annealing(xs, ys, w, h, seed, max_iters=2000, t_start=0.05, t_end=0.001):
        """
        Simulated annealing to find better positions.
        Radii are recomputed as max feasible at each step.
        """
        np.random.seed(seed)
        xs_curr = xs.copy()
        ys_curr = ys.copy()
        rs_curr = max_feasible_radii(xs_curr, ys_curr, w, h)
        score_curr = np.sum(rs_curr)
        
        best_xs = xs_curr.copy()
        best_ys = ys_curr.copy()
        best_score = score_curr
        
        n_iter = max_iters
        for it in range(n_iter):
            if time.time() > deadline - 2:
                break
            
            # Temperature schedule (exponential cooling)
            t = t_start * (t_end / t_start) ** (it / max(n_iter - 1, 1))
            
            # Generate perturbation
            step_size = t * min(w, h)
            dx = np.random.normal(0, step_size, N)
            dy = np.random.normal(0, step_size, N)
            
            xs_prop = xs_curr + dx
            ys_prop = ys_curr + dy
            xs_prop = np.clip(xs_prop, 0, w)
            ys_prop = np.clip(ys_prop, 0, h)
            
            rs_prop = max_feasible_radii(xs_prop, ys_prop, w, h)
            score_prop = np.sum(rs_prop)
            
            # Accept if better or with probability exp(-delta/t)
            delta = score_prop - score_curr
            if delta > 0 or np.random.rand() < np.exp(-delta / (t + 1e-10)):
                xs_curr = xs_prop
                ys_curr = ys_prop
                rs_curr = rs_prop
                score_curr = score_prop
                
                if score_curr > best_score:
                    best_score = score_curr
                    best_xs = xs_curr.copy()
                    best_ys = ys_curr.copy()
        
        return best_xs, best_ys, best_score

    def optimize_fixed_box(w, h, xs_init, ys_init, penalty_weight=1000.0):
        """
        Optimize circle positions and radii in a fixed box [0,w] x [0,h].
        Variables: [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
        Objective: -sum(r) + penalty for violations.
        """
        n = N
        rs_init = max_feasible_radii(xs_init, ys_init, w, h)
        x0 = np.concatenate([xs_init, ys_init, rs_init])
        bounds = [(0, w)] * n + [(0, h)] * n + [(1e-4, min(w, h))] * n

        def obj_grad(x):
            xs = x[0:n]
            ys = x[n:2*n]
            rs = x[2*n:3*n]

            val = -np.sum(rs)
            pen = 0.0
            dpen = np.zeros_like(x)

            # Pairwise overlap penalty
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    d_sq = dx * dx + dy * dy
                    d = np.sqrt(d_sq) if d_sq > 1e-14 else 1e-7
                    overlap = rs[i] + rs[j] - d
                    if overlap > 0:
                        pen += overlap * overlap
                        f = 2.0 * overlap
                        inv_d = 1.0 / d
                        dpen[i] += f * (-dx * inv_d)
                        dpen[n + i] += f * (-dy * inv_d)
                        dpen[2*n + i] += f
                        dpen[j] += f * (dx * inv_d)
                        dpen[n + j] += f * (dy * inv_d)
                        dpen[2*n + j] += f

            # Boundary penalties
            for i in range(n):
                # Left: r_i - x_i > 0
                o = rs[i] - xs[i]
                if o > 0:
                    pen += o * o
                    f = 2.0 * o
                    dpen[i] += f * (-1.0)
                    dpen[2*n + i] += f
                # Right: r_i - (w - x_i) > 0
                o = rs[i] - (w - xs[i])
                if o > 0:
                    pen += o * o
                    f = 2.0 * o
                    dpen[i] += f
                    dpen[2*n + i] += f
                # Bottom: r_i - y_i > 0
                o = rs[i] - ys[i]
                if o > 0:
                    pen += o * o
                    f = 2.0 * o
                    dpen[n + i] += f * (-1.0)
                    dpen[2*n + i] += f
                # Top: r_i - (h - y_i) > 0
                o = rs[i] - (h - ys[i])
                if o > 0:
                    pen += o * o
                    f = 2.0 * o
                    dpen[n + i] += f
                    dpen[2*n + i] += f

            val += penalty_weight * pen
            grad = np.zeros_like(x)
            grad[2*n:3*n] = -1.0
            grad += penalty_weight * dpen
            return val, grad

        try:
            res = minimize(obj_grad, x0, jac=True, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10,
                                    'maxls': 50, 'maxfun': 10000})
            xs = res.x[0:n]
            ys = res.x[n:2*n]
            rs = res.x[2*n:3*n]
            # Clamp to valid range
            xs = np.clip(xs, 0, w)
            ys = np.clip(ys, 0, h)
            rs = np.clip(rs, 1e-4, min(w, h))
            score = compute_score(xs, ys, rs, w, h)
            if score > -np.inf:
                return score, np.column_stack([xs, ys, rs])
        except Exception:
            pass
        return -np.inf, None

    def force_based_refine(xs, ys, w, h, max_iters=500):
        """
        Use force-based relaxation to improve positions.
        """
        xs_curr = xs.copy()
        ys_curr = ys.copy()
        rs_curr = max_feasible_radii(xs_curr, ys_curr, w, h)
        score_curr = np.sum(rs_curr)
        
        best_xs = xs_curr.copy()
        best_ys = ys_curr.copy()
        best_score = score_curr
        
        step = 0.01 * min(w, h)
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            energy, fx, fy = soft_potential_energy(xs_curr, ys_curr, rs_curr, w, h)
            
            if energy < 1e-12:
                # No overlaps, try to increase radii by spreading
                break
            
            # Normalize forces
            f_mag = np.sqrt(fx**2 + fy**2)
            f_mag[f_mag < 1e-10] = 1e-10
            fx_norm = fx / f_mag
            fy_norm = fy / f_mag
            
            # Move in direction of force
            xs_curr += step * fx_norm
            ys_curr += step * fy_norm
            xs_curr = np.clip(xs_curr, 0, w)
            ys_curr = np.clip(ys_curr, 0, h)
            
            rs_new = max_feasible_radii(xs_curr, ys_curr, w, h)
            score_new = np.sum(rs_new)
            
            if score_new > best_score:
                best_score = score_new
                best_xs = xs_curr.copy()
                best_ys = ys_curr.copy()
                rs_curr = rs_new
                score_curr = score_new
            else:
                # Revert
                xs_curr = best_xs
                ys_curr = best_ys
                rs_curr = max_feasible_radii(xs_curr, ys_curr, w, h)
                score_curr = best_score
                step *= 0.9  # Reduce step size
        
        return best_xs, best_ys, best_score

    # Main search loop with diverse strategies
    # Focus on aspect ratios near 1 (square) since perimeter is fixed
    w_values = np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])
    seeds = list(range(1, 31))
    
    runs_done = 0
    max_runs = 60
    
    # Phase 1: Grid and random inits with gradient optimization
    for w in w_values:
        if time.time() > deadline - 10:
            break
        h = 2.0 - w
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            if runs_done >= max_runs:
                break

            # Choose init strategy
            if seed % 4 == 0:
                xs, ys, rs = grid_init(w, h, seed)
            elif seed % 4 == 1:
                xs, ys, rs = random_init(w, h, seed)
            elif seed % 4 == 2:
                xs, ys, rs = hex_init(w, h, seed)
            else:
                xs, ys, rs = random_init(w, h, seed + 1000)

            score, result = optimize_fixed_box(w, h, xs, ys)
            if score > best_score:
                best_score = score
                best_result = result.copy()
                best_w = w

            runs_done += 1

    # Phase 2: Simulated annealing on best aspect ratio
    if best_result is not None and time.time() < deadline - 15:
        w_sa = best_w
        h_sa = 2.0 - w_sa
        for seed in range(1, 11):
            if time.time() > deadline - 15:
                break
            
            # Use best result as starting point with perturbation
            np.random.seed(seed + 5000)
            xs_sa = best_result[:, 0] + np.random.normal(0, 0.05 * w_sa, N)
            ys_sa = best_result[:, 1] + np.random.normal(0, 0.05 * h_sa, N)
            xs_sa = np.clip(xs_sa, 0, w_sa)
            ys_sa = np.clip(ys_sa, 0, h_sa)
            
            xs_opt, ys_opt, sa_score = simulated_annealing(
                xs_sa, ys_sa, w_sa, h_sa, seed=seed, 
                max_iters=3000, t_start=0.03, t_end=0.0005
            )
            
            if sa_score > best_score:
                best_score = sa_score
                rs_opt = max_feasible_radii(xs_opt, ys_opt, w_sa, h_sa)
                best_result = np.column_stack([xs_opt, ys_opt, rs_opt])
                best_w = w_sa
    
    # Phase 3: Force-based refinement on best result
    if best_result is not None and time.time() < deadline - 5:
        w_fb = best_w
        h_fb = 2.0 - w_fb
        xs_fb = best_result[:, 0]
        ys_fb = best_result[:, 1]
        
        xs_ref, ys_ref, fb_score = force_based_refine(xs_fb, ys_fb, w_fb, h_fb, max_iters=300)
        
        if fb_score > best_score:
            best_score = fb_score
            rs_ref = max_feasible_radii(xs_ref, ys_ref, w_fb, h_fb)
            best_result = np.column_stack([xs_ref, ys_ref, rs_ref])
            best_w = w_fb
    
    # Phase 4: Final gradient polish
    if best_result is not None and time.time() < deadline - 3:
        w_final = best_w
        h_final = 2.0 - w_final
        xs_final = best_result[:, 0]
        ys_final = best_result[:, 1]
        
        score_final, result_final = optimize_fixed_box(w_final, h_final, xs_final, ys_final, penalty_weight=2000.0)
        if score_final > best_score:
            best_score = score_final
            best_result = result_final.copy()
            best_w = w_final

    if best_result is not None:
        return best_result
    else:
        # Fallback: simple grid
        w = 1.0
        h = 1.0
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side * w
                    cy = (j + 0.5) / side * h
                    r = 0.5 / side * min(w, h) * 0.95
                    cells.append([cx, cy, r])
        return np.array(cells[:N])
