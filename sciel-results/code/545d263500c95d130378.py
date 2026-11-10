import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    N = 8
    w = 1.0 / N
    slopes = np.arange(N, dtype=np.float64) / N

    def get_union_info(c, z):
        """
        Returns length, and for each interval i, whether it is the leftmost start 
        or rightmost end of its connected component.
        """
        starts = z + c * slopes
        ends = starts + w
        order = np.argsort(starts, kind="stable")
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        length = 0.0
        current_end = -np.inf
        comp_start_idx = -1
        comp_end_idx = -1
        
        is_leftmost = np.zeros(N, dtype=bool)
        is_rightmost = np.zeros(N, dtype=bool)
        
        for idx in range(N):
            s = s_sorted[idx]
            e = e_sorted[idx]
            orig_idx = order[idx]
            
            if s > current_end:
                # New component
                if comp_start_idx != -1:
                    # Close previous component
                    is_leftmost[comp_start_idx] = True
                    is_rightmost[comp_end_idx] = True
                    length += (ends[comp_end_idx] - starts[comp_start_idx])
                comp_start_idx = orig_idx
                comp_end_idx = orig_idx
                current_end = e
            else:
                # Overlapping
                if e > current_end:
                    # Extend current component
                    is_rightmost[comp_end_idx] = False # The previous rightmost is no longer the rightmost
                    comp_end_idx = orig_idx
                    current_end = e
                    # Note: is_leftmost for comp_start_idx remains True for now
                # If e <= current_end, the interval is completely inside, no change to boundaries
        
        if comp_start_idx != -1:
            is_leftmost[comp_start_idx] = True
            is_rightmost[comp_end_idx] = True
            length += (ends[comp_end_idx] - starts[comp_start_idx])
            
        return length, is_leftmost, is_rightmost

    def get_criticals(z):
        crit = [0.0, 1.0]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                denom = i - j
                # s_i = s_j
                c1 = N * (z[j] - z[i]) / denom
                if 0.0 < c1 < 1.0:
                    crit.append(c1)
                # s_i = e_j
                c2 = N * (z[j] - z[i] + w) / denom
                if 0.0 < c2 < 1.0:
                    crit.append(c2)
                # e_i = s_j
                c3 = N * (z[j] - z[i] - w) / denom
                if 0.0 < c3 < 1.0:
                    crit.append(c3)
                # e_i = e_j (same as s_i = s_j)
        return sorted(set(crit))

    def compute_cost_and_grad(z):
        if not np.all(np.isfinite(z)):
            return 1e10, np.zeros(2*N)
        
        x = z[:N]
        y = z[N:]
        
        crit_x = get_criticals(x)
        crit_y = get_criticals(y)
        crit = sorted(set(crit_x + crit_y))
        
        total_cost = 0.0
        grad = np.zeros(2*N, dtype=np.float64)
        
        for k in range(len(crit) - 1):
            c0 = crit[k]
            c1 = crit[k+1]
            h = c1 - c0
            if h < 1e-12:
                continue
            
            Lx0, lmx0, rmx0 = get_union_info(c0, x)
            Ly0, lmy0, rmy0 = get_union_info(c0, y)
            Lx1, lmx1, rmx1 = get_union_info(c1, x)
            Ly1, lmy1, rmy1 = get_union_info(c1, y)
            
            # Lx(c) = Ax*c + Bx
            # Ly(c) = Ay*c + By
            Ax = (Lx1 - Lx0) / h
            Bx = Lx0 - Ax * c0
            Ay = (Ly1 - Ly0) / h
            By = Ly0 - Ay * c0
            
            # Integrate (Ax*c + Bx)(Ay*c + By) = Ax*Ay*c^2 + (Ax*By + Ay*Bx)*c + Bx*By
            I0 = h
            I1 = (c1**2 - c0**2) / 2.0
            I2 = (c1**3 - c0**3) / 3.0
            
            total_cost += Ax*Ay*I2 + (Ax*By + Ay*Bx)*I1 + Bx*By*I0
            
            # Gradient
            # d/dx_i Lx(c) = rmx_i(c) - lmx_i(c)
            # d/dy_j Ly(c) = rmy_j(c) - lmy_j(c)
            
            # We need the integral of dLx/dx_i * Ly(c) dc
            # dLx/dx_i is 0, 1, or -1 in the interval?
            # Actually, the topology is fixed, so rmx and lmx are constant in [c0, c1] (almost everywhere).
            # We can just use the values at c0 (or c1, they are the same).
            
            dLx_dx = rmx0 - lmx0
            dLy_dy = rmy0 - lmy0
            
            # grad_x_i += integral of dLx_dx_i * (Ay*c + By) dc
            # = dLx_dx_i * (Ay*I1 + By*I0)
            
            grad[:N] += dLx_dx * (Ay*I1 + By*I0)
            grad[N:] += dLy_dy * (Ax*I1 + Bx*I0)
            
        return total_cost, grad

    def compute_cost(z):
        c, _ = compute_cost_and_grad(z)
        return c

    # Initial guess: Keich
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    z0 = np.concatenate([keich, keich])
    
    best_z = z0.copy()
    best_cost = compute_cost(z0)
    
    rng = np.random.default_rng(42)
    inits = [z0.copy()]
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        if best_cost < 1e-2:
            z_new = best_z + rng.normal(0, 0.01, 2*N)
        else:
            z_new = rng.uniform(-0.5, 0.5, 2*N)
        inits.append(z_new)
        
    for z_init in inits:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(compute_cost, z_init, jac=lambda z: compute_cost_and_grad(z)[1], method="L-BFGS-B",
                           options={"maxiter": 4000, "gtol": 1e-12, "ftol": 1e-14, "maxls": 40, "maxcor": 20})
            if np.all(np.isfinite(res.x)) and res.fun < best_cost:
                best_cost = res.fun
                best_z = res.x.copy()
        except Exception:
            pass

    x = best_z[:N]
    y = best_z[N:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x[i_flat]
    base_y = y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    if not np.all(np.isfinite(positions)):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions
