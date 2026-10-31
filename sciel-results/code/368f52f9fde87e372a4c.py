"""Circle packing in perimeter-4 rectangle: SLSQP optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_value = 0.0
    best_result = None

    def check_validity(x, y, r, w, h, tol=1e-9):
        if w <= 0.05 or h <= 0.05:
            return False, 0.0
        if np.any(r <= 0):
            return False, 0.0
        if np.any(x < r - tol) or np.any(x > w - r + tol):
            return False, 0.0
        if np.any(y < r - tol) or np.any(y > h - r + tol):
            return False, 0.0
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy
        rs = r[:, None] + r[None, :]
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < rs[iu] ** 2 - tol):
            return False, 0.0
        return True, float(np.sum(r))

    def optimize_packing(w, h, seed):
        """
        Optimize circle positions and radii to maximize sum of radii.
        Variables: [x1, y1, r1, ..., xN, yN, rN]
        """
        rng = np.random.default_rng(seed)
        
        # Initial guess: grid-like or random
        inits = []
        
        # 1. Grid initialization
        side = int(np.ceil(np.sqrt(N)))
        for i in range(N):
            row = i // side
            col = i % side
            x0 = (col + 0.5) * w / side
            y0 = (row + 0.5) * h / side
            r0 = 0.5 * min(w/side, h/side) * 0.9
            inits.append([x0, y0, r0])
            
        # 2. Random initialization
        for _ in range(3):
            x0 = rng.uniform(0.2*w, 0.8*w, N)
            y0 = rng.uniform(0.2*h, 0.8*h, N)
            r0 = np.full(N, 0.05)
            inits.append(list(zip(x0, y0, r0)))

        best_local_val = 0.0
        best_local_z = None

        for init in inits:
            z0 = np.array([v for circle in init for v in circle])
            
            # Define objective: -sum(r)
            def objective(z):
                r = z[2::3]
                return -np.sum(r)

            # Define constraints
            def constraints(z):
                x = z[0::3]
                y = z[1::3]
                r = z[2::3]
                
                # Wall constraints
                c1 = x - r
                c2 = w - x - r
                c3 = y - r
                c4 = h - y - r
                
                # Pairwise constraints
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                d2 = dx*dx + dy*dy
                rs = r[:, None] + r[None, :]
                iu = np.triu_indices(N, k=1)
                c_pair = d2[iu] - rs[iu]**2
                
                # Combine all
                return np.concatenate([c1, c2, c3, c4, c_pair])

            # Bounds
            r_max = min(w, h) / 2.0
            bounds = []
            for i in range(N):
                bounds.append((0.0, w))
                bounds.append((0.0, h))
                bounds.append((1e-6, r_max))

            # Run minimization
            try:
                res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                               constraints=[{'type': 'ineq', 'fun': constraints}],
                               options={'maxiter': 200, 'ftol': 1e-9})
                
                if res.success:
                    z_opt = res.x
                    x_opt = z_opt[0::3]
                    y_opt = z_opt[1::3]
                    r_opt = z_opt[2::3]
                    
                    # Check validity
                    valid, val = check_validity(x_opt, y_opt, r_opt, w, h)
                    if valid and val > best_local_val:
                        best_local_val = val
                        best_local_z = (x_opt, y_opt, r_opt)
                        
            except Exception:
                pass

        return best_local_val, best_local_z

    # Sweep aspect ratios
    aspect_ratios = np.linspace(0.5, 1.5, 11)
    
    for k, w0 in enumerate(aspect_ratios):
        h0 = 2.0 - w0
        if h0 < 0.35 or w0 < 0.35:
            continue
        
        remaining = deadline - (time.time() - start_time)
        if remaining < 5.0:
            break
            
        seeds = [1000 + k, 2000 + k, 3000 + k]
        
        for seed in seeds:
            remaining = deadline - (time.time() - start_time)
            if remaining < 2.0:
                break
            
            try:
                val, result = optimize_packing(w0, h0, seed)
                if result is not None:
                    x, y, r = result
                    if val > best_value:
                        best_value = val
                        best_result = (x.copy(), y.copy(), r.copy(), w0, h0)
            except Exception:
                continue

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, w, h = best_result
    
    for _ in range(8):
        rt = r * (1 - 1e-9)
        if check_validity(x, y, rt, w, h)[0]:
            r = rt
        else:
            break

    return np.column_stack((x, y, r))
