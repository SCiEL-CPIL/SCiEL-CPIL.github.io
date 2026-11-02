"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    def compute_overlap_penalty(xs, ys, rs, beta=10.0):
        """Smooth overlap penalty using soft-min surrogate."""
        n = len(rs)
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                min_dist = rs[i] + rs[j]
                # Smooth approximation of max(0, min_dist - dist)
                # Using softplus-like function: (1/beta)*log(1 + exp(beta*(min_dist - dist)))
                val = min_dist - dist
                if val > 1.0 / beta:
                    penalty += val
                else:
                    penalty += np.log1p(np.exp(beta * val)) / beta
        return penalty

    def check_and_update(x, w, h):
        nonlocal best_value, best_result
        if time.time() > deadline:
            return
        xs = x[0::3]
        ys = x[1::3]
        rs = x[2::3]
        # Check bounds
        if np.any(rs < 1e-9):
            return
        if np.any(xs - rs < -1e-6) or np.any(xs + rs > w + 1e-6):
            return
        if np.any(ys - rs < -1e-6) or np.any(ys + rs > h + 1e-6):
            return
        # Check disjointness
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                min_dist = rs[i] + rs[j]
                if dist_sq < (min_dist - 1e-7) ** 2:
                    return
        val = np.sum(rs)
        if val > best_value:
            best_value = val
            best_result = x.copy()

    def objective(params, w, h, beta=10.0):
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        if np.any(rs <= 0):
            return 1e10 + np.sum(np.minimum(rs, 0) ** 2)
        penalty = compute_overlap_penalty(xs, ys, rs, beta)
        # Also add boundary violation penalty
        boundary_pen = 0.0
        for i in range(N):
            if xs[i] < rs[i]:
                boundary_pen += (rs[i] - xs[i]) ** 2
            if xs[i] > w - rs[i]:
                boundary_pen += (xs[i] - (w - rs[i])) ** 2
            if ys[i] < rs[i]:
                boundary_pen += (rs[i] - ys[i]) ** 2
            if ys[i] > h - rs[i]:
                boundary_pen += (ys[i] - (h - rs[i])) ** 2
        return penalty + boundary_pen - np.sum(rs)

    def sa_optimize(w, h, x0, max_iters=2000, seed=42):
        """Simulated annealing for fixed w, h."""
        xs = x0[0::3].copy()
        ys = x0[1::3].copy()
        rs = x0[2::3].copy()
        
        r_max = min(w, h) / 2.0
        T = 0.05
        cooling = 0.999
        current_cost = compute_overlap_penalty(xs, ys, rs) - np.sum(rs)
        
        rng = np.random.default_rng(seed)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            # Pick a random circle and perturb
            idx = rng.integers(0, N)
            old_x, old_y, old_r = xs[idx], ys[idx], rs[idx]
            
            # Perturb position and radius
            scale = T * 0.3
            xs[idx] += rng.normal(0, scale)
            ys[idx] += rng.normal(0, scale)
            rs[idx] += rng.normal(0, scale * 0.5)
            
            # Clamp to bounds
            xs[idx] = np.clip(xs[idx], rs[idx], w - rs[idx])
            ys[idx] = np.clip(ys[idx], rs[idx], h - rs[idx])
            rs[idx] = np.clip(rs[idx], 1e-4, r_max)
            
            new_cost = compute_overlap_penalty(xs, ys, rs) - np.sum(rs)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_cost = new_cost
            else:
                xs[idx], ys[idx], rs[idx] = old_x, old_y, old_r
            
            T *= cooling
        
        x_final = np.zeros(3 * N)
        x_final[0::3] = xs
        x_final[1::3] = ys
        x_final[2::3] = rs
        return x_final

    # Try multiple aspect ratios
    widths = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5]
    
    best_x_for_warm = None
    
    for w in widths:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Initialize with grid
        a = int(np.ceil(np.sqrt(N * w / h)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
            
        x0 = np.zeros(3 * N)
        for i in range(N):
            col = i % a
            row = i // a
            x0[3 * i] = (col + 0.5) * w / a
            x0[3 * i + 1] = (row + 0.5) * h / b
            x0[3 * i + 2] = min(w / a, h / b) * 0.45
        
        # L-BFGS-B optimization with smooth objective
        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-4, r_max))
        
        try:
            # First pass with moderate beta
            res = minimize(lambda p: objective(p, w, h, beta=5.0), x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            if time.time() > deadline:
                break
            check_and_update(res.x, w, h)
            
            # Second pass with higher beta for more precise optimization
            if time.time() < deadline - 10:
                res2 = minimize(lambda p: objective(p, w, h, beta=20.0), res.x, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10})
                check_and_update(res2.x, w, h)
                if best_x_for_warm is None or np.sum(res2.x[2::3]) > np.sum(best_x_for_warm[2::3]):
                    best_x_for_warm = res2.x.copy()
        except Exception:
            pass
        
        # Simulated annealing refinement
        if time.time() < deadline - 15:
            # Use best result so far as warm start if available
            x_sa_start = best_x_for_warm if best_x_for_warm is not None else res.x
            x_sa = sa_optimize(w, h, x_sa_start, max_iters=1500, seed=42)
            check_and_update(x_sa, w, h)
            
            # Second SA pass with different seed
            if time.time() < deadline - 10:
                rng_sa = np.random.default_rng(123)
                x_sa2 = x_sa.copy()
                # Add small perturbation
                for i in range(N):
                    x_sa2[3*i] += rng_sa.normal(0, 0.01)
                    x_sa2[3*i+1] += rng_sa.normal(0, 0.01)
                    x_sa2[3*i+2] += rng_sa.normal(0, 0.005)
                x_sa2 = sa_optimize(w, h, x_sa2, max_iters=1000, seed=123)
                check_and_update(x_sa2, w, h)

    # Fallback if no valid result found
    if best_result is None:
        a, b = 7, 3
        w, h = 1.4, 0.6
        r = 0.1 - 1e-4
        best_result = np.zeros(3 * N)
        for i in range(a):
            for j in range(b):
                idx = i * b + j
                best_result[3 * idx] = (i + 0.5) * w / a
                best_result[3 * idx + 1] = (j + 0.5) * h / b
                best_result[3 * idx + 2] = r
    
    # Final high-precision polish pass
    if time.time() < deadline - 5 and best_result is not None:
        xs = best_result[0::3].copy()
        ys = best_result[1::3].copy()
        rs = best_result[2::3].copy()
        
        # Try to polish with very tight tolerance
        try:
            # Determine w, h from current best
            # We need to find which aspect ratio gave the best result
            # For simplicity, try a few common ones
            for test_w in [0.8, 0.9, 1.0, 1.1, 1.2]:
                test_h = 2.0 - test_w
                # Check if current result fits in this rectangle
                if np.all(xs - rs >= -1e-6) and np.all(xs + rs <= test_w + 1e-6) and \
                   np.all(ys - rs >= -1e-6) and np.all(ys + rs <= test_h + 1e-6):
                    # Polish with this aspect ratio
                    bounds = []
                    r_max = min(test_w, test_h) / 2.0
                    for i in range(N):
                        bounds.append((0, test_w))
                        bounds.append((0, test_h))
                        bounds.append((1e-4, r_max))
                    
                    # Very tight optimization
                    polish_res = minimize(lambda p: objective(p, test_w, test_h, beta=50.0), 
                                         best_result, method='L-BFGS-B', bounds=bounds,
                                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
                    check_and_update(polish_res.x, test_w, test_h)
                    break
        except Exception:
            pass
    
    # Format output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = best_result[3 * i:3 * i + 3]
    
    return result
# EVOLVE-BLOCK-END
