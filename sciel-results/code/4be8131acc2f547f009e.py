"""Circle packing in perimeter-4 rectangle using unified penalty-based NLP."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    
    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6
        
    obj = -np.sum(r)
    
    # Vectorized pairwise overlap penalty
    # Compute all pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]
    
    # Only upper triangle (i < j)
    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]
    
    # Smooth penalty: use max(0, overlap)^2 but with smooth approximation
    # overlap = r_sum - sqrt(d_sq) when d_sq < r_sum^2
    # Use soft version: penalty = (r_sum^2 - d_sq)^2 / (4 * r_sum) when positive
    # This is smooth and approximates the overlap penalty
    diff = r_sum_pairs * r_sum_pairs - d_sq_pairs
    # Only penalize when diff > 0 (overlap)
    # Use softplus-like: max(diff, 0) but smooth
    # penalty = penalty_weight * (max(diff,0))^2 / (4 * r_sum^2)
    # Simpler: just use diff where positive
    pos_diff = np.maximum(diff, 0.0)
    overlap_penalty = penalty_weight * np.sum(pos_diff * pos_diff) / (4.0 * np.sum(r_sum_pairs * r_sum_pairs) + 1e-10)
    
    # Actually, let's use the direct overlap formula but vectorized
    # overlap = r_sum - sqrt(d_sq) when d_sq < r_sum^2
    # penalty = sum(overlap^2)
    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)
    
    obj += overlap_penalty
    
    # Vectorized boundary penalty
    # Left: x_i - r_i >= 0
    left_viol = np.maximum(r - x, 0.0)
    # Right: w - x_i - r_i >= 0
    right_viol = np.maximum(x + r - w, 0.0)
    # Bottom: y_i - r_i >= 0
    bottom_viol = np.maximum(r - y, 0.0)
    # Top: h - y_i - r_i >= 0
    top_viol = np.maximum(y + r - h, 0.0)
    
    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )
    
    obj += boundary_penalty
    
    return obj

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    best_value = 0.0
    best_result = None
    
    # Parameters for penalty method
    penalty_weight = 1000.0
    
    # Define the unified parameter space objective function
    # z = [x1, y1, r1, ..., xN, yN, rN, w]
    # h = 2.0 - w
    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w
        
        if w <= 0.1 or h <= 0.1:
            return False, 0.0
            
        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d_sq < r_sum * r_sum - tol:
                    return False, 0.0
        
        val = float(np.sum(r))
        return True, val

    def force_relax(z, max_iter=500):
        """
        Force-directed relaxation to resolve overlaps with adaptive step size.
        Operates on unified vector z.
        """
        z = z.copy().astype(float)
        n = N
        dt = 0.005
        min_dt = 0.0001
        convergence_tol = 1e-8
        
        for it in range(max_iter):
            x = z[0::3][:n]
            y = z[1::3][:n]
            r = z[2::3][:n]
            w = z[-1]
            h = 2.0 - w
            
            if w < 0.1 or h < 0.1:
                break
                
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Wall repulsion
            for i in range(n):
                if x[i] - r[i] < 0:
                    fx[i] += (x[i] - r[i]) * 100
                if x[i] + r[i] > w:
                    fx[i] -= (x[i] + r[i] - w) * 100
                if y[i] - r[i] < 0:
                    fy[i] += (y[i] - r[i]) * 100
                if y[i] + r[i] > h:
                    fy[i] -= (y[i] + r[i] - h) * 100
            
            # Pairwise repulsion
            max_overlap = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d_sq = dx * dx + dy * dy
                    min_d = r[i] + r[j]
                    if d_sq < min_d * min_d and d_sq > 1e-12:
                        d = np.sqrt(d_sq)
                        overlap = min_d - d
                        max_overlap = max(max_overlap, overlap)
                        # Force proportional to overlap
                        f = overlap * 15
                        fx[i] += (dx / d) * f
                        fy[i] += (dy / d) * f
                        fx[j] -= (dx / d) * f
                        fy[j] -= (dy / d) * f
            
            # Check convergence
            if max_overlap < convergence_tol:
                break
            
            # Adaptive step size: reduce if forces are large
            force_mag = np.max(np.sqrt(fx * fx + fy * fy))
            if force_mag > 1.0:
                dt = max(min_dt, dt * 0.5)
            elif force_mag < 0.1:
                dt = min(0.01, dt * 1.1)
            
            # Update positions
            z[0::3][:n] += dt * fx
            z[1::3][:n] += dt * fy
            
            # Clamp positions to stay within bounds (hard constraint for stability)
            for i in range(n):
                z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
                z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])
                
        return z

    def generate_initial_guesses():
        guesses = []
        rng = np.random.default_rng(42)
        
        # 1. Grid-based starts with varying aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        side = int(np.ceil(np.sqrt(N)))
        
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                # Initial radius: slightly less than cell size
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
            
        # 2. Random starts
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                # Place randomly within bounds
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
            
        return guesses

    def local_refine(z_init, time_budget):
        """
        Refine a valid solution using SLSQP with hard constraints.
        This ensures the final result is strictly valid and maximizes sum(r).
        """
        if time_budget < 2:
            return z_init
            
        n = N
        z = z_init.copy()
        
        # Hard constraints for SLSQP
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            # x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx] - p[3*idx+2]})
            # w - x_i - r_i >= 0  (w is p[-1])
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[-1] - p[3*idx] - p[3*idx+2]})
            # y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx+1] - p[3*idx+2]})
            # h - y_i - r_i >= 0  (h = 2 - w)
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: (2.0 - p[-1]) - p[3*idx+1] - p[3*idx+2]})
            
        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, idx1=i, idx2=j: (
                        (p[3*idx1] - p[3*idx2])**2 + 
                        (p[3*idx1+1] - p[3*idx2+1])**2 - 
                        (p[3*idx1+2] + p[3*idx2+2])**2
                    )
                })
                
        bounds = []
        for i in range(n):
            bounds.append((0, 2)) # x
            bounds.append((0, 2)) # y
            bounds.append((0.01, 0.5)) # r
        bounds.append((0.3, 1.7)) # w
        
        # Objective: minimize -sum(r)
        def obj_hard(p):
            r = p[2::3][:n]
            return -np.sum(r)
            
        try:
            maxiter = int(min(300, time_budget * 30))
            res = minimize(obj_hard, z, method='SLSQP', bounds=bounds, 
                           constraints=constraints, 
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})
            if res.success or res.fun < 1e3:
                return res.x
        except Exception:
            pass
        return z

    def high_precision_polish(z_init, time_budget):
        """
        High-precision polish using multiple SLSQP passes with decreasing tolerances.
        """
        if time_budget < 3:
            return z_init
            
        z = z_init.copy()
        n = N
        
        constraints = []
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[-1] - p[3*idx] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx+1] - p[3*idx+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: (2.0 - p[-1]) - p[3*idx+1] - p[3*idx+2]})
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, idx1=i, idx2=j: (
                        (p[3*idx1] - p[3*idx2])**2 + 
                        (p[3*idx1+1] - p[3*idx2+1])**2 - 
                        (p[3*idx1+2] + p[3*idx2+2])**2
                    )
                })
                
        bounds = []
        for i in range(n):
            bounds.append((0, 2))
            bounds.append((0, 2))
            bounds.append((0.01, 0.5))
        bounds.append((0.3, 1.7))
        
        def obj_hard(p):
            r = p[2::3][:n]
            return -np.sum(r)
        
        # Multiple passes with decreasing tolerances
        tolerances = [1e-10, 1e-12, 1e-14]
        for ftol in tolerances:
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(obj_hard, z, method='SLSQP', bounds=bounds,
                               constraints=constraints,
                               options={'maxiter': 500, 'ftol': ftol, 'maxls': 200})
                if res.success:
                    z = res.x
            except Exception:
                break
        
        return z

    # Main Loop
    guesses = generate_initial_guesses()
    
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        # Relax initial overlaps
        z_relaxed = force_relax(z0)
        
        # Global optimization with penalty
        time_left = deadline - time.time()
        if time_left < 3:
            continue
            
        maxiter = int(min(400, time_left * 30))
        
        try:
            res = minimize(objective, z_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 100})
            
            z_opt = res.x
            
            # Check validity
            valid, val = check_validity(z_opt)
            
            if valid and val > best_value:
                best_value = val
                best_result = z_opt
            else:
                # If invalid, try to repair by shrinking radii
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair
                else:
                    # Try more aggressive repair
                    z_repair2 = z_opt.copy()
                    z_repair2[r_idx] *= 0.90
                    valid, val = check_validity(z_repair2)
                    if valid and val > best_value:
                        best_value = val
                        best_result = z_repair2
                        
        except Exception:
            continue

    # Final Refinement
    if best_result is not None:
        time_left = deadline - time.time()
        if time_left > 5:
            # First pass: standard local refinement
            refined = local_refine(best_result, time_left)
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined
            
            # Second pass: high-precision polish
            time_left2 = deadline - time.time()
            if time_left2 > 3:
                polished = high_precision_polish(best_result, time_left2)
                valid, val = check_validity(polished)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished

    if best_result is None:
        # Fallback: simple grid in unit square (w=1, h=1)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    
    # Safety shrink to ensure validity against floating point precision in evaluator
    r = r * (1 - 1e-7)
    
    return np.column_stack((x, y, r))

if __name__ == "__main__":
    res = construct()
    print(f"Sum of radii: {np.sum(res[:, 2])}")
