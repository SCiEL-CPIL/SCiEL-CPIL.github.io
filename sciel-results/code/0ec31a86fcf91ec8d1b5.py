# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_union_length(x, y_vals, a, n):
    """
    Compute the union length of the triangles at all heights y_vals.
    x: shape (n,)
    y_vals: shape (M,)
    a: shape (n,)
    n: int
    Returns: shape (M,)
    """
    # L_i(y) = x_i + a_i * y
    # Shape: (M, n)
    L = x[None, :] + a[None, :] * y_vals[:, None]
    
    # Sort L along axis 1
    # np.sort is O(M * n * log(n))
    L_sorted = np.sort(L, axis=1)
    
    # Width at height y is w(y) = (1-y)/n
    w = (1.0 - y_vals) / n  # Shape: (M,)
    
    # Gaps between consecutive sorted L's
    # Shape: (M, n-1)
    gaps = np.diff(L_sorted, axis=1)
    
    # Union length = w + sum(min(gap, w))
    # We need to sum min(gaps, w) along axis 1
    # min(gaps, w[:, None]) -> Shape (M, n-1)
    effective_gaps = np.minimum(gaps, w[:, None])
    
    # Sum along axis 1
    # Shape: (M,)
    sum_gaps = np.sum(effective_gaps, axis=1)
    
    return w + sum_gaps

def _area(x, y_vals, a, n):
    """
    Compute the area of the union of the triangles.
    """
    union_lens = _compute_union_length(x, y_vals, a, n)
    return np.mean(union_lens)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = np.array([(i + 1) / n for i in range(n)], dtype=np.float64)
    y_vals = np.array([(m + 0.5) / 20000 for m in range(20000)], dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    # Initial guess: Keich construction or simple heuristic
    # A good initial guess is to spread them out or use the Keich construction.
    # Keich: x_i = sum_{j=1}^{log2(n)} eps_j(i) * (1-j)/log2(n) * 2^{-j}
    # This puts them in [0, 1).
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    best_x = x0.copy()
    best_area = _area(best_x, y_vals, a, n)
    
    # Strategy: Coordinate Descent with Critical Point Sampling
    # For each i, find the best x_i by sampling critical points.
    # Critical points occur when L_i(y) = L_j(y) +/- w(y) for some y.
    # L_i(y) = x_i + a_i y
    # L_j(y) = x_j + a_j y
    # w(y) = (1-y)/n
    # x_i + a_i y = x_j + a_j y + (1-y)/n
    # x_i = x_j + (a_j - a_i) y + (1-y)/n
    # x_i = x_j + (a_j - a_i - 1/n) y + 1/n
    
    # Since y is continuous, the "critical" x_i values for a fixed j are a range.
    # However, the objective is piecewise linear in x_i. The minimum will be at a breakpoint.
    # Breakpoints happen when the order of L's changes or when a gap becomes exactly w.
    
    # A simpler and faster heuristic:
    # For each i, try a set of candidate x_i values:
    # 1. Current best_x[i]
    # 2. Values derived from alignment with other j's at specific y's (e.g., y=0, y=1, y=0.5)
    # 3. Random perturbations
    
    # To be fast, we limit the number of candidates and iterations.
    
    max_cd_iters = 20
    for iter_cd in range(max_cd_iters):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
                
            current_xi = best_x[i]
            best_local_xi = current_xi
            best_local_area = best_area
            
            # Generate candidates for x_i
            candidates = set()
            candidates.add(current_xi)
            
            # Add candidates based on alignment with other triangles at y=0 and y=1
            # At y=0: L_i = x_i, L_j = x_j. w = 1/n.
            # Critical: x_i = x_j +/- 1/n
            # At y=1: L_i = x_i + a_i, L_j = x_j + a_j. w = 0.
            # Critical: x_i + a_i = x_j + a_j => x_i = x_j + a_j - a_i
            
            for j in range(n):
                if j == i:
                    continue
                
                # y=0 alignment
                candidates.add(best_x[j] - 1.0/n)
                candidates.add(best_x[j] + 1.0/n)
                
                # y=1 alignment
                candidates.add(best_x[j] + a[j] - a[i])
                
                # Midpoint alignment y=0.5
                # x_i + a_i*0.5 = x_j + a_j*0.5 + 0.5/n
                # x_i = x_j + (a_j - a_i)*0.5 + 0.5/n
                candidates.add(best_x[j] + (a[j] - a[i]) * 0.5 + 0.5/n)
                candidates.add(best_x[j] + (a[j] - a[i]) * 0.5 - 0.5/n)
                
                # Small perturbations around current
                candidates.add(current_xi - 0.01)
                candidates.add(current_xi + 0.01)
                candidates.add(current_xi - 0.001)
                candidates.add(current_xi + 0.001)
            
            # Evaluate candidates
            # To speed up, we can batch evaluate if we had a smarter vectorization,
            # but for n=64 and ~200 candidates, it's 200 * 64 * 20000 ops ~ 256M ops.
            # This might be too slow if done for all 64 i's in one pass (16k evals).
            # 16k evals * 256M ops/eval is too much.
            
            # We need to be smarter.
            # Instead of checking all candidates for all i in one pass,
            # let's just do a few steps of "Best Improvement" or "First Improvement".
            
            # Let's limit candidates to the most promising ones:
            # 1. Current
            # 2. Best from previous iteration's neighbors?
            
            # Actually, let's just do a simple gradient-free local search (Powell-like)
            # or just a few random restarts with Nelder-Mead but with VERY strict limits.
            
            # Given the time constraint, let's stick to a simple local search.
            # We will try to optimize x_i by checking a small grid around current_xi.
            # Grid size: 10 points in [-0.05, 0.05] around current_xi.
            
            x_grid = np.linspace(current_xi - 0.05, current_xi + 0.05, 10)
            
            min_area_local = best_local_area
            best_xi_local = current_xi
            
            for xi_candidate in x_grid:
                if time.time() > deadline:
                    break
                trial_x = best_x.copy()
                trial_x[i] = xi_candidate
                area_val = _area(trial_x, y_vals, a, n)
                if area_val < min_area_local:
                    min_area_local = area_val
                    best_xi_local = xi_candidate
            
            if best_xi_local != current_xi and min_area_local < best_area:
                best_x[i] = best_xi_local
                best_area = min_area_local
                improved = True
                
    # If we have time, do a final polish with Nelder-Mead
    if time.time() < deadline - 60:
        try:
            from scipy.optimize import minimize
            # Use a very tight tolerance and low maxiter
            res = minimize(
                _area, best_x, args=(y_vals, a, n),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < best_area and np.all(np.isfinite(res.x)):
                best_x = res.x
                best_area = res.fun
        except Exception:
            pass

    # Ensure finite
    if not np.all(np.isfinite(best_x)):
        best_x = x0
        
    return best_x

# EVOLVE-BLOCK-END
