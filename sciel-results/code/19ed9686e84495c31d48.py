import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    n = 8
    start_time = time.time()
    deadline = 145.0  # 5s margin

    # Pre-compute indices for upper triangle of an n x n matrix
    # We need indices i, j such that i < j
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx) # 28 pairs

    def objective(params):
        """
        params: 16 floats (x0, y0, x1, y1, ..., x7, y7)
        Returns value to be minimized.
        We want to maximize prod(d_ij) / D^28
        Log objective: 28 * ln(D) - sum(ln(d_ij))
        """
        pts = params.reshape(n, 2)
        
        # Compute pairwise squared distances
        # dists2[i, j] = || pts[i] - pts[j] ||^2
        # Using broadcasting: (pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2
        # Sum over last axis
        
        # Vectorized distance calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists2 = np.sum(diff * diff, axis=-1)
        
        # Extract unique pair distances (upper triangle)
        # dists2[i_idx, j_idx] gives the squared distances for the 28 pairs
        pair_dists2 = dists2[i_idx, j_idx]
        
        # Check for zero distances (coincident points)
        if np.any(pair_dists2 < 1e-10):
            return 1e10
            
        # Diameter is the max distance
        # D = sqrt(max(pair_dists2))
        # We can work with squared distances to avoid sqrt?
        # ln(D) = 0.5 * ln(D^2)
        # ln(d_ij) = 0.5 * ln(d_ij^2)
        
        max_dist2 = np.max(pair_dists2)
        
        # Objective: 28 * ln(D) - sum(ln(d_ij))
        # = 28 * 0.5 * ln(D^2) - sum(0.5 * ln(d_ij^2))
        # = 0.5 * (28 * ln(max_dist2) - sum(ln(pair_dists2)))
        
        # To avoid log(0) or negative, we already checked for zeros.
        # However, if max_dist2 is very small, ln is large negative, but 28*ln(max) might dominate?
        # If points collapse, max_dist2 -> 0, ln -> -inf, 28*ln -> -inf, -sum(ln) -> +inf?
        # Let's check the limit. If all points coincide, d->0.
        # It's safer to add a small epsilon or just rely on the penalty for < 1e-10.
        
        try:
            log_max = np.log(max_dist2)
            log_sum = np.sum(np.log(pair_dists2))
            val = 0.5 * (28.0 * log_max - log_sum)
            return val
        except Exception:
            return 1e10

    def evaluate_points(pts):
        """Calculate the actual score for reporting/verification."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
        D = np.max(dists)
        if D == 0:
            return 0.0
        scaled_dists = dists * (2.0 / D)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        prod = np.prod(scaled_dists[mask])
        return prod / (n ** n)

    # Regular octagon as a good initial guess / fallback
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    regular_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_params = regular_oct.flatten()
    best_score = evaluate_points(regular_oct)
    
    # Bounds for the coordinates. 
    # Since the problem is scale invariant, we can restrict the search space to a reasonable box.
    # If the points are too far apart, it's just a scaled version of something inside the box.
    # However, differential evolution needs bounds. 
    # Let's use [-2, 2] for each coordinate. This covers a diameter of up to ~4*sqrt(2) ~ 5.6, which is fine.
    bounds = [(-2.0, 2.0)] * (2 * n)

    # Strategy:
    # 1. Differential Evolution to find a good global minimum.
    # 2. Local refinement with L-BFGS-B.
    # 3. If time permits, try a second DE run with different settings.

    try:
        # Differential Evolution
        de_res = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=100, # Limit iterations to control time
            tol=1e-8,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7,
            init='latinhypercube'
        )
        
        current_params = de_res.x
        current_obj = de_res.fun
        
        # Check if this is better than regular octagon in terms of objective (lower is better)
        # We should also compare the actual score to be safe
        current_pts = current_params.reshape(n, 2)
        current_score = evaluate_points(current_pts)
        
        if current_score > best_score:
            best_score = current_score
            best_params = current_params
            print(f"DE Initial Score: {best_score:.6f}")

    except Exception as e:
        print(f"DE failed: {e}")

    # Local Refinement
    if time.time() < deadline - 5:
        try:
            local_res = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            if local_res.success:
                local_pts = local_res.x.reshape(n, 2)
                local_score = evaluate_points(local_pts)
                
                if local_score > best_score:
                    best_score = local_score
                    best_params = local_res.x
                    print(f"Local Refinement Score: {best_score:.6f}")
        except Exception as e:
            print(f"Local refinement failed: {e}")

    # If we have time, try another global search with different parameters
    if time.time() < deadline - 15:
        try:
            de_res2 = differential_evolution(
                objective,
                bounds=bounds,
                seed=123,
                maxiter=50,
                tol=1e-8,
                popsize=10,
                mutation=(0.4, 1.2),
                recombination=0.6,
                init='soo' # Or just default
            )
            
            if de_res2.success:
                pts2 = de_res2.x.reshape(n, 2)
                score2 = evaluate_points(pts2)
                
                if score2 > best_score:
                    best_score = score2
                    best_params = de_res2.x
                    print(f"DE 2nd Attempt Score: {best_score:.6f}")
                    
                    # Polish this new best
                    if time.time() < deadline - 2:
                        try:
                            local_res2 = minimize(
                                objective,
                                best_params,
                                method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 100, 'ftol': 1e-15}
                            )
                            if local_res2.success:
                                pts3 = local_res2.x.reshape(n, 2)
                                score3 = evaluate_points(pts3)
                                if score3 > best_score:
                                    best_score = score3
                                    best_params = local_res2.x
                                    print(f"Polish 2nd Attempt Score: {best_score:.6f}")
                        except Exception:
                            pass
        except Exception as e:
            print(f"DE 2nd attempt failed: {e}")

    # Final Polish on the absolute best found so far
    if time.time() < deadline - 1:
        try:
            final_res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if final_res.success:
                final_pts = final_res.x.reshape(n, 2)
                final_score = evaluate_points(final_pts)
                if final_score > best_score:
                    best_score = final_score
                    best_params = final_res.x
        except Exception:
            pass

    final_pts = best_params.reshape(n, 2)
    
    # Ensure distinctness and bounds
    # Scale to ensure coordinates are within 10^6 (should be trivially satisfied)
    # And ensure no points are too close
    dists = np.sqrt(np.sum((final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :])**2, axis=-1))
    min_dist = np.min(dists[i_idx, j_idx])
    if min_dist < 1e-6:
        # Fallback to regular octagon if degeneracy occurred
        final_pts = regular_oct
        best_score = 1.0
    else:
        best_score = evaluate_points(final_pts)

    return final_pts.tolist()
