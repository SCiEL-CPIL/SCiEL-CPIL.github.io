import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    # Hard deadline with safety margin. The evaluator kills at 150s.
    # We aim to finish by 145s to be safe.
    deadline = start_time + 145.0
    
    n = 8
    # Number of unordered pairs
    num_pairs = n * (n - 1) // 2

    def exact_log_score(points):
        """
        Computes the log of the score for a set of points.
        Score = 16 * prod(d_ij) / D^28
        log_score = ln(16) + sum(ln(d_ij)) - 28 * ln(D)
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Use upper triangle to avoid double counting and diagonal
        i_idx, j_idx = np.triu_indices(n, k=1)
        dists = np.sqrt(dists_sq[i_idx, j_idx])
        
        if np.any(dists < 1e-12):
            return -np.inf
            
        D = np.max(dists)
        if D < 1e-12:
            return -np.inf
            
        # log(16) + sum(log(dists)) - 28 * log(D)
        log_prod = np.sum(np.log(dists))
        return np.log(16.0) + log_prod - 28.0 * np.log(D)

    def soft_max_log_score(points, beta):
        """
        Smooth approximation of the objective using soft-max for the diameter.
        Minimizes -log_score.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        i_idx, j_idx = np.triu_indices(n, k=1)
        dists = np.sqrt(dists_sq[i_idx, j_idx])
        
        if np.any(dists < 1e-12):
            return np.inf
            
        # Soft max approximation of log(D)
        # log(D) approx log(sum(exp(d/beta))) - log(num_pairs) ? No, standard log-sum-exp is for sum.
        # We want to approximate max(d). 
        # log(sum(exp(d/beta))) * beta approx max(d) + beta * log(num_terms) ?
        # Actually, max(d) = lim_{beta->0} beta * log(sum(exp(d/beta))).
        # So log(max(d)) approx log(beta) + log(sum(exp(d/beta))).
        
        scaled_d = dists / beta
        m = np.max(scaled_d)
        log_sum_exp = m + np.log(np.sum(np.exp(scaled_d - m)))
        
        # This approximates log(D)
        log_D_approx = np.log(beta) + log_sum_exp
        
        log_prod = np.sum(np.log(dists))
        
        # We want to maximize log_score = log(16) + log_prod - 28 * log_D
        # So we minimize -log_score = -log(16) - log_prod + 28 * log_D
        return -np.log(16.0) - log_prod + 28.0 * log_D_approx

    def try_update(points):
        nonlocal best_score, best_points
        if points is None or np.any(np.isnan(points)):
            return
        # Check distinctness
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_d = np.sqrt(np.min(dists_sq))
        if min_d < 1e-8:
            return
            
        s = exact_log_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Initialize with regular octagon
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = exact_log_score(reg_oct)
    best_points = reg_oct.copy()

    # Generate diverse starting points
    starts = [reg_oct.flatten()]
    
    # 1. Two squares with relative rotation
    # The regular octagon is two squares rotated by pi/4.
    # Optimal might be two squares with slightly different radii or rotation.
    for rot in np.linspace(0.0, 0.5, 20):
        a1 = 2 * np.pi * np.arange(4) / 4
        a2 = 2 * np.pi * (np.arange(4) / 4 + rot)
        r1 = 1.0
        r2 = 1.0
        sq1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
        sq2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        starts.append(np.vstack([sq1, sq2]).flatten())
        
    # 2. Antipodal pairs (center symmetry)
    # Points: (r_i cos theta_i, r_i sin theta_i) and (-r_i cos theta_i, -r_i sin theta_i)
    # This fixes the diameter to be 2*max(r_i).
    np.random.seed(42)
    for _ in range(20):
        thetas = np.sort(np.random.uniform(0, np.pi, 4))
        radii = np.random.uniform(0.5, 1.5, 4)
        pts1 = np.column_stack([radii * np.cos(thetas), radii * np.sin(thetas)])
        pts2 = -pts1
        starts.append(np.vstack([pts1, pts2]).flatten())

    # 3. Random perturbations of octagon
    for _ in range(20):
        perturb = np.random.normal(0, 0.1, (n, 2))
        starts.append((reg_oct + perturb).flatten())

    # 4. Pure random
    for _ in range(10):
        pts = np.random.uniform(-1.5, 1.5, (n, 2))
        starts.append(pts.flatten())

    # Phase 1: Local optimization on diverse starts using smooth surrogate
    # Use a moderate beta for smoothness
    beta_smooth = 10.0
    
    for s in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                lambda x, b=beta_smooth: soft_max_log_score(x.reshape(n, 2), b),
                s,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success or res.fun < np.inf:
                candidate = res.x.reshape(n, 2)
                try_update(candidate)
        except Exception:
            pass

    # Phase 2: Global search with Differential Evolution
    # Use the smooth surrogate for speed, but DE is robust to the approximation
    if time.time() < deadline - 40:
        bounds = [(-2.0, 2.0)] * (2 * n)
        try:
            # DE is slow, so limit maxiter and popsize
            de_res = differential_evolution(
                lambda x: soft_max_log_score(x, 10.0),
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=True,
                updateinterval=5
            )
            if de_res.fun < np.inf:
                candidate = de_res.x.reshape(n, 2)
                try_update(candidate)
        except Exception:
            pass

    # Phase 3: High-precision polish on the best candidate
    # Use increasingly sharp betas to approximate the exact max more closely
    # Then a final pass with the exact objective (if gradients are not needed, or use finite diff)
    # L-BFGS-B uses numerical gradients by default, which is fine for n=16 variables.
    
    current_best = best_points.copy()
    
    for beta_val in [1.0, 0.1, 0.01, 0.001]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                lambda x, b=beta_val: soft_max_log_score(x.reshape(n, 2), b),
                current_best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < np.inf:
                candidate = res.x.reshape(n, 2)
                s = exact_log_score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
                    current_best = candidate
        except Exception:
            pass

    # Final polish with exact objective (numerical gradient)
    if time.time() < deadline - 2:
        try:
            def exact_obj(x):
                return -exact_log_score(x.reshape(n, 2))
            
            res = minimize(
                exact_obj,
                current_best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.fun < np.inf:
                candidate = res.x.reshape(n, 2)
                s = exact_log_score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate
        except Exception:
            pass

    # Final validation
    # Check distinctness and bounds
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_d = np.sqrt(np.min(dists_sq))
    
    if min_d < 1e-8:
        # Fallback to regular octagon if points collapsed
        best_points = reg_oct

    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        # Scale down if needed, though bounds should prevent this
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]

if __name__ == "__main__":
    pts = construct()
    # Quick sanity check
    if pts:
        print("Constructed 8 points.")
        print(pts[0])
