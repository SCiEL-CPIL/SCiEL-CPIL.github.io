import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # safety margin for 630s kill

    def score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(x, beta=10.0):
        """
        Smooth approximation of - (d_min / d_max)^2 using log-sum-exp.
        d_min ~ -1/beta * log(sum(exp(-beta * d)))
        d_max ~  1/beta * log(sum(exp( beta * d)))
        """
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        dists = pdist(pts)
        if len(dists) == 0 or np.any(~np.isfinite(dists)):
            return 1e6

        # Numerical stability for log-sum-exp
        # d_min approx
        min_term = -beta * dists
        min_max_val = np.max(min_term)
        if min_max_val > 700:
            min_log_sum_exp = min_max_val + np.log(np.sum(np.exp(min_term - min_max_val)))
        else:
            min_log_sum_exp = np.log(np.sum(np.exp(min_term)))
        d_min_approx = -min_log_sum_exp / beta
        
        # d_max approx
        max_term = beta * dists
        max_max_val = np.max(max_term)
        if max_max_val > 700:
            max_log_sum_exp = max_max_val + np.log(np.sum(np.exp(max_term - max_max_val)))
        else:
            max_log_sum_exp = np.log(np.sum(np.exp(max_term)))
        d_max_approx = max_log_sum_exp / beta
        
        if d_max_approx < 1e-12:
            return 1e6
            
        ratio = d_min_approx / d_max_approx
        return -ratio ** 2

    def exact_objective(x):
        """Exact objective for local polishing."""
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        dists = pdist(pts)
        if len(dists) == 0:
            return 1e6
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1e6
        return -(d_min / d_max) ** 2

    best_score = 0.0
    best_points = None

    # 1. Global Search with Differential Evolution
    bounds = [(-10, 10)] * (n * d)
    
    try:
        de_result = differential_evolution(
            smooth_objective,
            bounds,
            args=(10.0,),
            maxiter=500,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=True,
            updating='immediate',
            init='sobol'
        )
        
        if de_result.success or de_result.fun < 0:
            cand = de_result.x.reshape((n, d))
            cand = cand - np.mean(cand, axis=0)
            s = score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
    except Exception:
        pass

    # 2. Local Refinement from multiple random starts + DE result
    candidates_to_polish = []
    if best_points is not None:
        candidates_to_polish.append(best_points.copy())
    
    rng = np.random.RandomState(123)
    for _ in range(10):
        pts = rng.randn(n, d) * 5
        pts = pts - np.mean(pts, axis=0)
        candidates_to_polish.append(pts)
    
    # Add Fibonacci sphere initial points
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    fib_pts = np.column_stack([x, y, z])
    fib_pts = fib_pts - np.mean(fib_pts, axis=0)
    candidates_to_polish.append(fib_pts)

    for init_pts in candidates_to_polish:
        if time.time() > deadline - 60:
            break
            
        x0 = init_pts.flatten()
        
        # Multi-stage beta annealing for smooth objective
        for beta in [10.0, 50.0, 100.0]:
            if time.time() > deadline - 60:
                break
            try:
                res_smooth = minimize(
                    smooth_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                if res_smooth.x is not None:
                    cand = res_smooth.x.reshape((n, d))
                    cand = cand - np.mean(cand, axis=0)
                    s = score(cand)
                    if s > best_score:
                        best_score = s
                        best_points = cand.copy()
                    x0 = res_smooth.x
            except Exception:
                pass

        # Polish with Exact Objective (Powell)
        try:
            res_exact = minimize(
                exact_objective,
                x0,
                method='Powell',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res_exact.x is not None:
                cand = res_exact.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
        except Exception:
            pass

        # Perturbed Powell to escape local minima
        if time.time() > deadline - 30:
            break
        try:
            x0_perturbed = x0 + rng.randn(len(x0)) * 0.1
            res_perturbed = minimize(
                exact_objective,
                x0_perturbed,
                method='Powell',
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            if res_perturbed.x is not None:
                cand = res_perturbed.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
        except Exception:
            pass

    # 3. Final high-precision polish with Nelder-Mead
    if best_points is not None and time.time() < deadline - 20:
        try:
            x0 = best_points.flatten()
            res_nm = minimize(
                exact_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10}
            )
            if res_nm.x is not None:
                cand = res_nm.x.reshape((n, d))
                cand = cand - np.mean(cand, axis=0)
                s = score(cand)
                if s > best_score:
                    best_score = s
                    best_points = cand.copy()
        except Exception:
            pass

    # Final Check and Normalization
    if best_points is not None and np.all(np.isfinite(best_points)):
        pass
    else:
        # Fallback: Fibonacci sphere
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        best_points = np.column_stack([x, y, z])
        best_points = best_points - np.mean(best_points, axis=0)

    return best_points
