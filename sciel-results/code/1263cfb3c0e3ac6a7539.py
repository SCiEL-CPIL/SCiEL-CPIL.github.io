import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # Precompute triple indices for vectorized area calculation
    # Number of triples is C(13, 3) = 286
    i, j, k = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
    mask = (i < j) & (j < k)
    ti, tj, tk = i[mask], j[mask], k[mask]

    def compute_score(points):
        """
        Compute min triangle area / convex hull area.
        Returns 0.0 if invalid.
        """
        try:
            # Check for degenerate hull
            hull = ConvexHull(points)
            area_hull = hull.volume
            if area_hull < 1e-12:
                return 0.0
            
            x = points[:, 0]
            y = points[:, 1]
            
            xi = x[ti]
            xj = x[tj]
            xk = x[tk]
            yi = y[ti]
            yj = y[tj]
            yk = y[tk]
            
            # Vectorized cross product magnitude
            cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
            areas = 0.5 * cross
            min_area = np.min(areas)
            
            return min_area / area_hull
        except Exception:
            return 0.0

    def objective(x_flat):
        """
        Objective function for minimization (we minimize negative score).
        """
        pts = x_flat.reshape(-1, 2)
        score = compute_score(pts)
        return -score

    # Initial bounds for coordinates.
    # Since the metric is scale invariant, the optimizer can use the full range.
    # We use [-1, 1] to keep numbers manageable.
    bounds = [(-1.0, 1.0)] * (N * 2)
    
    best_score = -np.inf
    best_points = None
    
    # Helper to update best
    def update_best(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Phase 1: Differential Evolution (Global Search)
    # DE is robust to non-smooth objectives like min().
    # We allocate most of the time here.
    # maxiter is set high, but we rely on the timeout check inside the worker if possible,
    # or just set a reasonable maxiter that fits in ~100s.
    # With 26 dimensions, DE can be slow. 
    # popsize=10 means 260 individuals. 
    # Let's try a moderate popsize and maxiter.
    
    try:
        # Use a seed for reproducibility if needed, but random is fine for exploration
        # DE is stochastic.
        
        # We cannot easily interrupt DE mid-run in scipy without a callback that raises an exception.
        # So we estimate the time.
        # One function eval takes ~1-2ms (ConvexHull is the bottleneck).
        # 286 triples calculation is fast.
        # ConvexHull on 13 points is very fast.
        
        # Let's run DE with a budget.
        # popsize=8, maxiter=200.
        # Total evals ~ 8 * 26 * 200 = 41600 evals.
        # At 2ms/eval, that's ~80 seconds.
        
        res = opt.differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=8,
            tol=1e-6,
            seed=42,
            init='sobol'  # Sobol is often better for high dim than latin hypercube
        )
        
        de_pts = res.x.reshape(-1, 2)
        de_score = -res.fun
        update_best(de_pts, de_score)
        
        print(f"DE finished. Score: {de_score:.6f}", flush=True)
        
    except Exception as e:
        print(f"DE failed: {e}", flush=True)
        # Fallback to a simple random search if DE fails completely
        if best_points is None:
            rng = np.random.default_rng(0)
            for _ in range(100):
                pts = rng.uniform(-1, 1, (N, 2))
                s = compute_score(pts)
                update_best(pts, s)
            print(f"Random search fallback. Score: {best_score:.6f}", flush=True)

    # Phase 2: Local Refinement with Nelder-Mead
    # NM is excellent for polishing the result of a global search.
    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            # NM options: maxiter, adaptive, xatol, fatol
            # We want high precision.
            res_nm = opt.minimize(
                objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'adaptive': True,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            
            nm_pts = res_nm.x.reshape(-1, 2)
            nm_score = -res_nm.fun
            update_best(nm_pts, nm_score)
            
            print(f"NM finished. Score: {nm_score:.6f}", flush=True)
            
        except Exception as e:
            print(f"NM failed: {e}", flush=True)

    # Phase 3: Second pass of NM or a different local method if time remains
    if best_points is not None and time.time() - start_time < time_limit - 5:
        try:
            # Try a slightly perturbed start to see if we can jump out of a local min
            # Or just run NM again with tighter tolerances
            res_nm2 = opt.minimize(
                objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'adaptive': True,
                    'xatol': 1e-12,
                    'fatol': 1e-14
                }
            )
            nm2_pts = res_nm2.x.reshape(-1, 2)
            nm2_score = -res_nm2.fun
            update_best(nm2_pts, nm2_score)
            print(f"NM2 finished. Score: {nm2_score:.6f}", flush=True)
        except Exception as e:
            print(f"NM2 failed: {e}", flush=True)

    # Fallback if everything failed
    if best_points is None:
        # Regular polygon as last resort
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
