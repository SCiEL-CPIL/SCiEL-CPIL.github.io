import numpy as np
import time
from scipy.optimize import minimize, basinhopping, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s for safety

    n = 8
    # Precompute pair indices for 28 unordered pairs
    pair_indices = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    idx_i = pair_indices[:, 0]
    idx_j = pair_indices[:, 1]

    # Constants
    LOG_2 = np.log(2.0)
    LOG_8 = np.log(8.0)
    # log(Score) = 2*sum(log(d_ij)) + 56*log(2) - 56*log(D) - 8*log(8)
    CONST = 56.0 * LOG_2 - 8.0 * LOG_8

    def objective(x):
        """Minimize -log(Score) with soft diameter penalty."""
        if time.time() > deadline:
            return 1e10
        
        points = x.reshape(n, 2)
        
        # Pairwise distances
        diff = points[idx_i, :] - points[idx_j, :]
        dists_sq = np.sum(diff**2, axis=1)
        dists = np.sqrt(dists_sq)
        
        # Check for distinct points
        if np.any(dists < 1e-10):
            return 1e10
        
        D = np.max(dists)
        if D < 1e-10:
            return 1e10
        
        # Soft penalty for diameter: penalize if any distance > 2.0 (after scaling)
        # But since we normalize by D, the diameter is always 2.0 after scaling.
        # Instead, we work in the original space and the score is scale-invariant.
        # The key is to maximize the product of scaled distances.
        
        sum_log_d = np.sum(np.log(dists))
        log_score = CONST + 2.0 * sum_log_d - 56.0 * np.log(D)
        
        return -log_score

    def objective_with_penalty(x, penalty_weight=100.0):
        """Objective with soft penalty for very small distances (distinctness)."""
        base = objective(x)
        if base > 1e9:
            return base
        
        points = x.reshape(n, 2)
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Penalize if any distance is too small (points too close)
        min_d = np.min(dists)
        if min_d < 0.1:
            penalty = penalty_weight * (0.1 - min_d)**2
            return base + penalty
        
        return base

    # Generate diverse initializations
    rng = np.random.RandomState(42)
    starts = []

    # 1. Regular octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    oct_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    starts.append(oct_pts.flatten())

    # 2. Two squares with various configurations (key insight from inspiration)
    # Square 1: radius r1, angle offset a1
    # Square 2: radius r2, angle offset a2
    for r1 in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]:
        for r2 in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5]:
            for theta in [0.0, np.pi/8, np.pi/6, np.pi/4, np.pi/3, 3*np.pi/8]:
                pts = []
                for k in range(4):
                    a1 = k * np.pi / 2
                    pts.append([r1 * np.cos(a1), r1 * np.sin(a1)])
                for k in range(4):
                    a2 = theta + k * np.pi / 2
                    pts.append([r2 * np.cos(a2), r2 * np.sin(a2)])
                starts.append(np.array(pts).flatten())

    # 3. Random perturbations of octagon
    for _ in range(10):
        noise = rng.normal(0, 0.3, 16)
        starts.append(oct_pts.flatten() + noise)

    # 4. Pure random starts
    for _ in range(5):
        starts.append(rng.uniform(-3, 3, 16))

    # Local refinement with Powell (fast, derivative-free)
    best_x = starts[0].copy()
    best_val = objective(best_x)

    for s in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            res = minimize(
                objective_with_penalty,
                s,
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            val = objective(res.x)
            if val < best_val:
                best_val = val
                best_x = res.x.copy()
        except Exception:
            pass

    # Basinhopping to escape local minima
    if time.time() < deadline - 25.0:
        try:
            def step(x):
                return x + rng.normal(0, 0.1, len(x))
            
            bh = basinhopping(
                objective_with_penalty,
                best_x,
                niter=50,
                stepsize=0.1,
                minimizer_kwargs={'method': 'Powell', 'options': {'maxiter': 100}},
                seed=42,
                callback=lambda x, f, accept, step, new_f: time.time() < deadline - 10.0
            )
            if bh.fun < best_val:
                best_val = bh.fun
                best_x = bh.x.copy()
        except Exception:
            pass

    # Differential evolution for global search if time remains
    if time.time() < deadline - 20.0:
        try:
            bounds = [(-5.0, 5.0)] * 16
            de_res = differential_evolution(
                objective_with_penalty,
                bounds,
                maxiter=30,
                popsize=12,
                tol=1e-8,
                seed=42,
                init='sobol'
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            final_res = minimize(
                objective,
                best_x,
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-15, 'fatol': 1e-15}
            )
            if final_res.fun < best_val:
                best_val = final_res.fun
                best_x = final_res.x.copy()
        except Exception:
            pass

    best_points = best_x.reshape(n, 2)
    return [[float(p[0]), float(p[1])] for p in best_points]
