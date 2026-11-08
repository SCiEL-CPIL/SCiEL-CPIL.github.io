import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 13
# Precompute all triangle indices
_TRI_IDX = np.array(list(itertools.combinations(range(N), 3)))
_NUM_TRI = len(_TRI_IDX)

def score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        if len(points) < 3:
            return 0.0
        
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        idx = _TRI_IDX
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        return min_area / hull_area
    except Exception:
        return 0.0

def neg_score(points_flat):
    points = points_flat.reshape(N, 2)
    return -score(points)

def neg_score_softmin(points_flat, beta):
    """
    Smoothed objective using log-sum-exp soft-min of triangle areas / hull area.
    Minimizes - (softmin(areas) / hull_area)
    """
    try:
        points = points_flat.reshape(N, 2)
        
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 1e6
        except Exception:
            return 1e6
            
        idx = _TRI_IDX
        p1 = points[idx[:, 0]]
        p2 = points[idx[:, 1]]
        p3 = points[idx[:, 2]]
        
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        # Log-sum-exp soft-min: -log(sum(exp(-beta * areas))) / beta
        # Shift for numerical stability
        min_area = np.min(areas)
        shifted = areas - min_area
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min_val = min_area - (1.0 / beta) * log_sum
        
        return -soft_min_val / hull_area
    except Exception:
        return 1e6

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    best_score = 0.0
    best_points = None

    def update_best(points):
        nonlocal best_score, best_points
        s = score(points)
        if s > best_score:
            best_score = s
            best_points = points

    rng = np.random.default_rng(42)

    # Generate diverse initial configurations
    starting_points = []

    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    starting_points.append(init_reg)

    # 2. Noisy regular 13-gon with varying noise levels
    for sigma in [0.05, 0.1, 0.15, 0.2]:
        noise = rng.normal(0, sigma, (N, 2))
        starting_points.append(init_reg + noise)

    # 3. Two rings: 7 outer, 6 inner
    for r_in in [0.3, 0.4, 0.5, 0.6, 0.7]:
        angles_7 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles_6 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        p7 = np.column_stack((np.cos(angles_7), np.sin(angles_7)))
        p6 = np.column_stack((r_in * np.cos(angles_6), r_in * np.sin(angles_6)))
        starting_points.append(np.vstack([p7, p6]))

    # 4. Three rings: 5 outer, 4 mid, 4 inner
    for r_mid, r_in in [(0.5, 0.25), (0.6, 0.3), (0.7, 0.4), (0.75, 0.5)]:
        angles_5 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        angles_4 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        p5 = np.column_stack((np.cos(angles_5), np.sin(angles_5)))
        p4m = np.column_stack((r_mid * np.cos(angles_4), r_mid * np.sin(angles_4)))
        p4i = np.column_stack((r_in * np.cos(angles_4 + np.pi/4), r_in * np.sin(angles_4 + np.pi/4)))
        starting_points.append(np.vstack([p5, p4m, p4i]))

    # 5. Random points in unit disk
    for _ in range(5):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        starting_points.append(pts)

    # Multi-stage annealing: coarse to fine
    betas = [5.0, 20.0, 100.0, 500.0, 2000.0]

    # Phase 1: Quick L-BFGS-B from each starting point with coarse beta
    for start in starting_points:
        if time.time() > deadline - 15:
            break
        current = start.flatten()
        for beta in betas[:2]:  # Only coarse stages for all starts
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(
                    neg_score_softmin,
                    current,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 30}
                )
                if res.fun < 1e5:
                    update_best(res.x.reshape(N, 2))
                    current = res.x
            except Exception:
                pass

    # Phase 2: Fine-tune the best candidates
    if best_points is not None:
        # Generate perturbations around the best point
        fine_candidates = [best_points]
        for sigma in [0.02, 0.05, 0.1]:
            for _ in range(3):
                noise = rng.normal(0, sigma, (N, 2))
                fine_candidates.append(best_points + noise)

        for cand in fine_candidates:
            if time.time() > deadline - 10:
                break
            current = cand.flatten()
            for beta in betas[2:]:  # Fine stages
                if time.time() > deadline - 5:
                    break
                try:
                    res = minimize(
                        neg_score_softmin,
                        current,
                        args=(beta,),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                    )
                    if res.fun < 1e5:
                        update_best(res.x.reshape(N, 2))
                        current = res.x
                except Exception:
                    pass

    # Phase 3: Final polish with true objective
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_true = minimize(
                neg_score,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            if res_true.fun < 0:
                update_best(res_true.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 4: If time remains, run a short DE with tight bounds around best
    if best_points is not None and time.time() < deadline - 25:
        try:
            center = best_points.mean(axis=0)
            spread = np.abs(best_points - center).max()
            b = max(spread * 1.2, 0.3)
            bounds_tight_flat = []
            for i in range(N):
                bounds_tight_flat.append((best_points[i, 0] - b, best_points[i, 0] + b))
                bounds_tight_flat.append((best_points[i, 1] - b, best_points[i, 1] + b))

            de2 = differential_evolution(
                neg_score_softmin,
                bounds=bounds_tight_flat,
                args=(500.0,),
                seed=99,
                maxiter=30,
                popsize=8,
                tol=1e-6,
                mutation=(0.2, 0.6),
                recombination=0.7,
                polish=False,
                init='latinhypercube'
            )

            if de2.fun < 1e5:
                de2_points = de2.x.reshape(N, 2)
                res_p = minimize(
                    neg_score_softmin,
                    de2_points.flatten(),
                    args=(2000.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                if res_p.fun < 1e5:
                    update_best(res_p.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = init_reg

    return best_points
