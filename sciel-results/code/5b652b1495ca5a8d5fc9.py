import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 16
D = 2

def _pairwise_distances(pts):
    """Compute all pairwise Euclidean distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def _score(pts):
    """Compute (max/min)^2 for pairwise distances."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    dmin = d.min()
    dmax = d.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2

def _smooth_min_dist(pts, tau):
    """
    Smooth approximation of the minimum pairwise distance.
    Uses log-sum-exp trick for numerical stability.
    Returns a value close to min(d_ij) for large tau.
    """
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    d_valid = d[~np.eye(N, dtype=bool)]
    
    if np.any(d_valid < 1e-12):
        return 0.0
        
    # Smooth min: -1/tau * log(sum(exp(-tau * d)))
    # To avoid overflow, subtract min:
    d_min = np.min(d_valid)
    exp_terms = np.exp(-tau * (d_valid - d_min))
    smooth_min = d_min - (1.0 / tau) * np.log(np.sum(exp_terms))
    return smooth_min

def _diameter(pts):
    """Compute the maximum pairwise distance (diameter)."""
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, 0)
    return np.max(d)

def _penalized_objective(pts_flat, tau, lambda_penalty):
    """
    Objective: Minimize -smooth_min_dist + lambda * max(0, diameter - 1)^2
    We want to maximize min_dist subject to diameter <= 1.
    So we minimize -min_dist + penalty.
    """
    pts = pts_flat.reshape(N, 2)
    
    # Smooth min distance
    smin = _smooth_min_dist(pts, tau)
    
    # Diameter constraint
    diam = _diameter(pts)
    if diam > 1.0:
        penalty = (diam - 1.0) ** 2
    else:
        penalty = 0.0
        
    return -smin + lambda_penalty * penalty

def _make_hexagonal_cluster(n, spread=6):
    """Generate a hexagonal lattice cluster of n points centered at origin."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])

    pts = []
    for i in range(-spread, spread + 1):
        for j in range(-spread, spread + 1):
            p = i * v1 + j * v2
            pts.append(p)
    pts = np.array(pts)

    # Center the cluster
    pts -= pts.mean(axis=0)

    # Sort by distance to center and pick the closest n
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    return pts[order[:n]]

def _make_circular_config(n):
    """1 on center + 7 on inner ring + 8 on outer ring, scaled to fit."""
    pts = [np.array([0.0, 0.0])]
    r1 = 0.5
    r2 = 1.0
    for k in range(7):
        theta = 2 * np.pi * k / 7
        pts.append(r1 * np.array([np.cos(theta), np.sin(theta)]))
    for k in range(8):
        theta = 2 * np.pi * k / 8 + np.pi / 8
        pts.append(r2 * np.array([np.cos(theta), np.sin(theta)]))
    return np.array(pts)

def _make_grid_config(n):
    """4x4 grid, centered and scaled to unit diameter."""
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    pts = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    pts -= pts.mean(axis=0)
    # Scale to diameter 1
    diam = _diameter(pts)
    if diam > 0:
        pts /= diam
    return pts

def _make_random_config(n, seed=0):
    """Random points in unit square, centered and scaled."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    pts -= pts.mean(axis=0)
    diam = _diameter(pts)
    if diam > 0:
        pts /= diam
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_pts = None
    best_score = np.inf

    def update_best(pts, score):
        nonlocal best_pts, best_score
        if score < best_score:
            best_score = score
            best_pts = pts.copy()

    # 1. Generate diverse initial candidates
    candidates = [
        _make_hexagonal_cluster(N),
        _make_circular_config(N),
        _make_grid_config(N),
        _make_random_config(N, seed=1),
        _make_random_config(N, seed=2),
        _make_random_config(N, seed=3),
        _make_random_config(N, seed=4),
    ]

    # Evaluate all candidates
    for c in candidates:
        s = _score(c)
        update_best(c, s)

    # 2. Global Search using Differential Evolution
    # We use the penalized smooth objective for DE.
    # The parameters tau and lambda are fixed for DE to keep it simple, 
    # but we use a moderate tau to ensure smoothness.
    
    if time.time() < deadline - 50:
        try:
            tau_de = 20.0
            lambda_de = 100.0
            
            def de_objective(pts_flat):
                return _penalized_objective(pts_flat, tau_de, lambda_de)

            bounds = [(-1, 1)] * (N * 2)
            result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                polish=False,
                init='sobol',
            )
            if result.fun < np.inf:
                new_pts = result.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    # 3. Local Refinement using L-BFGS-B with annealing
    # We refine the best candidates found so far.
    
    def local_refine(pts_init, time_limit):
        """Refine a configuration using L-BFGS-B with annealing tau and lambda."""
        current_pts = pts_init.copy()
        current_score = _score(current_pts)
        
        # Annealing schedule: 
        # Start with low tau (smooth) and low lambda (loose constraint)
        # Increase tau to sharpen the min approximation
        # Increase lambda to enforce the diameter constraint strictly
        # The objective is -smooth_min + lambda * penalty
        # As lambda -> inf, we enforce diam <= 1.
        # As tau -> inf, smooth_min -> min.
        
        # We want to maximize min_dist. 
        # The penalty drives diam down to 1.
        
        stages = [
            (10.0, 10.0),   # Smooth, loose
            (20.0, 50.0),   # Sharper, tighter
            (50.0, 200.0),  # Sharper, tighter
            (100.0, 1000.0),# Very sharp, very tight
            (200.0, 5000.0),# Very sharp, very tight
        ]
        
        for tau, lam in stages:
            if time.time() > time_limit:
                break
                
            def obj_flat(x):
                return _penalized_objective(x, tau, lam)
                
            try:
                res = minimize(
                    obj_flat,
                    current_pts.ravel(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                
                if new_score < current_score:
                    current_pts = new_pts
                    current_score = new_score
            except Exception:
                pass
        
        return current_pts, current_score

    # Refine top candidates
    candidates_to_refine = list(candidates)
    if best_pts is not None:
        candidates_to_refine.append(best_pts)
    
    # Sort by raw score to prioritize best ones
    candidates_to_refine.sort(key=lambda x: _score(x))
    
    # Refine the top 5 distinct candidates
    refined_count = 0
    for c in candidates_to_refine:
        if time.time() > deadline - 20:
            break
        if refined_count >= 5:
            break
            
        # Avoid refining exact duplicates
        is_dup = False
        for r in candidates_to_refine[:refined_count]:
            if np.allclose(c, r, atol=1e-5):
                is_dup = True
                break
        if is_dup:
            continue
            
        try:
            refined_pts, refined_score = local_refine(c, deadline - 15)
            update_best(refined_pts, refined_score)
            refined_count += 1
        except Exception:
            pass

    # 4. Final Polish with Nelder-Mead on the raw objective
    # This helps with the final non-smooth steps that gradient methods might miss.
    if best_pts is not None and time.time() < deadline - 5:
        try:
            def raw_obj(x):
                pts = x.reshape(N, 2)
                return _score(pts)
            
            res = minimize(
                raw_obj,
                best_pts.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_score:
                new_pts = res.x.reshape(N, 2)
                new_score = _score(new_pts)
                update_best(new_pts, new_score)
        except Exception:
            pass

    if best_pts is None:
        return _make_hexagonal_cluster(N)

    return best_pts
