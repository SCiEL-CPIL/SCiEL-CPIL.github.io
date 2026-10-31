import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping, SLSQP
import time
from scipy.special import logsumexp

N = 16
D = 2

def _pairwise_dists_sq(pts):
    """Compute all pairwise squared distances."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(n, k=1)
    return dists_sq[iu]

def _score(pts):
    """Compute (max/min)^2."""
    d2 = _pairwise_dists_sq(pts)
    dmin2 = d2.min()
    dmax2 = d2.max()
    if dmin2 < 1e-12:
        return 1e10
    return dmax2 / dmin2

def _soft_score(pts, beta=20.0):
    """Smooth surrogate using log-sum-exp for min and max squared distances."""
    d2 = _pairwise_dists_sq(pts)
    # Soft max of d2
    d_max_soft = logsumexp(beta * d2) / beta
    # Soft min of d2
    d_min_soft = -logsumexp(-beta * d2) / beta
    if d_min_soft < 1e-12:
        return 1e10
    return d_max_soft / d_min_soft

def _hex_init_19():
    """Generate 19 points in a hexagonal lattice."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([np.cos(angle), np.sin(angle)])
    # Ring 2: 12 points
    for i in range(12):
        angle = i * np.pi / 6
        # In hex lattice, ring 2 points are at distance 2
        # But the angles are not simply i*pi/6 for a perfect hexagon of radius 2?
        # Yes, for a regular hexagon, the vertices are at angles 0, 60, 120, 180, 240, 300.
        # The midpoints of the sides are at 30, 90, 150, 210, 270, 330.
        # Distance is 2 for all.
        pts.append([2.0*np.cos(angle), 2.0*np.sin(angle)])
    return np.array(pts, dtype=float)

def _get_initial_candidates():
    """Generate diverse initial candidates."""
    candidates = []
    
    # 1. 4x4 Grid
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    grid -= 1.5
    candidates.append(grid)
    
    # 2. Hex 19 minus 3 variations
    hex19 = _hex_init_19()
    # Indices of Ring 2 points are 7 to 18 (12 points)
    # Remove 3 points.
    # Variation A: Remove 3 adjacent
    idx_remove_A = [7, 8, 9]
    candidates.append(np.delete(hex19, idx_remove_A, axis=0))
    
    # Variation B: Remove 3 spread out (every 4th)
    idx_remove_B = [7, 11, 15]
    candidates.append(np.delete(hex19, idx_remove_B, axis=0))
    
    # Variation C: Remove 3 to make it more square-like?
    # Just random removal
    rng = np.random.RandomState(42)
    for _ in range(5):
        idx_remove = rng.choice(np.arange(7, 19), 3, replace=False)
        candidates.append(np.delete(hex19, idx_remove, axis=0))
        
    # 3. Random
    for seed in [1, 2, 3]:
        rng = np.random.RandomState(seed)
        pts = rng.rand(N, D) * 4 - 2
        candidates.append(pts)
        
    return candidates

def _constrained_objective(x, penalty_weight=100.0):
    """
    Objective: Minimize d_max^2 + penalty * (violation of d_min >= 1)^2
    We assume the points are scaled such that d_min should be ~1.
    """
    pts = x.reshape(N, D)
    d2 = _pairwise_dists_sq(pts)
    dmax2 = d2.max()
    dmin2 = d2.min()
    
    # Penalty if min dist is less than 1
    # We want d_min >= 1.
    # violation = max(0, 1 - dmin2)
    violation = max(0.0, 1.0 - dmin2)
    penalty = penalty_weight * violation**2
    
    return dmax2 + penalty

def _local_optimize(pts, method='SLSQP'):
    """Refine a candidate using local optimization."""
    x0 = pts.flatten()
    
    if method == 'SLSQP':
        # SLSQP can handle constraints, but we use penalty for robustness
        # Let's try SLSQP with bounds
        bounds = [(-5, 5)] * (N * D)
        res = minimize(
            _constrained_objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if res.success:
            return res.x.reshape(N, D)
        else:
            return res.x.reshape(N, D) # Return best attempt
            
    elif method == 'L-BFGS-B':
        res = minimize(
            _soft_score, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        return res.x.reshape(N, D)
        
    return pts

def construct():
    """Main entry point."""
    deadline = time.time() + 145  # Safety margin
    
    best_score = float('inf')
    best_pts = None
    
    candidates = _get_initial_candidates()
    
    # 1. Local optimization from each candidate
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
        
        # Try SLSQP (constrained/penalty)
        try:
            opt_pts = _local_optimize(cand, method='SLSQP')
            score = _score(opt_pts)
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
        except Exception:
            pass
            
        # Try L-BFGS-B on soft score
        try:
            opt_pts_soft = _local_optimize(cand, method='L-BFGS-B')
            score_soft = _score(opt_pts_soft)
            if score_soft < best_score:
                best_score = score_soft
                best_pts = opt_pts_soft.copy()
        except Exception:
            pass

    # 2. Basin Hopping on the best candidate found so far
    if best_pts is None:
        best_pts = candidates[0]
        
    if time.time() < deadline - 10:
        def bh_objective(x):
            pts = x.reshape(N, D)
            return _soft_score(pts, beta=20.0)
        
        def bh_minimizer(x):
            # Use SLSQP for local minimization within basin hopping
            # SLSQP is slower but more robust for this non-smooth objective
            try:
                res = minimize(
                    _constrained_objective, 
                    x, 
                    method='SLSQP', 
                    bounds=[(-5, 5)] * (N * D),
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                return _soft_score(res.x.reshape(N, D), beta=20.0), res.x, {}
            except Exception:
                return bh_objective(x), x, {}
        
        try:
            result_bh = basinhopping(
                bh_objective, best_pts.flatten(), 
                minimizer=bh_minimizer,
                niter=10, # Reduce iterations to fit time
                T=0.5,
                seed=42
            )
            if time.time() < deadline - 5:
                pts_bh = result_bh.x.reshape(N, D)
                score_bh = _score(pts_bh)
                if score_bh < best_score:
                    best_score = score_bh
                    best_pts = pts_bh.copy()
        except Exception:
            pass

    # 3. Final Polish with Exact Objective (Powell is good for non-differentiable)
    if time.time() < deadline - 2:
        def exact_obj(x):
            pts = x.reshape(N, D)
            return _score(pts)
        
        try:
            res = minimize(exact_obj, best_pts.flatten(), method='Powell',
                           options={'maxiter': 100, 'ftol': 1e-12})
            pts_pol = res.x.reshape(N, D)
            score_pol = _score(pts_pol)
            if score_pol < best_score:
                best_score = score_pol
                best_pts = pts_pol.copy()
        except Exception:
            pass

    return best_pts
