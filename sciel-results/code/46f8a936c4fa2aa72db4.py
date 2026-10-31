"""Combined strategy: Geometric initialization with normalized local refinement."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

DEADLINE = 110.0  # safety margin before 120s limit (or 150s if that's the real limit, but 110 is safe)

def _pairwise_dists_sq(P):
    """Return matrix of squared pairwise distances, shape (n, n)."""
    diff = P[:, None, :] - P[None, :, :]
    return np.sum(diff * diff, axis=-1)

def _score(P):
    """(max pairwise dist / min pairwise dist)^2 over distinct pairs."""
    D2 = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, k=1)
    vals = D2[iu]
    vmin = vals.min()
    vmax = vals.max()
    if vmin <= 1e-15:
        return 1e12
    return (vmax / vmin)

def _generate_hexagonal_candidate(seed=0):
    """
    Generate a candidate configuration from a hexagonal lattice.
    We pick 16 points from a hexagonal grid.
    """
    # Hexagonal lattice vectors
    # a1 = (1, 0), a2 = (0.5, sqrt(3)/2)
    # Points are i*a1 + j*a2
    
    # We want to search for a subset of 16 points in a hexagonal lattice
    # that minimizes the diameter/side ratio.
    
    # A known good configuration for small N is a subset of the triangular lattice.
    # Let's try to find a "compact" cluster of 16 points in the triangular lattice.
    
    # Generate a larger grid of triangular lattice points
    # Center at (0,0)
    # Radius R in terms of lattice steps
    R = 3 
    pts = []
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            x = i + 0.5 * j
            y = np.sqrt(3)/2 * j
            pts.append([x, y])
    
    pts = np.array(pts)
    
    # We need to select 16 points.
    # Strategy: Start with the 16 closest to the origin.
    # Then we can try to "swap" points to improve the metric.
    # Or just use the 16 closest as a strong initial guess.
    
    dists = np.sum(pts**2, axis=1)
    idx = np.argsort(dists)
    
    # Take the first 16
    selected_idx = idx[:N]
    P = pts[selected_idx].copy()
    
    # Add some random rotation to break symmetry and potentially escape local minima
    # This is crucial because the hexagonal lattice has high symmetry.
    angle = np.random.uniform(0, 2*np.pi, seed=seed)
    c, s = np.cos(angle), np.sin(angle)
    R_mat = np.array([[c, -s], [s, c]])
    P = P @ R_mat.T
    
    # Add small noise
    P += np.random.normal(0, 0.01, P.shape, seed=seed+1)
    
    return P

def _normalize_and_fix(P):
    """
    Normalize P such that:
    1. The pair with the minimum distance is fixed at (0,0) and (1,0).
    2. This removes translation, rotation, and scale degrees of freedom.
    3. Returns the parameters for the remaining N-2 points.
    """
    D2 = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, k=1)
    d2_flat = D2[iu]
    
    # Find min distance pair
    min_idx = np.argmin(d2_flat)
    i0, i1 = iu[0][min_idx], iu[1][min_idx]
    
    # Ensure i0 < i1 for consistency? Not strictly necessary if we permute, 
    # but let's be consistent.
    if i0 > i1:
        i0, i1 = i1, i0
        
    p0 = P[i0]
    p1 = P[i1]
    
    v = p1 - p0
    norm_v = np.linalg.norm(v)
    
    if norm_v < 1e-8:
        # Degenerate, return zeros
        return np.zeros((N-2)*2)
        
    # Rotation matrix to align v with x-axis
    # v = [vx, vy]
    # We want v @ R_mat.T = [1, 0] (after scaling)
    # Actually, we transform points: P' = (P - p0) @ R_mat.T / norm_v
    # Let's check:
    # p0' = 0
    # p1' = (p1 - p0) @ R_mat.T / norm_v = v @ R_mat.T / norm_v
    # We want v @ R_mat.T = [norm_v, 0]
    # Let R_mat = [[c, s], [-s, c]]
    # v @ R_mat.T = [vx*c + vy*s, -vx*s + vy*c]
    # We want -vx*s + vy*c = 0 => tan(theta) = vy/vx ? No.
    # Let's use standard rotation.
    # If we rotate by -angle, where angle = atan2(vy, vx)
    # c = cos(angle), s = sin(angle)
    # R_mat = [[c, s], [-s, c]] is rotation by -angle?
    # [1, 0] @ R_mat.T = [c, -s]. If angle=0, c=1, s=0 -> [1,0].
    # v = [r, 0] -> [r*c, -r*s]. If angle=0, [r, 0].
    # v = [0, r] (angle=pi/2). c=0, s=1. R_mat = [[0, 1], [-1, 0]].
    # v @ R_mat.T = [0, r] @ [[0, -1], [1, 0]] = [r, 0]. Correct.
    
    angle = np.arctan2(v[1], v[0])
    c = np.cos(angle)
    s = np.sin(angle)
    R_mat = np.array([[c, s], [-s, c]])
    
    transformed = (P - p0) @ R_mat.T / norm_v
    
    # Reorder points so that i0 is at index 0 and i1 is at index 1
    # Create permutation
    perm = list(range(N))
    perm[0] = i0
    perm[1] = i1
    # Fill the rest
    remaining = [i for i in range(N) if i != i0 and i != i1]
    perm[2:] = remaining
    
    transformed = transformed[perm]
    
    # Now transformed[0] is (0,0) and transformed[1] is (1,0)
    # Parameters are coordinates of points 2..N-1
    params = transformed[2:].flatten()
    
    return params

def _reconstruct(params):
    """Reconstruct P from normalized parameters."""
    P = np.zeros((N, 2))
    P[0] = [0.0, 0.0]
    P[1] = [1.0, 0.0]
    P[2:] = params.reshape(-1, 2)
    return P

def _smooth_objective(params, beta=50.0):
    """
    Smooth surrogate for local refinement.
    Minimizes the soft range of squared distances.
    """
    P = _reconstruct(params)
    D2 = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, k=1)
    vals = D2[iu]
    
    # Soft max and soft min of squared distances
    # Using log-sum-exp trick for stability
    m = vals.max()
    logsumexp_max = m + np.log(np.sum(np.exp(vals - m)))
    
    mn = vals.min()
    logsumexp_min_neg = -mn + np.log(np.sum(np.exp(-vals + mn)))
    soft_min = -logsumexp_min_neg
    
    if soft_min <= 1e-8:
        return 1e6
        
    # Objective: (soft_max - soft_min) / soft_min
    # This approximates the ratio of max to min
    return (logsumexp_max - soft_min) / soft_min

def construct():
    t_start = time.time()
    
    best_score = 1e12
    best_params = None
    
    # Generate multiple diverse initial candidates from hexagonal lattice
    # We use different rotations and potentially different subsets (though we stick to closest 16 for simplicity and robustness)
    # Actually, let's try a few different "seeds" for the random rotation and noise.
    
    candidates = []
    for i in range(5):
        P_init = _generate_hexagonal_candidate(seed=i)
        params_init = _normalize_and_fix(P_init)
        candidates.append(params_init)
        
    # Also add a random initialization just in case
    np.random.seed(123)
    P_rand = np.random.uniform(-5, 5, (N, 2))
    params_rand = _normalize_and_fix(P_rand)
    candidates.append(params_rand)
    
    bnds = [(-5, 5)] * (2 * (N - 2))
    
    for idx, init_params in enumerate(candidates):
        if time.time() > DEADLINE - 10:
            break
            
        # Initial score check
        P_check = _reconstruct(init_params)
        current_score = _score(P_check)
        
        if current_score < best_score:
            best_score = current_score
            best_params = init_params.copy()
            
        # Optimization phases
        # Phase 1: Coarse, low beta
        if time.time() < DEADLINE - 30:
            try:
                res1 = minimize(
                    _smooth_objective,
                    init_params,
                    method='L-BFGS-B',
                    bounds=bnds,
                    args=(10.0,),
                    options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4}
                )
                if res1.fun < 1e9:
                    P_cand = _reconstruct(res1.x)
                    cand_score = _score(P_cand)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = res1.x
                        if cand_score < best_score:
                            best_score = cand_score
                            best_params = res1.x.copy()
            except Exception:
                pass

        # Phase 2: Medium, higher beta
        if time.time() < DEADLINE - 15:
            try:
                res2 = minimize(
                    _smooth_objective,
                    init_params,
                    method='L-BFGS-B',
                    bounds=bnds,
                    args=(50.0,),
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res2.fun < 1e9:
                    P_cand = _reconstruct(res2.x)
                    cand_score = _score(P_cand)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = res2.x
                        if cand_score < best_score:
                            best_score = cand_score
                            best_params = res2.x.copy()
            except Exception:
                pass

        # Phase 3: Fine, very high beta (almost exact)
        if time.time() < DEADLINE - 5:
            try:
                res3 = minimize(
                    _smooth_objective,
                    init_params,
                    method='L-BFGS-B',
                    bounds=bnds,
                    args=(100.0,),
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res3.fun < 1e9:
                    P_cand = _reconstruct(res3.x)
                    cand_score = _score(P_cand)
                    if cand_score < current_score:
                        current_score = cand_score
                        init_params = res3.x
                        if cand_score < best_score:
                            best_score = cand_score
                            best_params = res3.x.copy()
            except Exception:
                pass

    # Final sanity check and distinctness
    if best_params is None:
        # Fallback to a simple grid
        P_best = np.array([[i % 4 - 1.5, i // 4 - 1.5] for i in range(N)], dtype=float)
    else:
        P_best = _reconstruct(best_params)
        
    # Ensure distinctness
    D2 = _pairwise_dists_sq(P_best)
    iu = np.triu_indices(N, k=1)
    if D2[iu].min() < 1e-10:
        # Perturb
        np.random.seed(42)
        P_best += np.random.randn(N, D) * 1e-5

    return P_best
# EVOLVE-BLOCK-END
