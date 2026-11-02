"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

def _all_triangle_areas(pts):
    """Compute all triangle areas among triples of points, vectorized."""
    n = len(pts)
    if n < 3:
        return np.array([])
    
    # Vectorized calculation of cross products for all pairs (i, j) and k
    # To save memory and time, we compute areas for fixed i, j and all k > j
    # But for N=13, O(N^3) is 2197, which is tiny.
    # Let's just do it simply but efficiently.
    
    # P_i, P_j, P_k
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i) |
    
    # We can use broadcasting
    # pts shape (N, 2)
    # i, j, k indices
    i_idx = np.arange(N)
    i_idx = i_idx[:, None, None] # (N, 1, 1)
    j_idx = np.arange(N)
    j_idx = j_idx[None, :, None] # (1, N, 1)
    k_idx = np.arange(N)
    k_idx = k_idx[None, None, :] # (1, 1, N)
    
    # We only care about i < j < k
    # Mask
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    # Cross product (2D)
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    areas = 0.5 * np.abs(cross)
    
    return areas[mask]

def _min_triangle_area(pts):
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_points(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _neg_score(pts_flat):
    pts = pts_flat.reshape(-1, 2)
    return -_score_points(pts)

def _generate_symmetric_points(params, K):
    """
    Generate points from symmetric parameters.
    params: array of length 2*K.
    For each ring k: radius r_k, angle offset theta_k.
    Points in ring k are equally spaced.
    """
    pts = []
    for k in range(K):
        r = params[2*k]
        theta0 = params[2*k+1]
        # Number of points in this ring
        # We need to distribute N points among K rings.
        # This function assumes a fixed distribution passed in or determined by K.
        # To make it flexible, we'll define the distribution outside or use a fixed one.
        # Let's use a helper to get the counts.
        pass
    return np.array([]) # Placeholder

def _build_symmetric_points(params, counts):
    """
    Build points from radii and angles.
    params: [r0, a0, r1, a1, ...]
    counts: [n0, n1, ...]
    """
    pts = []
    idx = 0
    for k, count in enumerate(counts):
        if count == 0:
            continue
        r = params[idx]
        theta0 = params[idx+1]
        idx += 2
        angles = theta0 + 2 * np.pi * np.arange(count) / count
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        ring_pts = np.column_stack([x, y])
        pts.append(ring_pts)
    
    if not pts:
        return np.zeros((N, 2))
    return np.vstack(pts)

def _neg_symmetric_score(params, counts):
    """
    Negative score for symmetric configuration.
    """
    pts = _build_symmetric_points(params, counts)
    # Ensure valid shape
    if len(pts) != N:
        return 0.0
    return -_score_points(pts)

def _optimize_symmetric(counts, maxiter=50, popsize=10):
    """
    Optimize a specific ring configuration using Differential Evolution.
    """
    K = len(counts)
    # Bounds for radii (0 to 1) and angles (0 to 2pi)
    bounds = []
    for _ in range(K):
        bounds.append((0.0, 1.5)) # Radius can go up to 1.5 to allow hull expansion? No, normalize later. 
                                  # Actually, scale is irrelevant for ratio, but DE needs bounds.
                                  # Let's fix scale by setting max radius to 1.0? 
                                  # No, let's allow 0 to 1. The ratio is scale invariant.
                                  # But if all radii are 0, score is 0.
                                  # Let's use 0 to 1.
        bounds.append((0.0, 2 * np.pi))
    
    # Adjust bounds for radii to 0, 1
    bounds = [(0.0, 1.0), (0.0, 2 * np.pi)] * K
    
    try:
        result = differential_evolution(
            _neg_symmetric_score,
            bounds=bounds,
            args=(counts,),
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            polish=True, # Use L-BFGS for final polish
            tol=1e-5
        )
        if result.success or result.fun < 0:
            pts = _build_symmetric_points(result.x, counts)
            if len(pts) == N:
                return pts
    except Exception:
        pass
    return None

def _local_refine(pts):
    """
    Refine a starting configuration using Nelder-Mead on full 2D coordinates.
    """
    try:
        # Normalize initial points to unit area hull? 
        # The score is invariant to scaling, but Nelder-Mead might drift.
        # Let's just run it.
        result = minimize(
            _neg_score,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if result.fun is not None:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -1.0
    best_pts = None
    
    # 1. Generate diverse initializations
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts.copy())
    
    # 12-gon + center
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    pts = np.vstack([pts, center])
    inits.append(pts.copy())
    
    # 2. Symmetry-Constrained Search
    # Define several ring distributions that sum to 13
    # (counts, description)
    distributions = [
        [13],                  # Single ring
        [12, 1],               # 12 + 1
        [11, 2],               # 11 + 2
        [10, 3],               # 10 + 3
        [9, 4],                # 9 + 4
        [8, 5],                # 8 + 5
        [7, 6],                # 7 + 6
        [6, 5, 2],             # 6 + 5 + 2
        [5, 5, 3],             # 5 + 5 + 3
        [4, 4, 5],             # 4 + 4 + 5
        [3, 3, 3, 4],          # 3 + 3 + 3 + 4
        [2, 2, 2, 2, 5],       # 2+2+2+2+5
        [1, 1, 1, 1, 1, 8],    # 1*5 + 8
        [6, 7],                # 6 + 7
        [4, 9],                # 4 + 9
        [5, 8],                # 5 + 8
    ]
    
    # Evaluate initializations first
    for pts in inits:
        score = _score_points(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Run symmetric optimization for each distribution
    for counts in distributions:
        if time.time() > deadline - 20:
            break
        
        sym_pts = _optimize_symmetric(counts, maxiter=30, popsize=8)
        if sym_pts is not None:
            score = _score_points(sym_pts)
            if score > best_score:
                best_score = score
                best_pts = sym_pts.copy()
                
    # 3. Local Refinement of the best symmetric solution
    if best_pts is not None:
        if time.time() < deadline - 10:
            refined_pts = _local_refine(best_pts)
            score = _score_points(refined_pts)
            if score > best_score:
                best_score = score
                best_pts = refined_pts.copy()
                
            # Try a few random perturbations of the best symmetric solution
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                perturbed = best_pts + np.random.normal(0, 0.05, size=best_pts.shape)
                refined_pts = _local_refine(perturbed)
                score = _score_points(refined_pts)
                if score > best_score:
                    best_score = score
                    best_pts = refined_pts.copy()

    # 4. If we still have time, try a global DE on full 2D space from the best point
    if time.time() < deadline - 15 and best_pts is not None:
        # Use the best point as a seed for DE? DE doesn't take x0 in standard scipy way easily for all versions, 
        # but we can initialize the population near the best point.
        # Or just run a short DE with a tight bound around the best point.
        # Actually, let's just do a Powell method which is good for non-differentiable.
        try:
            result = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-8, 'xtol': 1e-8}
            )
            if result.fun is not None:
                candidate_pts = result.x.reshape(-1, 2)
                candidate_score = _score_points(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        return inits[0]
        
    return best_pts
# EVOLVE-BLOCK-END
