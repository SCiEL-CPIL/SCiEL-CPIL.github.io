"""Combined program: Hexagonal lattice subset enumeration with Powell refinement."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import itertools
import math

DEADLINE = 115.0  # Safety margin

def _score(P):
    """
    Calculate (max_dist / min_dist)^2 for a set of points.
    pts: np.array of shape (N, 2)
    """
    if P.shape[0] != N:
        return 1e10
    # Compute pairwise squared distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Get upper triangle indices to avoid self-distances and duplicates
    iu = np.triu_indices(N, 1)
    d_sq = dists_sq[iu]
    
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _generate_hex_cluster():
    """
    Generate the 19-point hexagonal cluster (center + 2 rings).
    Ring 0: 1 point (dist 0)
    Ring 1: 6 points (dist 1)
    Ring 2: 12 points (dist sqrt(3) and 2)
    Total 19 points.
    """
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    hex_points = []
    # Iterate over a range that covers the cluster
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            d_sq = np.sum(p**2)
            # The 19-point cluster includes all points with dist_sq <= 4
            if d_sq <= 4.0 + 1e-9:
                hex_points.append(p)
    
    return np.array(hex_points)

def construct():
    t_start = time.time()
    
    # 1. Generate the 19-point hexagonal cluster
    hex_points = _generate_hex_cluster()
    
    if len(hex_points) != 19:
        # Fallback to a grid if generation fails
        P = np.array([[i % 4 - 1.5, i // 4 - 1.5] for i in range(N)], dtype=float)
        return P

    # 2. Identify outer ring points (dist_sq == 4.0)
    d_sqs = np.sum(hex_points**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    
    # There should be 12 outer points
    if len(outer_indices) != 12:
        # Fallback
        P = np.array([[i % 4 - 1.5, i // 4 - 1.5] for i in range(N)], dtype=float)
        return P

    # 3. Enumerate all combinations of removing 3 points from the outer ring
    # This leaves 19 - 3 = 16 points.
    # C(12, 3) = 220 combinations.
    best_candidates = []
    
    for remove_idx in itertools.combinations(outer_indices, 3):
        remove_set = set(remove_idx)
        keep_indices = [i for i in range(19) if i not in remove_set]
        pts = hex_points[keep_indices]
        score = _score(pts)
        best_candidates.append((score, pts))
        
        # Check time periodically
        if time.time() > DEADLINE - 30:
            break

    # Sort candidates by score
    best_candidates.sort(key=lambda x: x[0])
    
    # 4. Refine the top K candidates
    # We will refine the top 5 candidates to ensure we find the global optimum
    # within this structural family.
    K = 5
    best_score = float('inf')
    best_points = None
    
    for i in range(min(K, len(best_candidates))):
        if time.time() > DEADLINE - 10:
            break
            
        init_score, init_pts = best_candidates[i]
        
        # Use Powell method for refinement
        # Powell is derivative-free and robust for this type of non-smooth objective
        try:
            # We optimize all 32 coordinates (16 points * 2)
            # To reduce degrees of freedom, we could fix one point at origin and one on x-axis,
            # but for robustness and simplicity, we optimize all.
            # However, translation and rotation invariance means the optimizer might drift.
            # It's better to fix p0 at (0,0) and p1 at (1,0) if possible, or just let it optimize.
            # Given the high symmetry, fixing is safer.
            
            # Let's fix p0 and p1 to remove 4 DOF.
            # Find the pair with minimum distance in init_pts to use as the reference pair.
            diff = init_pts[:, np.newaxis, :] - init_pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            iu = np.triu_indices(N, 1)
            d_sq = dists_sq[iu]
            
            min_idx = np.argmin(d_sq)
            p, q = np.unravel_index(min_idx, (N, N))
            if p > q:
                p, q = q, p
            
            p0 = init_pts[p]
            p1 = init_pts[q]
            
            v = p1 - p0
            norm_v = np.linalg.norm(v)
            
            if norm_v < 1e-8:
                # Degenerate case, skip
                continue
                
            # Rotation matrix to align v with x-axis
            cos_t = v[0] / norm_v
            sin_t = v[1] / norm_v
            R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
            
            # Transform all points
            transformed = (init_pts - p0) @ R_mat.T / norm_v
            
            # Reorder points so that p is at index 0 and q is at index 1
            perm = list(range(N))
            perm[0] = p
            perm[1] = q
            remaining = [idx for idx in range(N) if idx != p and idx != q]
            perm[2:] = remaining
            
            transformed = transformed[perm]
            
            # init_params are the coordinates of points 2..15
            init_params = transformed[2:].flatten()
            
            # Define objective function
            def objective(params):
                pts = np.zeros((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = params.reshape(-1, 2)
                
                # Quick check for distinctness
                # If points are too close, return large value
                d2 = _score(pts)
                return d2
            
            # Optimize
            res = minimize(objective, init_params, method='Powell', 
                           options={'maxiter': 100, 'maxfev': 1000, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                # Reconstruct points
                final_pts = np.zeros((N, 2))
                final_pts[0] = [0.0, 0.0]
                final_pts[1] = [1.0, 0.0]
                final_pts[2:] = res.x.reshape(-1, 2)
                
                # Verify score
                actual_score = _score(final_pts)
                if actual_score < best_score:
                    best_score = actual_score
                    best_points = final_pts
                    
        except Exception:
            pass

    # 5. Final sanity check
    if best_points is None:
        # Fallback to the best unoptimized candidate
        if best_candidates:
            best_points = best_candidates[0][1]
        else:
            best_points = np.array([[i % 4 - 1.5, i // 4 - 1.5] for i in range(N)], dtype=float)
            
    # Ensure distinctness
    d2 = _score(best_points)
    if d2 > 1e10:
        # Perturb if invalid
        np.random.seed(42)
        best_points += np.random.randn(N, D) * 1e-5
        
    return best_points
# EVOLVE-BLOCK-END
