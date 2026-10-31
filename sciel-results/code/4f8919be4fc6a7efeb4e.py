"""Improved program: Hexagonal lattice subset (17 points) with refinement."""
N = 16
D = 2

import time
import numpy as np
from scipy.optimize import minimize
import itertools

DEADLINE = 115.0

def _score(P):
    """
    Calculate (max_dist / min_dist)^2 for a set of points.
    """
    if P.shape[0] != N:
        return 1e10
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    d_sq = dists_sq[iu]
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _generate_hex_cluster_17():
    """
    Generate the 17-point hexagonal cluster (center + 2 rings).
    """
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            d_sq = np.sum(p**2)
            if d_sq <= 4.0 + 1e-9:
                hex_points.append(p)
    return np.array(hex_points)

def construct():
    t_start = time.time()
    
    # 1. Generate the 17-point hexagonal cluster
    hex_points = _generate_hex_cluster_17()
    
    if len(hex_points) != 17:
        # Fallback to a 4x4 grid if generation fails
        P = np.array([[i % 4 - 1.5, i // 4 - 1.5] for i in range(N)], dtype=float)
        return P

    # 2. Enumerate all combinations of removing 1 point from the 17
    best_candidates = []
    for remove_idx in range(17):
        keep_indices = [i for i in range(17) if i != remove_idx]
        pts = hex_points[keep_indices]
        score = _score(pts)
        best_candidates.append((score, pts))
        
    # Sort candidates by score
    best_candidates.sort(key=lambda x: x[0])
    
    # 3. Refine the top K candidates
    K = 5
    best_score = float('inf')
    best_points = None
    
    for i in range(min(K, len(best_candidates))):
        if time.time() > DEADLINE - 10:
            break
            
        init_score, init_pts = best_candidates[i]
        
        try:
            # Fix p0 and p1 to remove translational and rotational invariance
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
                continue
                
            cos_t = v[0] / norm_v
            sin_t = v[1] / norm_v
            R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
            
            transformed = (init_pts - p0) @ R_mat.T / norm_v
            
            perm = list(range(N))
            perm[0] = p
            perm[1] = q
            remaining = [idx for idx in range(N) if idx != p and idx != q]
            perm[2:] = remaining
            
            transformed = transformed[perm]
            
            init_params = transformed[2:].flatten()
            
            def objective(params):
                pts = np.zeros((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = params.reshape(-1, 2)
                return _score(pts)
            
            res = minimize(objective, init_params, method='Powell', 
                           options={'maxiter': 100, 'maxfev': 1000, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                final_pts = np.zeros((N, 2))
                final_pts[0] = [0.0, 0.0]
                final_pts[1] = [1.0, 0.0]
                final_pts[2:] = res.x.reshape(-1, 2)
                
                actual_score = _score(final_pts)
                if actual_score < best_score:
                    best_score = actual_score
                    best_points = final_pts
                    
        except Exception:
            pass

    if best_points is None:
        best_points = best_candidates[0][1]
        
    return best_points
