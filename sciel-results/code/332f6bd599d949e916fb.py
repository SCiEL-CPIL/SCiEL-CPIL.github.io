"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, leave margin

    def score_points(pts):
        """Compute (max_dist / min_dist)^2 for a set of points."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        min_d = np.min(d)
        max_d = np.max(d)
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d)**2

    # Generate a triangular lattice
    # We'll use a lattice with spacing 1, and search over a range of centers
    sqrt3 = np.sqrt(3)
    
    # Generate lattice points in a reasonable range
    R = 8
    lattice_pts = []
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            x = q + 0.5 * r
            y = (sqrt3 / 2) * r
            lattice_pts.append((x, y))
    
    lattice_pts = np.array(lattice_pts)
    
    # For efficiency, we'll try different approaches:
    # 1. Try hexagonal clusters of various sizes
    # 2. Use local search on the lattice
    
    best_score = 1e10
    best_pts = None
    
    # Approach 1: Try compact hexagonal-like clusters
    # A hexagonal cluster of radius k has 1 + 3k(k+1) points
    # k=1: 7 points, k=2: 19 points
    # For 16 points, we can take a radius-2 hexagon (19 pts) and remove 3
    
    # Generate all points in a radius-2 hexagon
    def hex_points(radius):
        pts = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                # Check if in hexagon
                if abs(q + r) <= radius and abs(q - r) <= radius and abs(q) <= radius and abs(r) <= radius:
                    # Actually, hexagon condition: max(|q|, |r|, |q+r|) <= radius
                    if max(abs(q), abs(r), abs(q + r)) <= radius:
                        x = q + 0.5 * r
                        y = (sqrt3 / 2) * r
                        pts.append((x, y))
        return np.array(pts)
    
    # Try radius 2 hexagon (19 points) - remove 3 to get 16
    hex2 = hex_points(2)
    n_hex2 = len(hex2)
    
    if n_hex2 >= N:
        # Try all combinations of 16 from 19 (816 combinations)
        # This is feasible
        indices = list(range(n_hex2))
        for combo in combinations(indices, N):
            if time.time() > deadline - 20:
                break
            pts = hex2[list(combo)]
            s = score_points(pts)
            if s < best_score:
                best_score = s
                best_pts = pts.copy()
    
    # Approach 2: Try radius 3 hexagon (37 points) - sample subsets
    hex3 = hex_points(3)
    n_hex3 = len(hex3)
    
    if n_hex3 >= N:
        # Try random subsets and local search
        np.random.seed(42)
        for trial in range(200):
            if time.time() > deadline - 30:
                break
            # Start with 16 random points from hex3
            idx = np.random.choice(n_hex3, N, replace=False)
            pts = hex3[idx]
            s = score_points(pts)
            if s < best_score:
                best_score = s
                best_pts = pts.copy()
    
    # Approach 3: Local search on the lattice
    # Start from best_pts and try swapping points
    if best_pts is not None:
        # Find which lattice points are in best_pts
        # Create a set of lattice point indices
        lattice_set = {}
        for i, p in enumerate(lattice_pts):
            key = (round(p[0], 6), round(p[1], 6))
            lattice_set[key] = i
        
        # Map best_pts to lattice indices
        in_best = []
        for p in best_pts:
            key = (round(p[0], 6), round(p[1], 6))
            if key in lattice_set:
                in_best.append(lattice_set[key])
        
        if len(in_best) == N:
            in_best_set = set(in_best)
            
            # Try swapping each point with nearby lattice points
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for i in range(N):
                    if time.time() > deadline - 10:
                        break
                    # Try swapping point i with nearby lattice points
                    p = best_pts[i]
                    # Find nearby lattice points (within distance 2)
                    nearby = []
                    for j, lp in enumerate(lattice_pts):
                        if j in in_best_set:
                            continue
                        d = np.linalg.norm(lp - p)
                        if d < 2.0:
                            nearby.append((d, j))
                    
                    nearby.sort()
                    for d, j in nearby[:10]:  # Try top 10 nearest
                        if time.time() > deadline - 10:
                            break
                        # Swap
                        new_pts = best_pts.copy()
                        new_pts[i] = lattice_pts[j]
                        new_score = score_points(new_pts)
                        if new_score < best_score:
                            best_score = new_score
                            best_pts = new_pts
                            in_best_set.remove(in_best[i])
                            in_best[i] = j
                            in_best_set.add(j)
                            improved = True
    
    # Approach 4: Continuous optimization refinement
    if best_pts is not None and time.time() < deadline - 5:
        # Fix first two points to remove translation/rotation invariance
        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu]
        
        min_idx = np.argmin(d)
        pair = np.unravel_index(min_idx, (N, N))
        i0, i1 = pair[0], pair[1]
        if i0 > i1:
            i0, i1 = i1, i0
        
        p0 = best_pts[i0]
        p1 = best_pts[i1]
        v = p1 - p0
        norm_v = np.linalg.norm(v)
        
        if norm_v > 1e-8:
            cos_t = v[0] / norm_v
            sin_t = v[1] / norm_v
            R_mat = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
            
            transformed = (best_pts - p0) @ R_mat.T / norm_v
            
            # Reorder
            perm = list(range(N))
            perm[0] = i0
            perm[1] = i1
            remaining = [i for i in range(N) if i != i0 and i != i1]
            perm[2:] = remaining
            transformed = transformed[perm]
            
            init_params = transformed[2:].flatten()
            
            def objective(params):
                if time.time() > deadline:
                    return 1e10
                pts = np.zeros((N, 2))
                pts[0] = [0.0, 0.0]
                pts[1] = [1.0, 0.0]
                pts[2:] = params.reshape(-1, 2)
                
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                iu = np.triu_indices(N, 1)
                d = dists[iu]
                
                min_d = np.min(d)
                max_d = np.max(d)
                
                if min_d < 1e-8:
                    return 1e10
                
                return (max_d / min_d)**2
            
            bnds = [(-10, 10)] * 28
            
            try:
                res = minimize(objective, init_params, method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if res.fun < best_score:
                    best_score = res.fun
                    final_pts = np.zeros((N, 2))
                    final_pts[0] = [0.0, 0.0]
                    final_pts[1] = [1.0, 0.0]
                    final_pts[2:] = res.x.reshape(-1, 2)
                    best_pts = final_pts
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback: return a simple configuration
        best_pts = np.zeros((N, 2))
        for i in range(N):
            best_pts[i] = [i, 0]
    
    return best_pts

# EVOLVE-BLOCK-END
