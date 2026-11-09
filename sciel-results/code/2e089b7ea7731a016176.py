"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area
    return min_area


def _min_pairwise_distance(points):
    """Compute the minimum pairwise distance."""
    n = len(points)
    min_d = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_d:
                min_d = d
    return min_d


def _project_to_triangle(p):
    """Project point p onto the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    
    if y < 0:
        y = 0
    if y > h:
        y = h
    
    # Left boundary: x >= y / sqrt(3)
    if x < y / np.sqrt(3):
        x = y / np.sqrt(3)
    
    # Right boundary: x <= 1 - y / sqrt(3)
    if x > 1.0 - y / np.sqrt(3):
        x = 1.0 - y / np.sqrt(3)
    
    return np.array([x, y])


def construct():
    """Construct 11 points in the unit equilateral triangle to maximize minimum triangle area."""
    h = np.sqrt(3) / 2
    
    # Phase 1: Generate a dense set of candidate points
    # Use a grid in Cartesian coordinates, filtered to the triangle
    grid_res = 40
    xs = np.linspace(0, 1, grid_res)
    ys = np.linspace(0, h, grid_res)
    
    candidates = []
    for x in xs:
        for y in ys:
            if y <= np.sqrt(3) * x and y <= np.sqrt(3) * (1 - x):
                candidates.append(np.array([x, y]))
    
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # Phase 2: Find good initial configurations by maximizing minimum pairwise distance
    # Use furthest-point sampling with multiple random restarts
    best_packing_points = None
    best_packing_dist = 0.0
    
    n_restarts = 100
    for r in range(n_restarts):
        np.random.seed(1000 + r)
        picked_indices = []
        remaining_indices = list(range(n_cand))
        
        # Start with a random point
        start_idx = np.random.choice(remaining_indices)
        picked_indices.append(start_idx)
        remaining_indices.remove(start_idx)
        
        while len(picked_indices) < N:
            # Calculate min distance to already picked points for all remaining
            # Vectorized for speed
            picked_pts = candidates[picked_indices]
            # dists[i, j] is distance between candidates[remaining_indices[i]] and picked_pts[j]
            # This might be memory intensive if n_cand is large, but 1600 candidates * 11 picked is fine.
            # Actually, let's do it in a loop to be safe with memory and clarity, or use broadcasting.
            # Broadcasting: candidates[remaining] shape (M, 2), picked_pts shape (K, 2)
            # diff shape (M, K, 2)
            diff = candidates[remaining_indices][:, np.newaxis, :] - picked_pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_dists = np.min(dists, axis=1)
            
            best_idx_local = np.argmax(min_dists)
            best_idx_global = remaining_indices[best_idx_local]
            picked_indices.append(best_idx_global)
            remaining_indices.remove(best_idx_global)
        
        pts = candidates[picked_indices]
        min_dist = _min_pairwise_distance(pts)
        
        if min_dist > best_packing_dist:
            best_packing_dist = min_dist
            best_packing_points = pts.copy()
    
    # Phase 3: Local polish to maximize minimum triangle area
    # We will polish the best packing found, and also a few other good packings if time permits.
    # For simplicity and speed, we just polish the best one found in Phase 2.
    # However, the best min-distance packing might not be the best min-area configuration.
    # Let's keep the top 5 packings and polish them.
    
    # Re-run to collect top 5 packings
    packings = []
    for r in range(n_restarts):
        np.random.seed(1000 + r)
        picked_indices = []
        remaining_indices = list(range(n_cand))
        
        start_idx = np.random.choice(remaining_indices)
        picked_indices.append(start_idx)
        remaining_indices.remove(start_idx)
        
        while len(picked_indices) < N:
            picked_pts = candidates[picked_indices]
            diff = candidates[remaining_indices][:, np.newaxis, :] - picked_pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            min_dists = np.min(dists, axis=1)
            best_idx_local = np.argmax(min_dists)
            best_idx_global = remaining_indices[best_idx_local]
            picked_indices.append(best_idx_global)
            remaining_indices.remove(best_idx_global)
        
        pts = candidates[picked_indices]
        min_dist = _min_pairwise_distance(pts)
        packings.append((min_dist, pts))
        
    # Sort by min_dist descending
    packings.sort(key=lambda x: x[0], reverse=True)
    top_packings = [p[1] for p in packings[:5]]
    
    best_final_points = top_packings[0].copy()
    best_final_area = _min_triangle_area(best_final_points)
    
    def neg_min_area(x_flat):
        pts = x_flat.reshape(N, 2)
        pts_clipped = np.array([_project_to_triangle(p) for p in pts])
        return -_min_triangle_area(pts_clipped)
    
    for init_pts in top_packings:
        x0 = init_pts.flatten()
        
        # Run Nelder-Mead
        result = minimize(neg_min_area, x0, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        trial_points = result.x.reshape(N, 2)
        trial_points = np.array([_project_to_triangle(p) for p in trial_points])
        trial_area = _min_triangle_area(trial_points)
        
        if trial_area > best_final_area:
            best_final_area = trial_area
            best_final_points = trial_points
    
    return best_final_points

# EVOLVE-BLOCK-END
