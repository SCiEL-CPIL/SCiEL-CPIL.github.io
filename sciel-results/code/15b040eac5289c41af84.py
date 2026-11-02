import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def compute_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area == 0:
        return 0.0

    min_area = float('inf')
    n = len(points)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
                if area < min_area:
                    min_area = area

    return min_area / hull_area

def construct():
    start_time = time.time()
    timeout = 100.0
    
    # Grid parameters
    # A larger grid allows for better approximation of the continuous optimum.
    # 60x60 = 3600 points.
    grid_size = 60
    grid_points = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)], dtype=float)
    num_grid_points = len(grid_points)
    
    # Mapping from grid index to (row, col) for neighbor lookup
    # We'll just use index arithmetic.
    # Index = row * grid_size + col
    
    best_score = 0.0
    best_points = None
    
    # Precompute neighbors for each grid point
    # neighbor_indices[i] is a list of valid neighbor indices
    neighbors = [[] for _ in range(num_grid_points)]
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            # 4-neighbors
            if i > 0: neighbors[idx].append((i-1)*grid_size + j)
            if i < grid_size-1: neighbors[idx].append((i+1)*grid_size + j)
            if j > 0: neighbors[idx].append(i*grid_size + (j-1))
            if j < grid_size-1: neighbors[idx].append(i*grid_size + (j+1))
            # 8-neighbors (diagonals) might help escape local minima
            if i > 0 and j > 0: neighbors[idx].append((i-1)*grid_size + (j-1))
            if i > 0 and j < grid_size-1: neighbors[idx].append((i-1)*grid_size + (j+1))
            if i < grid_size-1 and j > 0: neighbors[idx].append((i+1)*grid_size + (j-1))
            if i < grid_size-1 and j < grid_size-1: neighbors[idx].append((i+1)*grid_size + (j+1))

    def evaluate_indices(indices):
        """Compute score for a set of grid indices."""
        pts = grid_points[indices]
        return compute_score(pts)

    while time.time() - start_time < timeout:
        # Random initialization
        current_indices = np.random.choice(num_grid_points, N, replace=False)
        current_score = evaluate_indices(current_indices)
        
        # Hill Climbing
        improved = True
        while improved and time.time() - start_time < timeout:
            improved = False
            best_move_idx = -1
            best_move_target = -1
            best_move_score = current_score
            
            # Try to move each point
            for k in range(N):
                old_idx = current_indices[k]
                for neighbor_idx in neighbors[old_idx]:
                    # Check if neighbor is already occupied
                    if neighbor_idx in current_indices:
                        continue
                    
                    # Try the move
                    new_indices = current_indices.copy()
                    new_indices[k] = neighbor_idx
                    new_score = evaluate_indices(new_indices)
                    
                    if new_score > best_move_score:
                        best_move_score = new_score
                        best_move_idx = k
                        best_move_target = neighbor_idx
            
            if best_move_idx != -1:
                current_indices[best_move_idx] = best_move_target
                current_score = best_move_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_points = grid_points[current_indices].copy()
        
        # If we found a good solution, we can try to polish it continuously
        # But we must be careful with time.
    
    # Final Polish
    if best_points is not None and time.time() - start_time < timeout - 5:
        def neg_score(p):
            pts = p.reshape(N, 2)
            return -compute_score(pts)
            
        try:
            res = minimize(neg_score, best_points.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
            if -res.fun > best_score:
                best_points = res.x.reshape(N, 2)
                best_score = -res.fun
        except Exception:
            pass

    if best_points is None:
        # Fallback
        return np.random.rand(N, 2)
        
    return best_points
