import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    N = 30
    # Precompute triple indices
    # i < j < k
    idx_i, idx_j, idx_k = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                idx_i.append(i)
                idx_j.append(j)
                idx_k.append(k)
    idx_i = np.array(idx_i)
    idx_j = np.array(idx_j)
    idx_k = np.array(idx_k)
    
    def evaluate(points):
        # points: (N, 2) array
        xi = points[idx_i, 0]
        xj = points[idx_j, 0]
        xk = points[idx_k, 0]
        yi = points[idx_i, 1]
        yj = points[idx_j, 1]
        yk = points[idx_k, 1]
        
        # Determinant = (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
        # = xj*yk - xj*yi - xi*yk + xi*yi - yj*xk + yj*xi + yi*xk - yi*xi
        # = xj*yk - xj*yi - xi*yk - yj*xk + yj*xi + yi*xk
        # Simplified:
        det = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        abs_det = np.abs(det)
        
        # Exclude zero area
        non_zero_mask = abs_det > 0
        if not np.any(non_zero_mask):
            return 0
        
        areas = abs_det[non_zero_mask]
        # Count frequencies
        # Max area is bounded. (60*60)*2 = 7200.
        # We can use np.bincount for speed
        # Ensure max value is reasonable for bincount
        max_area = np.max(areas)
        if max_area > 10000: # Safety check
             # Fallback to unique counts if area is huge (unlikely in [-30,30])
             _, counts = np.unique(areas, return_counts=True)
             return np.max(counts)
        
        counts = np.bincount(areas)
        return np.max(counts)
        
    def random_grid(w, h):
        # Centered grid
        xs = np.arange(-w//2, w - w//2)
        ys = np.arange(-h//2, h - h//2)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack((xx.ravel(), yy.ravel()))
        return pts
        
    # Try a few initial configurations
    candidates = []
    # 5x6 grid
    candidates.append(random_grid(6, 5))
    # 4x8 grid (if it fits)
    candidates.append(random_grid(8, 4))
    # 3x10 grid
    candidates.append(random_grid(10, 3))
    # 2x15 grid
    candidates.append(random_grid(15, 2))
    # 1x30 grid (degenerate, area 0 for all, score 0) - skip
    
    best_score = 0
    best_points = None
    
    for pts in candidates:
        # Ensure distinct and in bounds
        if len(np.unique(pts, axis=0)) != N:
            continue
        if np.any(pts < -30) or np.any(pts > 30):
            continue
            
        score = evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    if best_points is None:
        # Fallback to simple grid
        best_points = random_grid(6, 5)
        best_score = evaluate(best_points)
        
    # Simulated Annealing
    current_points = best_points
    current_score = best_score
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, N - 1)
        
        # Propose a move: small random integer shift
        dx = random.randint(-2, 2)
        dy = random.randint(-2, 2)
        if dx == 0 and dy == 0:
            continue
            
        new_x = current_points[idx, 0] + dx
        new_y = current_points[idx, 1] + dy
        
        # Check bounds
        if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
            continue
            
        # Check distinctness
        # Simple check: if new point equals any existing point (except itself), reject
        # We can check this efficiently
        new_pt = np.array([new_x, new_y])
        # Check against all other points
        # dists = np.linalg.norm(current_points - new_pt, axis=1)
        # if np.min(dists) < 1e-6 and not (current_points[idx] == new_pt).all():
        #     continue
        # Faster:
        others = np.delete(current_points, idx, axis=0)
        if np.any(np.all(others == new_pt, axis=1)):
            continue
            
        # Apply move
        old_pt = current_points[idx].copy()
        current_points[idx] = new_pt
        
        new_score = evaluate(current_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_points = current_points.copy()
        else:
            # Accept with probability exp(-delta/T)
            delta = current_score - new_score
            if delta > 0:
                p = np.exp(-delta / T)
                if random.random() > p:
                    # Reject
                    current_points[idx] = old_pt
            else:
                # This case shouldn't happen if new_score <= current_score and delta > 0
                # But if new_score == current_score, delta=0, p=1, accept
                pass
                
        T *= cooling
        
    # Convert numpy array to list of tuples
    return [(int(x), int(y)) for x, y in best_points]
