import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def fast_evaluate(points):
        pts = np.array(points, dtype=np.int64)
        p1 = pts[:, np.newaxis, np.newaxis, :]
        p2 = pts[np.newaxis, :, np.newaxis, :]
        p3 = pts[np.newaxis, np.newaxis, :, :]
        
        v1 = p2 - p1
        v2 = p3 - p1
        
        det = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
        abs_det = np.abs(det)
        
        non_zero = abs_det[abs_det > 0]
        
        if len(non_zero) == 0:
            return 0
        
        unique_areas, counts = np.unique(non_zero, return_counts=True)
        return int(np.max(counts))

    def is_valid(points):
        if len(points) != 30:
            return False
        if len(set(points)) != 30:
            return False
        for x, y in points:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                return False
        return True

    def local_search(points, time_limit):
        current_points = [p for p in points]
        current_score = fast_evaluate(current_points)
        
        while time.time() < time_limit:
            improved = False
            # Try a batch of moves
            for _ in range(20):
                if time.time() > time_limit:
                    break
                    
                idx = random.randint(0, 29)
                old_x, old_y = current_points[idx]
                
                # Small step
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                if dx == 0 and dy == 0:
                    continue
                    
                new_x = old_x + dx
                new_y = old_y + dy
                
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_points = current_points[:]
                new_points[idx] = (new_x, new_y)
                
                if len(set(new_points)) != 30:
                    continue
                
                new_score = fast_evaluate(new_points)
                
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
                    improved = True
                    break
            
            # If stuck, try a bigger jump?
            # For now, just continue.
            
        return current_points, current_score

    candidates = []
    
    # Generate parallel line configurations
    # L lines, y = 0..L-1
    # Points distributed as evenly as possible
    for L in range(2, 11):
        # Base count per line
        base = 30 // L
        rem = 30 % L
        
        pts = []
        valid = True
        for i in range(L):
            n_i = base + (1 if i < rem else 0)
            # AP: 0..n_i-1
            # Shift to center? 0..n_i-1 is fine for small n_i
            for x in range(n_i):
                pts.append((x, i))
                
        if len(pts) == 30 and is_valid(pts):
            candidates.append(pts)
            
        # Also try shifted APs?
        # e.g. x = 1..n_i
        # Or x = -n_i//2 .. n_i//2
        # The relative differences are the same, so the area counts are the same.
        # So one representative per L is enough.
        
    # Add Grids just in case
    for w in range(3, 11):
        for h in range(3, 11):
            if w * h == 30:
                pts = [(x, y) for x in range(w) for y in range(h)]
                if is_valid(pts):
                    candidates.append(pts)

    # Evaluate
    best_points = None
    best_score = 0
    
    for pts in candidates:
        score = fast_evaluate(pts)
        if score > best_score:
            best_score = score
            best_points = pts
        if time.time() > deadline - 5.0:
            break
            
    if best_points is None:
        best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
        best_score = fast_evaluate(best_points)

    # Local Search
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        best_points, best_score = local_search(best_points, deadline - 1.0)

    return best_points

# EVOLVE-BLOCK-END
