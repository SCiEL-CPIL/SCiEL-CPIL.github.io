import numpy as np
import time

def squares_in_square10() -> tuple:
    n = 10
    best_score = 0
    best_centers = None
    best_sides = None
    
    start_time = time.time()
    time_limit = 590  # Safety margin
    
    def is_valid(centers, sides):
        # Bounds
        if np.any(centers[:, 0] < sides / 2 - 1e-9) or np.any(centers[:, 0] > 1 - sides / 2 + 1e-9):
            return False
        if np.any(centers[:, 1] < sides / 2 - 1e-9) or np.any(centers[:, 1] > 1 - sides / 2 + 1e-9):
            return False
        
        # Overlap
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        # Vectorized overlap check
        # dx[i, j] = |cx[i] - cx[j]|
        dx = np.abs(cx[:, np.newaxis] - cx[np.newaxis, :])
        dy = np.abs(cy[:, np.newaxis] - cy[np.newaxis, :])
        
        # req[i, j] = (s[i] + s[j]) / 2
        req = (s[:, np.newaxis] + s[np.newaxis, :]) / 2
        
        # Overlap if dx < req AND dy < req
        # We want NO overlap, so we want (dx >= req) OR (dy >= req) for all i != j
        # Note: dx[i,i] = 0, req[i,i] = s[i]. 0 < s[i] is true. 
        # But we don't care about i==j.
        # Mask out diagonal
        mask = ~np.eye(n, dtype=bool)
        
        # Check if any pair overlaps
        # overlap_mask[i, j] = (dx[i, j] < req[i, j]) AND (dy[i, j] < req[i, j])
        overlap_mask = (dx < req) & (dy < req)
        
        # If any off-diagonal element is True, invalid
        if np.any(overlap_mask[mask]):
            return False
            
        return True

    def get_score(centers, sides):
        return np.sum(sides)

    def local_search(centers, sides, max_iter=10000):
        current_score = get_score(centers, sides)
        step_size = 0.05
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
                
            improved = False
            # Randomly shuffle order of operations to avoid bias
            order = np.random.permutation(n)
            
            for i in order:
                # Try moving center x
                for delta in [step_size, -step_size]:
                    new_cx = centers[i, 0] + delta
                    # Check bounds quickly
                    if new_cx < sides[i] / 2 or new_cx > 1 - sides[i] / 2:
                        continue
                    new_centers = centers.copy()
                    new_centers[i, 0] = new_cx
                    if is_valid(new_centers, sides):
                        new_score = get_score(new_centers, sides)
                        if new_score > current_score:
                            centers = new_centers
                            current_score = new_score
                            improved = True
                            step_size = max(0.01, step_size * 0.9) # Anneal step size
                            
                # Try moving center y
                for delta in [step_size, -step_size]:
                    new_cy = centers[i, 1] + delta
                    if new_cy < sides[i] / 2 or new_cy > 1 - sides[i] / 2:
                        continue
                    new_centers = centers.copy()
                    new_centers[i, 1] = new_cy
                    if is_valid(new_centers, sides):
                        new_score = get_score(new_centers, sides)
                        if new_score > current_score:
                            centers = new_centers
                            current_score = new_score
                            improved = True
                            step_size = max(0.01, step_size * 0.9)

                # Try changing side
                for delta in [step_size, -step_size]:
                    new_s = sides[i] + delta
                    if new_s <= 0:
                        continue
                    # Check bounds for new side
                    if new_s > 1:
                        continue
                    if centers[i, 0] < new_s / 2 or centers[i, 0] > 1 - new_s / 2:
                        continue
                    if centers[i, 1] < new_s / 2 or centers[i, 1] > 1 - new_s / 2:
                        continue
                        
                    new_sides = sides.copy()
                    new_sides[i] = new_s
                    if is_valid(centers, new_sides):
                        new_score = get_score(centers, new_sides)
                        if new_score > current_score:
                            sides = new_sides
                            current_score = new_score
                            improved = True
                            step_size = max(0.01, step_size * 0.9)
                            
            if not improved:
                # Try a random kick to escape local minimum
                # Pick a random square and move it randomly, even if score drops, 
                # but only if it's valid. Then continue.
                # This is basic SA.
                i = np.random.randint(n)
                delta_x = np.random.uniform(-0.1, 0.1)
                delta_y = np.random.uniform(-0.1, 0.1)
                delta_s = np.random.uniform(-0.05, 0.05)
                
                new_cx = centers[i, 0] + delta_x
                new_cy = centers[i, 1] + delta_y
                new_s = sides[i] + delta_s
                
                if new_s > 0 and new_s <= 1:
                    if new_cx >= new_s / 2 and new_cx <= 1 - new_s / 2:
                        if new_cy >= new_s / 2 and new_cy <= 1 - new_s / 2:
                            new_centers = centers.copy()
                            new_centers[i, 0] = new_cx
                            new_centers[i, 1] = new_cy
                            new_sides = sides.copy()
                            new_sides[i] = new_s
                            
                            if is_valid(new_centers, new_sides):
                                # Accept with some probability (SA) or always if valid?
                                # For now, always accept if valid to explore.
                                # But this might degrade score. 
                                # We track best separately, so this is fine.
                                centers = new_centers
                                sides = new_sides
                                current_score = get_score(centers, sides)
                                step_size = 0.05 # Reset step size
                                
        return centers, sides, current_score

    def init_config():
        # Start with grid
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(n, side)
        
        # Randomize slightly
        # Jitter centers
        centers[:, 0] += np.random.uniform(-0.05, 0.05, n)
        centers[:, 1] += np.random.uniform(-0.05, 0.05, n)
        # Jitter sides
        sides += np.random.uniform(-0.02, 0.02, n)
        
        # Ensure validity, otherwise revert to grid
        if not is_valid(centers, sides):
             centers = np.stack([cx, cy], axis=1)
             sides = np.full(n, side)
             
        return centers, sides

    # Main loop
    while time.time() - start_time < time_limit:
        c, s = init_config()
        c, s, score = local_search(c, s)
        
        if score > best_score:
            best_score = score
            best_centers = c.copy()
            best_sides = s.copy()

    return best_centers, best_sides
