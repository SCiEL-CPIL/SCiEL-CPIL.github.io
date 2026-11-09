import numpy as np
import time
import random

def _is_feasible(centers, sides, tol=1e-7):
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def _max_side_given_others(centers, sides, i, tol=1e-7):
    """Calculate max side for square i given others are fixed."""
    n = len(sides)
    cx, cy = centers[i]
    s = sides[i]
    
    # Boundary constraints
    max_s_boundary = min(2*cx, 2*(1-cx), 2*cy, 2*(1-cy))
    
    max_s = max_s_boundary
    for j in range(n):
        if j == i: continue
        cx_j, cy_j = centers[j]
        s_j = sides[j]
        
        # Distance to center of j
        dx = abs(cx - cx_j)
        dy = abs(cy - cy_j)
        
        # We need |cx - cx_j| >= (s+s_j)/2 OR |cy - cy_j| >= (s+s_j)/2
        # This means either s <= 2*dx - s_j OR s <= 2*dy - s_j
        # So s <= min(2*dx - s_j, 2*dy - s_j) is NOT correct.
        # It is s <= (2*dx - s_j) IF we separate in x.
        # OR s <= (2*dy - s_j) IF we separate in y.
        # We can choose the better separation.
        # So the constraint is s <= max(2*dx - s_j, 2*dy - s_j) ? No.
        # If we separate in x, we need cx - s/2 >= cx_j + s_j/2 OR cx + s/2 <= cx_j - s_j/2.
        # This implies s <= 2*dx - s_j.
        # Similarly for y.
        # Since we can choose to separate in x OR y, the max allowed s is the max of the two limits?
        # Wait. If s is large, we might not be able to separate in x. But we can separate in y.
        # So the condition is: (2*dx - s_j >= s) OR (2*dy - s_j >= s).
        # This is equivalent to: s <= max(2*dx - s_j, 2*dy - s_j).
        # Let's verify.
        # If s <= 2*dx - s_j, we can separate in x.
        # If s <= 2*dy - s_j, we can separate in y.
        # If either is true, we are feasible.
        # So the max s allowed by pair (i,j) is indeed max(2*dx - s_j, 2*dy - s_j).
        
        limit = max(2*dx - s_j, 2*dy - s_j)
        if limit < max_s:
            max_s = limit
            
    return max(0.0, max_s)

def _local_refine(centers, sides, tol=1e-7, max_iter=1000, deadline=None):
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    if deadline is None:
        deadline = time.time() + 100
        
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        improved = False
        
        # 1. Grow each square to max possible size
        for i in range(n):
            new_s = _max_side_given_others(best_centers, best_sides, i, tol)
            if new_s > best_sides[i] + 1e-9:
                best_sides[i] = new_s
                improved = True
        
        # 2. Try to move squares to improve packing
        # Simple heuristic: try to move each square slightly in 4 directions
        # and see if it allows for a larger total sum after regrowth
        for i in range(n):
            s = best_sides[i]
            cx, cy = best_centers[i]
            
            # Check if we can move
            # We want to move to a position that allows this square or others to be larger.
            # For simplicity, just try small random moves or grid moves.
            # Given N=10, we can afford a small grid search.
            for dx in [-0.05, 0.05]:
                for dy in [-0.05, 0.05]:
                    new_cx = cx + dx
                    new_cy = cy + dy
                    # Check bounds
                    if new_cx - s/2 < 0 or new_cx + s/2 > 1: continue
                    if new_cy - s/2 < 0 or new_cy + s/2 > 1: continue
                    
                    new_centers = best_centers.copy()
                    new_centers[i] = [new_cx, new_cy]
                    
                    # Check if this move allows for a better configuration
                    # We simulate regrowing all squares
                    sim_sides = best_sides.copy()
                    sim_centers = new_centers.copy()
                    
                    # Regrow all squares
                    for k in range(n):
                        sim_sides[k] = _max_side_given_others(sim_centers, sim_sides, k, tol)
                        
                    if np.sum(sim_sides) > best_val + 1e-7:
                        best_centers = sim_centers
                        best_sides = sim_sides
                        best_val = np.sum(best_sides)
                        improved = True
                        break
                if improved: break
            if improved: break
            
        if not improved:
            break
            
    return best_centers, best_sides

def _gap_filling_pack(random_seed=0):
    """
    Construct a packing using a greedy gap-filling heuristic.
    Returns centers and sides.
    """
    random.seed(random_seed)
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # List of empty rectangles: (x_min, y_min, x_max, y_max)
    empty_rects = [(0.0, 0.0, 1.0, 1.0)]
    
    for k in range(n):
        best_rect_idx = -1
        best_size = 0
        best_cx = 0
        best_cy = 0
        
        for idx, (x1, y1, x2, y2) in enumerate(empty_rects):
            w = x2 - x1
            h = y2 - y1
            s = min(w, h)
            if s > best_size:
                best_size = s
                best_rect_idx = idx
                best_cx = (x1 + x2) / 2
                best_cy = (y1 + y2) / 2
            elif s == best_size and s > 0:
                # Tie-breaking: randomize to explore different packings
                if random.random() < 0.5:
                    best_size = s
                    best_rect_idx = idx
                    best_cx = (x1 + x2) / 2
                    best_cy = (y1 + y2) / 2
        
        if best_rect_idx == -1:
            break
            
        centers[k] = [best_cx, best_cy]
        sides[k] = best_size
        
        # Update empty rectangles
        x1, y1, x2, y2 = empty_rects[best_rect_idx]
        cx, cy = centers[k]
        s = sides[k]
        
        new_rects = []
        # Left
        if cx - s/2 > x1:
            new_rects.append((x1, y1, cx - s/2, y2))
        # Right
        if cx + s/2 < x2:
            new_rects.append((cx + s/2, y1, x2, y2))
        # Bottom
        if cy - s/2 > y1:
            new_rects.append((cx - s/2, y1, cx + s/2, cy - s/2))
        # Top
        if cy + s/2 < y2:
            new_rects.append((cx - s/2, cy + s/2, cx + s/2, y2))
        
        # Replace the used rectangle with the new ones
        empty_rects[best_rect_idx] = None # Mark for removal
        empty_rects = [r for r in empty_rects if r is not None]
        empty_rects.extend(new_rects)
        
        # Optional: Merge adjacent rectangles? Not strictly necessary for N=10.
        
    return centers, sides

def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 575
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # 1. Try the known good 3.0 packing
    c0 = np.array([
        [0.25, 0.75],
        [0.75, 0.75],
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
    ])
    s0 = np.array([0.5, 0.5] + [0.25]*8)
    if _is_feasible(c0, s0):
        best_centers = c0.copy()
        best_sides = s0.copy()
        best_val = np.sum(s0)
        
    # 2. Try multiple gap-filling packings with different seeds
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        c, s = _gap_filling_pack(random_seed=i)
        
        # Refine this packing
        refine_deadline = min(deadline - 10, time.time() + 15)
        try:
            rc, rs = _local_refine(c, s, max_iter=200, deadline=refine_deadline)
            if _is_feasible(rc, rs):
                val = np.sum(rs)
                if val > best_val:
                    best_val = val
                    best_centers = rc.copy()
                    best_sides = rs.copy()
        except Exception:
            pass
            
    # 3. Final polish on the best found
    if best_centers is not None and time.time() < deadline - 5:
        refine_deadline = min(deadline - 2, time.time() + 30)
        try:
            rc, rs = _local_refine(best_centers, best_sides, max_iter=500, deadline=refine_deadline)
            if _is_feasible(rc, rs):
                val = np.sum(rs)
                if val > best_val:
                    best_val = val
                    best_centers = rc.copy()
                    best_sides = rs.copy()
        except Exception:
            pass
            
    return best_centers, best_sides
