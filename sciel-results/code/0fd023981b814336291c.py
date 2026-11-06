import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    def score(points):
        """Calculate the maximum multiplicity of a positive doubled area."""
        n = len(points)
        if n < 3:
            return 0
        areas = Counter()
        # Optimization: Precompute differences? 
        # For N=30, N^3 is 27000. 27000 * (few ops) is fast.
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    det = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                    if det > 0:
                        areas[det] += 1
        if not areas:
            return 0
        return max(areas.values())

    def make_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for x in range(w):
            for y in range(h):
                pts.append((x + offset_x, y + offset_y))
        return pts

    candidates = []
    
    # 1. Seed: 5x6 grid
    candidates.append(make_grid(6, 5))
    
    # 2. 4x8 grid (32 points), remove 2
    grid_48 = make_grid(8, 4) # 8 wide, 4 high
    # Remove corners: (0,0), (7,3)
    pts_48_c = [p for p in grid_48 if p != (0,0) and p != (7,3)]
    candidates.append(pts_48_c)
    # Remove other corners: (0,3), (7,0)
    pts_48_c2 = [p for p in grid_48 if p != (0,3) and p != (7,0)]
    candidates.append(pts_48_c2)
    # Remove edge centers
    pts_48_e = [p for p in grid_48 if p != (3,0) and p != (4,3)]
    candidates.append(pts_48_e)
    
    # 3. 3x10 grid (30 points)
    candidates.append(make_grid(10, 3))
    
    # 4. 2x15 grid (30 points)
    candidates.append(make_grid(15, 2))
    
    # 5. 5x7 grid (35 points), remove 5
    grid_57 = make_grid(7, 5)
    # Remove corners
    corners = [(0,0), (6,0), (0,4), (6,4)]
    # Need to remove 1 more. Remove (3,0)
    pts_57_c = [p for p in grid_57 if p not in corners and p != (3,0)]
    candidates.append(pts_57_c)
    
    # 6. 6x5 grid is same as 5x6.
    
    # 7. 4x7 grid (28 points) + 2 extra? 
    # Let's stick to full grids or simple removals.
    
    # 8. 5x6 grid shifted? No, relative positions matter.
    
    best_score = 0
    best_points = candidates[0]
    
    for pts in candidates:
        if len(pts) != 30:
            continue
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
        if time.time() > deadline:
            break
            
    # Local Search on the best candidate
    # Try swapping a point with a nearby point not in the set
    # This is expensive, so do it only if time remains.
    if time.time() < deadline - 10:
        current_pts = best_points
        current_score = best_score
        # Identify the bounding box of current points
        xs = [p[0] for p in current_pts]
        ys = [p[1] for p in current_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Generate a pool of candidate replacement points within [-30, 30]
        # near the current grid
        pool = []
        for x in range(max(0, min_x - 5), min(61, max_x + 6)):
            for y in range(max(0, min_y - 5), min(61, max_y + 6)):
                p = (x, y)
                if p not in current_pts:
                    pool.append(p)
        
        # Randomly try some swaps
        random.seed(42)
        steps = 0
        while time.time() < deadline - 5 and steps < 200:
            steps += 1
            # Pick a random point to remove
            idx_remove = random.randint(0, 29)
            p_remove = current_pts[idx_remove]
            
            # Pick a random point to add
            if not pool:
                break
            idx_add = random.randint(0, len(pool) - 1)
            p_add = pool[idx_add]
            
            new_pts = [p for i, p in enumerate(current_pts) if i != idx_remove]
            new_pts.append(p_add)
            
            s = score(new_pts)
            if s > current_score:
                current_score = s
                current_pts = new_pts
                # Update pool: remove p_add, add p_remove
                pool.remove(p_add)
                if p_remove not in pool:
                    pool.append(p_remove)
            # If not improved, we don't change state (Hill Climbing)
            # To allow escaping local minima, we could accept worse moves, 
            # but for this short time, hill climbing is safer.
            
    return current_pts
