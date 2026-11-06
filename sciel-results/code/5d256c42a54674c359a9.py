import itertools
import time

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    def max_area_multiplicity(points):
        n = len(points)
        if n < 3:
            return 0
        counts = {}
        # Precompute differences for speed? Or just direct calculation.
        # N=30 is small, direct is fine.
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    # Doubled area = |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                    area2 = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                    if area2 > 0:
                        counts[area2] = counts.get(area2, 0) + 1
        if not counts:
            return 0
        return max(counts.values())

    def make_grid(w, h, offset_x=0, offset_y=0):
        pts = []
        for x in range(w):
            for y in range(h):
                pts.append((x + offset_x, y + offset_y))
        return pts

    candidates = []
    
    # Standard grids of size 30
    candidates.append(make_grid(6, 5))
    candidates.append(make_grid(5, 6))
    candidates.append(make_grid(10, 3))
    candidates.append(make_grid(3, 10))
    candidates.append(make_grid(15, 2))
    candidates.append(make_grid(2, 15))
    
    # Larger grids pruned to 30
    # 6x6 = 36 points. Remove 6.
    # Try removing corners
    g66 = make_grid(6, 6)
    # Remove 4 corners and 2 center-ish
    corners = [(0,0), (0,5), (5,0), (5,5)]
    to_remove = set(corners)
    # Add 2 more
    to_remove.add((2,2))
    to_remove.add((3,3))
    pruned_66 = [p for p in g66 if p not in to_remove]
    if len(pruned_66) == 30:
        candidates.append(pruned_66)
        
    # Try removing just corners from 6x6? No, need to remove 6.
    # What if we take 6x6 and remove a 1x6 line?
    # e.g. remove x=5
    pruned_66_col = [p for p in g66 if p[0] != 5]
    if len(pruned_66_col) == 30:
        candidates.append(pruned_66_col)
        
    # 5x7 = 35 points. Remove 5.
    g57 = make_grid(5, 7)
    # Remove 5 corners? Only 4. Remove 4 corners + 1 center.
    corners_57 = [(0,0), (0,6), (4,0), (4,6)]
    to_remove_57 = set(corners_57)
    to_remove_57.add((2,3))
    pruned_57 = [p for p in g57 if p not in to_remove_57]
    if len(pruned_57) == 30:
        candidates.append(pruned_57)
        
    # 4x8 = 32 points. Remove 2.
    g48 = make_grid(4, 8)
    to_remove_48 = {(0,0), (3,7)}
    pruned_48 = [p for p in g48 if p not in to_remove_48]
    if len(pruned_48) == 30:
        candidates.append(pruned_48)

    # 7x5 = 35. Remove 5.
    g75 = make_grid(7, 5)
    corners_75 = [(0,0), (0,4), (6,0), (6,4)]
    to_remove_75 = set(corners_75)
    to_remove_75.add((3,2))
    pruned_75 = [p for p in g75 if p not in to_remove_75]
    if len(pruned_75) == 30:
        candidates.append(pruned_75)

    best_score = -1
    best_points = candidates[0]
    
    for pts in candidates:
        # Ensure distinct and in bounds
        if len(set(pts)) != 30:
            continue
        # Check bounds [-30, 30]
        valid = True
        for x, y in pts:
            if not (-30 <= x <= 30 and -30 <= y <= 30):
                valid = False
                break
        if not valid:
            continue
            
        score = max_area_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            
        if time.time() > deadline:
            break
            
    return best_points
