import time
import random
from itertools import combinations

def _score(points):
    """Compute the maximum multiplicity of any single positive doubled area."""
    n = len(points)
    if n < 3:
        return 0
    # Use a dictionary to count areas
    area_counts = {}
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                # Doubled area = |xi(yj - yk) + xj(yk - yi) + xk(yi - yj)|
                d = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if d > 0:
                    if d in area_counts:
                        area_counts[d] += 1
                    else:
                        area_counts[d] = 1
    if not area_counts:
        return 0
    return max(area_counts.values())

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin
    
    candidates = []
    
    # 1. Two parallel lines of 15 points each
    # y=0: x=0..14
    # y=1: x=0..14
    pts_2line = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    candidates.append(pts_2line)
    
    # 2. Three parallel lines of 10 points each
    # y=0,1,2: x=0..9
    pts_3line = [(x, 0) for x in range(10)] + [(x, 1) for x in range(10)] + [(x, 2) for x in range(10)]
    candidates.append(pts_3line)
    
    # 3. Four parallel lines
    # y=0: 8 pts, y=1: 8 pts, y=2: 7 pts, y=3: 7 pts
    pts_4line = [(x, 0) for x in range(8)] + [(x, 1) for x in range(8)] + [(x, 2) for x in range(7)] + [(x, 3) for x in range(7)]
    candidates.append(pts_4line)
    
    # 4. 6x5 Grid
    pts_grid = [(x, y) for x in range(6) for y in range(5)]
    candidates.append(pts_grid)
    
    # 5. 10x3 Grid
    pts_grid2 = [(x, y) for x in range(10) for y in range(3)]
    candidates.append(pts_grid2)
    
    # Evaluate candidates
    best_points = None
    best_score = 0
    
    for pts in candidates:
        if len(pts) != 30:
            continue
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= x <= 30 and -30 <= y <= 30) for x, y in pts):
            continue
        
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts

    if best_points is None:
        # Fallback
        return [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]

    # Local Search
    # Try to improve best_points
    current_points = [p for p in best_points]
    current_score = best_score
    no_improve_count = 0
    
    while time.time() - t0 < DEADLINE:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(30))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - t0 > DEADLINE:
                break
                
            x, y = current_points[idx]
            # Try small moves
            moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1), (2, 0), (-2, 0), (0, 2), (0, -2)]
            for dx, dy in moves:
                new_x, new_y = x + dx, y + dy
                if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                    continue
                
                new_pt = (new_x, new_y)
                # Check if new_pt is already in current_points (excluding idx)
                if new_pt in current_points[:idx] + current_points[idx+1:]:
                    continue
                
                # Apply move
                old_pt = current_points[idx]
                current_points[idx] = new_pt
                
                s = _score(current_points)
                
                if s > current_score:
                    current_score = s
                    improved = True
                    no_improve_count = 0
                    # Keep the move
                else:
                    # Revert
                    current_points[idx] = old_pt
                    no_improve_count += 1
        
        if not improved:
            break
        if no_improve_count > 50:
            break
            
    return current_points
