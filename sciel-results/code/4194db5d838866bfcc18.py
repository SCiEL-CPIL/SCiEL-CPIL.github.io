import itertools
import numpy as np
from collections import Counter

def construct():
    def count_max_multiplicity(points):
        areas = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    det = abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                    if det > 0:
                        areas.append(det)
        if not areas:
            return 0
        return max(Counter(areas).values())
    
    def generate_grid(m, n, sx=1, sy=1, ox=0, oy=0):
        pts = []
        for i in range(m):
            for j in range(n):
                x = ox + i * sx
                y = oy + j * sy
                if -30 <= x <= 30 and -30 <= y <= 30:
                    pts.append((x, y))
        return pts

    candidates = []
    
    # Test all m x n grids where m*n >= 30
    # We can take the first 30 if m*n > 30, but rectangular subsets might be worse.
    # Prefer m*n == 30.
    for m in range(1, 31):
        for n in range(1, 31):
            if m * n == 30:
                candidates.append(generate_grid(m, n))
                candidates.append(generate_grid(n, m))
            elif m * n > 30:
                # Only add if we can form a good subgrid? 
                # For now, skip >30 to avoid complexity, or just add 3x11, 4x8 etc.
                # 4x8 = 32. 5x7=35.
                pass
    
    # Add specific promising ones
    candidates.append(generate_grid(3, 10))
    candidates.append(generate_grid(10, 3))
    candidates.append(generate_grid(2, 15))
    candidates.append(generate_grid(15, 2))
    candidates.append(generate_grid(5, 6))
    candidates.append(generate_grid(6, 5))
    
    best_points = None
    best_score = -1
    
    for pts in candidates:
        if len(pts) != 30:
            continue
        if len(set(pts)) != 30:
            continue
        if any(not (-30 <= x <= 30 and -30 <= y <= 30) for x, y in pts):
            continue
        
        score = count_max_multiplicity(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        
    return best_points
