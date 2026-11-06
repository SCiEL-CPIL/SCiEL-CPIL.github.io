import time
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 80.0  # 90s limit, 10s safety margin
    
    def score_points(points):
        """Compute max multiplicity of positive doubled area."""
        areas = []
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if det > 0:
                        areas.append(det)
        if not areas:
            return 0
        return max(Counter(areas).values())
    
    # Initial: 3 parallel lines, 10 points each
    # Lines at y=0,1,2; x=0..9
    points = []
    for y in range(3):
        for x in range(10):
            points.append((x, y))
    
    best_score = score_points(points)
    best_points = points[:]
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_points
    
    # Try to improve by local search: move one point
    import random
    random.seed(42)
    
    # Precompute for speed
    def score_points_fast(points):
        areas = []
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i+1, n):
                x2, y2 = points[j]
                for k in range(j+1, n):
                    x3, y3 = points[k]
                    det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                    if det > 0:
                        areas.append(det)
        if not areas:
            return 0
        c = Counter(areas)
        return max(c.values())
    
    # Try different line configurations
    for num_lines in [2, 3, 4, 5, 6]:
        if time.time() - start_time > deadline:
            break
        pts_per_line = 30 // num_lines
        remainder = 30 % num_lines
        
        # Try different line spacings
        for spacing in range(1, 6):
            if time.time() - start_time > deadline:
                break
            for x_start in range(-30, 31 - pts_per_line):
                if time.time() - start_time > deadline:
                    break
                for x_range in range(pts_per_line, pts_per_line + 15):
                    if time.time() - start_time > deadline:
                        break
                    if x_start + x_range > 31:
                        break
                    
                    points = []
                    for li in range(num_lines):
                        y = li * spacing
                        if y > 30:
                            break
                        n_pts = pts_per_line + (1 if li < remainder else 0)
                        for xi in range(n_pts):
                            x = x_start + xi
                            if x < -30 or x > 30:
                                break
                            points.append((x, y))
                    
                    if len(points) != 30:
                        continue
                    # Check distinct
                    if len(set(points)) != 30:
                        continue
                    
                    s = score_points_fast(points)
                    if s > best_score:
                        best_score = s
                        best_points = points[:]
    
    return best_points
