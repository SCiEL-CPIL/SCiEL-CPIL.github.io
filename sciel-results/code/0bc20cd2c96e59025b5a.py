import random
import time
import itertools
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Helper: Check if 4 points are collinear
    def has_four_collinear(points):
        n = len(points)
        line_map = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
                elif dx == 0 and dy < 0:
                    dy = -dy
                
                a = dy
                b = -dx
                c = a * x1 + b * y1
                key = (a, b, c)
                if key not in line_map:
                    line_map[key] = []
                line_map[key].append(i)
                
        for pts in line_map.values():
            if len(pts) >= 4:
                return True
        return False

    def count_collinear_triples(points):
        n = len(points)
        line_map = {}
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                dx //= g
                dy //= g
                if dx < 0:
                    dx = -dx
                    dy = -dy
                elif dx == 0 and dy < 0:
                    dy = -dy
                
                a = dy
                b = -dx
                c = a * x1 + b * y1
                key = (a, b, c)
                if key not in line_map:
                    line_map[key] = []
                line_map[key].append(i)
        
        total_triples = 0
        for pts in line_map.values():
            k = len(pts)
            if k >= 3:
                # Number of triples is C(k, 3)
                total_triples += k * (k - 1) * (k - 2) // 6
        return total_triples

    # Initial random valid configuration
    best_points = None
    best_triple_count = -1
    
    # Generate initial points
    while time.time() < deadline - 5:
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append((x, y))
        
        if not has_four_collinear(pts):
            triple_count = count_collinear_triples(pts)
            if best_points is None or triple_count > best_triple_count:
                best_points = pts
                best_triple_count = triple_count
                
        if best_points is None:
            continue

        # Local Search: Move one point to increase triple count
        current_points = best_points
        current_triples = best_triple_count
        improved = False
        
        # Try to improve current best
        for _ in range(10): # Few steps per outer loop iteration to keep time check frequent
            if time.time() > deadline - 2:
                break
            
            idx = random.randint(0, 19)
            
            # Try moving this point
            for _ in range(10):
                new_x = random.randint(-100, 100)
                new_y = random.randint(-100, 100)
                
                # Check if point already exists (excluding idx)
                # Optimization: check against a set of existing points excluding current idx
                existing = set()
                for k in range(20):
                    if k != idx:
                        existing.add((current_points[k][0], current_points[k][1]))
                
                if (new_x, new_y) in existing:
                    continue
                
                new_points = current_points.copy()
                new_points[idx] = (new_x, new_y)
                
                if has_four_collinear(new_points):
                    continue
                
                new_triples = count_collinear_triples(new_points)
                
                # We want to MAXIMIZE triples to minimize the max independent set
                if new_triples > current_triples:
                    current_points = new_points
                    current_triples = new_triples
                    improved = True
                    break
                elif new_triples == current_triples:
                    # Accept equal with low prob to escape local minima
                    if random.random() < 0.05:
                        current_points = new_points
                        # current_triples stays same
                        break

        if current_triples > best_triple_count:
            best_points = current_points
            best_triple_count = current_triples

    return [[p[0], p[1]] for p in best_points]
