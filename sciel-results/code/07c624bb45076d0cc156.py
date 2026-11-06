"""Bootstrap spawn: MODEL-AND-OPTIMIZE approach for equal-area triangles from 30 points."""
import time
import random
from collections import Counter
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin under 90s

    def evaluate(points):
        """Compute the maximum multiplicity of any positive doubled area."""
        counts = Counter()
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx1 = x2 - x1
                dy1 = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    dx2 = x3 - x1
                    dy2 = y3 - y1
                    det = abs(dx1 * dy2 - dy1 * dx2)
                    if det > 0:
                        counts[det] += 1
        if not counts:
            return 0
        return max(counts.values())

    def try_config(points, current_best_score):
        """Try a configuration and return updated best if improved."""
        if len(set(points)) != 30:
            return None, current_best_score
        if any(not (-30 <= px <= 30 and -30 <= py <= 30) for px, py in points):
            return None, current_best_score
        score = evaluate(points)
        if score > current_best_score:
            return points, score
        return None, current_best_score

    # Start with a known good configuration
    best_points = [(x, 0) for x in range(15)] + [(x, 1) for x in range(15)]
    best_score = evaluate(best_points)

    # Lattice section approach: select 30 points from a dense grid
    # and use combinatorial optimization to maximize equal-area triples
    # For a regular grid, equal-area triangles are abundant
    
    # Try various grid configurations and select subsets
    factors_30 = [(1,30), (2,15), (3,10), (5,6), (6,5), (10,3), (15,2), (30,1)]
    
    # Systematic grid search
    for w, h in factors_30:
        for sx in range(1, min(31//w + 1, 31)):
            for sy in range(1, min(31//h + 1, 31)):
                if time.time() > deadline:
                    return best_points
                
                pts = [(i * sx, j * sy) for i in range(w) for j in range(h)]
                
                max_x = (w - 1) * sx
                max_y = (h - 1) * sy
                if max_x > 30 or max_y > 30:
                    continue
                    
                result, new_score = try_config(pts, best_score)
                if result is not None:
                    best_points = result
                    best_score = new_score

    # Centered grids
    for w, h in factors_30:
        for sx in range(1, 6):
            for sy in range(1, 6):
                if time.time() > deadline:
                    return best_points
                    
                width = (w - 1) * sx
                height = (h - 1) * sy
                
                if width > 60 or height > 60:
                    continue
                    
                cx = -width // 2
                cy = -height // 2
                
                pts = [(cx + i * sx, cy + j * sy) for i in range(w) for j in range(h)]
                
                result, new_score = try_config(pts, best_score)
                if result is not None:
                    best_points = result
                    best_score = new_score

    # Sheared grids
    for w, h in [(6,5), (5,6), (10,3), (3,10)]:
        for sx in range(1, 5):
            for sy in range(1, 5):
                for sxy in range(-2, 3):
                    if time.time() > deadline:
                        return best_points
                        
                    pts = [(i * sx + j * sxy, j * sy) for i in range(w) for j in range(h)]
                    
                    result, new_score = try_config(pts, best_score)
                    if result is not None:
                        best_points = result
                        best_score = new_score

    # Local search: hill-climbing with larger neighborhood
    random.seed(42)
    
    improved = True
    max_iterations = 50
    iteration = 0
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        for idx in range(30):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[idx]
            # Try moves in a larger neighborhood
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = old_x + dx
                    new_y = old_y + dy
                    if not (-30 <= new_x <= 30 and -30 <= new_y <= 30):
                        continue
                    new_points = list(best_points)
                    new_points[idx] = (new_x, new_y)
                    if len(set(new_points)) != 30:
                        continue
                    score = evaluate(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Random restarts with perturbation
    if time.time() < deadline - 5:
        random.seed(123)
        for restart in range(100):
            if time.time() > deadline - 2:
                break
            
            if random.random() < 0.6:
                # Perturbation of best
                new_points = list(best_points)
                for _ in range(random.randint(1, 4)):
                    idx = random.randint(0, 29)
                    dx = random.randint(-4, 4)
                    dy = random.randint(-4, 4)
                    new_x = new_points[idx][0] + dx
                    new_y = new_points[idx][1] + dy
                    if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                        new_points[idx] = (new_x, new_y)
            else:
                # Grid-like structure with noise
                w, h = random.choice([(6,5), (5,6), (10,3), (3,10), (15,2), (2,15)])
                sx = random.randint(1, max(1, 30 // w))
                sy = random.randint(1, max(1, 30 // h))
                cx = random.randint(-5, 5)
                cy = random.randint(-5, 5)
                new_points = [(cx + i*sx + random.randint(-1,1), cy + j*sy + random.randint(-1,1)) 
                              for i in range(w) for j in range(h)]
            
            result, new_score = try_config(new_points, best_score)
            if result is not None:
                best_points = result
                best_score = new_score

    return best_points
# EVOLVE-BLOCK-END
