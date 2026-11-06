import time
import random
from math import gcd
from itertools import combinations

def count_exact_four_lines_fast(points):
    """Count lines containing exactly 4 of the 16 points. Returns (four_count, five_or_more_count)."""
    n = len(points)
    if n < 4:
        return 0, 0
    
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    four_count = 0
    five_or_more = 0
    for pair_count in line_counts.values():
        if pair_count == 6:  # C(4,2) = 6
            four_count += 1
        elif pair_count >= 10:  # C(5,2) = 10
            five_or_more += 1
    
    return four_count, five_or_more

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: 10s buffer before 120s kill
    
    best_points = None
    best_score = 0
    
    random.seed(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Sheared grids
    for shear in range(-5, 6):
        for sc in [10, 20, 30]:
            pts = []
            for i in range(4):
                for j in range(4):
                    x = sc * i + shear * j
                    y = sc * j
                    pts.append([x, y])
            if all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in pts):
                if len(set(tuple(p) for p in pts)) == 16:
                    initial_configs.append(pts)

    # Random configurations
    for _ in range(50):
        pts = [[random.randint(-50, 50), random.randint(-50, 50)] for _ in range(16)]
        if len(set(tuple(p) for p in pts)) == 16:
            initial_configs.append(pts)
    
    # Evaluate initial configs
    for pts in initial_configs:
        if time.time() > deadline:
            break
        s, f = count_exact_four_lines_fast(pts)
        if f == 0 and s > best_score:
            best_score = s
            best_points = [p[:] for p in pts]
    
    if best_points is None:
        # Fallback: return a valid grid if no good config found
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = 0

    current = [p[:] for p in best_points]
    current_score = best_score
    
    temperature = 10.0
    cooling = 0.999
    no_improve_count = 0
    
    # Pre-define a small set of moves to avoid generating them every time
    small_moves = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]
    
    while time.time() < deadline:
        no_improve_count += 1
        
        # Pick a random point to move
        idx = random.randint(0, 15)
        
        # Pick a move
        # 80% small moves, 20% medium random moves
        if random.random() < 0.8:
            dx, dy = random.choice(small_moves)
        else:
            dx = random.randint(-10, 10)
            dy = random.randint(-10, 10)
        
        new_pts = [p[:] for p in current]
        new_pts[idx] = [current[idx][0] + dx, current[idx][1] + dy]
        
        # Validation
        if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
            continue
        if len(set(tuple(p) for p in new_pts)) != 16:
            continue
        
        s, f = count_exact_four_lines_fast(new_pts)
        
        if f > 0:
            # Penalize five-or-more heavily
            delta = s - current_score - 100 * f
        else:
            delta = s - current_score
        
        # Acceptance
        if delta > 0 or random.random() < (2.718281828 ** (delta / max(temperature, 0.1))):
            current = new_pts
            current_score = s
            if s > best_score:
                best_score = s
                best_points = [p[:] for p in current]
                no_improve_count = 0
                temperature = 10.0 # Reset temperature on improvement
            else:
                temperature *= cooling
        
        if no_improve_count > 500:
            # Restart from best
            current = [p[:] for p in best_points]
            current_score = best_score
            temperature = 10.0
            no_improve_count = 0
            # Perturb current slightly
            idx = random.randint(0, 15)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            current[idx] = [current[idx][0] + dx, current[idx][1] + dy]
            # Ensure validity after perturbation
            if len(set(tuple(p) for p in current)) != 16 or not all(abs(p[0])<=1000 and abs(p[1])<=1000 for p in current):
                current = [p[:] for p in best_points]
                current_score = best_score

    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16, "Points not distinct"
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final), "Points out of bounds"
    s, f = count_exact_four_lines_fast(final)
    assert f == 0, f"Five or more collinear: {f}"
    
    return final
