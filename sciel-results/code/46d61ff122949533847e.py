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
                # Also flip c to maintain consistency
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
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0
    
    # Initialize with 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    score, five = count_exact_four_lines_fast(grid)
    best_score = score
    best_points = [p[:] for p in grid]
    
    # Try various structured starting points
    random.seed(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Scaled/shifted grids
    for sx in [1, 2, 3, 4]:
        for sy in [1, 2, 3, 4]:
            for ox in [-2, 0, 2]:
                for oy in [-2, 0, 2]:
                    pts = [[ox + sx*x, oy + sy*y] for x in range(4) for y in range(4)]
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
    
    # Simulated annealing / hill climbing
    current = [p[:] for p in best_points]
    current_score = best_score
    current_five = 0
    
    temperature = 50.0
    cooling = 0.995
    
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        
        # Try moving each point
        for idx in range(16):
            if time.time() > deadline:
                break
            
            # Try a set of candidate moves
            moves = []
            # Small moves
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    moves.append((dx, dy))
            # Medium moves
            for _ in range(10):
                moves.append((random.randint(-10, 10), random.randint(-10, 10)))
            # Large moves (occasionally)
            if random.random() < 0.1:
                moves.append((random.randint(-30, 30), random.randint(-30, 30)))
            
            for dx, dy in moves:
                if time.time() > deadline:
                    break
                
                new_pts = [p[:] for p in current]
                new_pts[idx] = [current[idx][0] + dx, current[idx][1] + dy]
                
                if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
                    continue
                if len(set(tuple(p) for p in new_pts)) != 16:
                    continue
                
                s, f = count_exact_four_lines_fast(new_pts)
                
                if f > 0:
                    # Penalize five-or-more collinear
                    delta = s - current_score - 10 * f
                else:
                    delta = s - current_score
                
                # Accept if improvement or with probability (simulated annealing)
                if delta > 0:
                    current = new_pts
                    current_score = s
                    current_five = f
                    improved = True
                    no_improve_count = 0
                    if s > best_score:
                        best_score = s
                        best_points = [p[:] for p in current]
                    break
                else:
                    # Accept with probability exp(delta / temperature)
                    if random.random() < (2.718281828 ** (delta / max(temperature, 0.1))):
                        current = new_pts
                        current_score = s
                        current_five = f
                        improved = True
                        no_improve_count = 0
                        if s > best_score:
                            best_score = s
                            best_points = [p[:] for p in current]
                        break
        
        # Cool down
        temperature *= cooling
        
        # If stuck for too long, restart from best
        if no_improve_count > 200 and time.time() < deadline - 10:
            current = [p[:] for p in best_points]
            current_score = best_score
            temperature = 30.0
            no_improve_count = 0
    
    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16, "Points not distinct"
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final), "Points out of bounds"
    s, f = count_exact_four_lines_fast(final)
    assert f == 0, f"Five or more collinear: {f}"
    
    return final
