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

def find_three_point_lines(points):
    """Find all lines with exactly 3 points. Returns list of (line_key, [indices])."""
    n = len(points)
    line_to_indices = {}
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
            if key not in line_to_indices:
                line_to_indices[key] = set()
            line_to_indices[key].add(i)
            line_to_indices[key].add(j)
    
    three_lines = []
    for key, indices in line_to_indices.items():
        if len(indices) == 3:
            three_lines.append((key, list(indices)))
    return three_lines

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
    
    # Main optimization loop with line-completion heuristic
    current = [p[:] for p in best_points]
    current_score = best_score
    current_five = 0
    
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        if restart > 0:
            # Restart from best with perturbation
            current = [p[:] for p in best_points]
            for _ in range(3):
                idx = random.randint(0, 15)
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                new_x = current[idx][0] + dx
                new_y = current[idx][1] + dy
                if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                    new_pts = [p[:] for p in current]
                    new_pts[idx] = [new_x, new_y]
                    if len(set(tuple(p) for p in new_pts)) == 16:
                        s, f = count_exact_four_lines_fast(new_pts)
                        if f == 0:
                            current = new_pts
                            current_score = s
            current_score, current_five = count_exact_four_lines_fast(current)
        
        temperature = 30.0
        cooling = 0.995
        no_improve_count = 0
        max_no_improve = 150
        
        while no_improve_count < max_no_improve and time.time() < deadline:
            improved = False
            
            # Line-completion heuristic: find 3-point lines and try to complete them
            if random.random() < 0.7:
                three_lines = find_three_point_lines(current)
                if three_lines:
                    random.shuffle(three_lines)
                    for line_key, indices in three_lines[:5]:  # Try up to 5
                        if time.time() > deadline:
                            break
                        
                        dx, dy, c = line_key
                        # Find candidate points on this line
                        # Line: dx*x + dy*y = -c (normalized)
                        # Try points near the existing 3 points
                        existing_pts = [current[i] for i in indices]
                        min_x = min(p[0] for p in existing_pts)
                        max_x = max(p[0] for p in existing_pts)
                        min_y = min(p[1] for p in existing_pts)
                        max_y = max(p[1] for p in existing_pts)
                        
                        candidates = []
                        # Try extending the line beyond the existing points
                        for t in range(-3, 4):
                            if t == 0:
                                continue
                            # Parametric: p = p0 + t*(dx, dy) normalized
                            for base_idx in indices:
                                px, py = current[base_idx]
                                # Move by t steps in direction (dx, dy)
                                # But dx,dy are normalized, so we need actual step
                                # Find a point on the line that's integer
                                # Try nearby integer points on the line
                                pass
                        
                        # Simpler: try moving each of the 3 points slightly, or adding a new point
                        # Actually, we need to move one point TO be on a line with 3 others
                        # Better approach: for each point NOT on this line, check if moving it
                        # to complete a 4-line helps
                        for idx in range(16):
                            if idx in indices:
                                continue
                            # Try moving point idx to various positions on lines with 3 other points
                            # For now, just do random small moves
                            for dx2 in range(-2, 3):
                                for dy2 in range(-2, 3):
                                    if dx2 == 0 and dy2 == 0:
                                        continue
                                    new_pts = [p[:] for p in current]
                                    new_pts[idx] = [current[idx][0] + dx2, current[idx][1] + dy2]
                                    if abs(new_pts[idx][0]) > 1000 or abs(new_pts[idx][1]) > 1000:
                                        continue
                                    if len(set(tuple(p) for p in new_pts)) != 16:
                                        continue
                                    
                                    s, f = count_exact_four_lines_fast(new_pts)
                                    
                                    if f > 0:
                                        delta = s - current_score - 10 * f
                                    else:
                                        delta = s - current_score
                                    
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
                            if improved:
                                break
                        if improved:
                            break
            
            # Standard hill climbing moves
            if not improved:
                for idx in range(16):
                    if time.time() > deadline:
                        break
                    
                    moves = []
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            if dx == 0 and dy == 0:
                                continue
                            moves.append((dx, dy))
                    for _ in range(10):
                        moves.append((random.randint(-10, 10), random.randint(-10, 10)))
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
                            delta = s - current_score - 10 * f
                        else:
                            delta = s - current_score
                        
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
                    
                    if improved:
                        break
            
            if not improved:
                no_improve_count += 1
            
            temperature *= cooling
        
        if time.time() > deadline - 5:
            break
    
    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16, "Points not distinct"
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final), "Points out of bounds"
    s, f = count_exact_four_lines_fast(final)
    assert f == 0, f"Five or more collinear: {f}"
    
    return final
