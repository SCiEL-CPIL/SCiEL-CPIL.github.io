"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _count_4_lines(points):
    """Count lines with exactly 4 points, return (count_4, has_5)."""
    n = len(points)
    # Map each line (defined by pair) to the set of points on it
    line_to_pts = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Normalize direction
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            dx_n = dx // g
            dy_n = dy // g
            # Canonical key: use the normalized direction and a reference point
            # To make it canonical, use the lexicographically smallest point as reference
            key = (dx_n, dy_n)
            line_to_pts[key].add(i)
            line_to_pts[key].add(j)
    
    # Now for each line direction, find all points on lines with that direction
    # Actually, we need to group by actual line, not just direction.
    # Let me redo this properly.
    
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            dx_n = dx // g
            dy_n = dy // g
            # Canonical line key: (direction, offset)
            # offset = cross product of direction with point
            # For direction (dx_n, dy_n), the line is dx_n*y - dy_n*x = c
            c = dx_n * y1 - dy_n * x1
            key = (dx_n, dy_n, c)
            line_points[key].add(i)
            line_points[key].add(j)
    
    count_4 = 0
    has_5 = False
    for key, pts in line_points.items():
        if len(pts) >= 5:
            has_5 = True
        if len(pts) == 4:
            count_4 += 1
    
    return count_4, has_5

def _count_4_lines_fast(points):
    """Faster counting using tuple-based line keys."""
    n = len(points)
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            dx_n = dx // g
            dy_n = dy // g
            c = dx_n * y1 - dy_n * x1
            key = (dx_n, dy_n, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    count_4 = 0
    has_5 = False
    for key, pts in line_points.items():
        sz = len(pts)
        if sz >= 5:
            has_5 = True
        elif sz == 4:
            count_4 += 1
    
    return count_4, has_5

def _get_3_point_lines(points):
    """Return list of (line_key, point_set) for lines with exactly 3 points."""
    n = len(points)
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i+1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            dx_n = dx // g
            dy_n = dy // g
            c = dx_n * y1 - dy_n * x1
            key = (dx_n, dy_n, c)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    three_lines = []
    for key, pts in line_points.items():
        if len(pts) == 3:
            three_lines.append((key, pts))
    return three_lines

def _score(points):
    """Return score: number of 4-point lines if valid, else -infinity."""
    count_4, has_5 = _count_4_lines_fast(points)
    if has_5:
        return -10**9
    return count_4

def _generate_initial_config(seed):
    """Generate a diverse initial configuration."""
    rng = random.Random(seed)
    configs = []
    
    # 4x4 grid
    configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # 4x4 grid shifted
    configs.append([[x+1, y+1] for x in range(4) for y in range(4)])
    
    # 5x4 grid subset (remove one column)
    base = [[x, y] for x in range(5) for y in range(4)]
    # Remove random points to get 16
    idx = list(range(20))
    rng.shuffle(idx)
    configs.append([base[i] for i in idx[:16]])
    
    # 4x5 grid subset
    base = [[x, y] for x in range(4) for y in range(5)]
    idx = list(range(20))
    rng.shuffle(idx)
    configs.append([base[i] for i in idx[:16]])
    
    # Random points in a box
    pts = []
    while len(pts) < 16:
        p = [rng.randint(-20, 20), rng.randint(-20, 20)]
        if p not in pts:
            pts.append(p)
    configs.append(pts)
    
    # Structured: two rows of 8
    configs.append([[x, 0] for x in range(8)] + [[x, 3] for x in range(8)])
    
    # Structured: 4 columns of 4 with offsets
    pts = []
    for c in range(4):
        offset = c * 2
        for r in range(4):
            pts.append([c + (r % 2), offset + r])
    configs.append(pts)
    
    # Near-grid with one perturbation
    base = [[x, y] for x in range(4) for y in range(4)]
    p = base[:]
    i = rng.randint(0, 15)
    p[i] = [p[i][0] + rng.randint(-3, 3), p[i][1] + rng.randint(-3, 3)]
    configs.append(p)
    
    # Another near-grid
    base = [[x, y] for x in range(4) for y in range(4)]
    p = base[:]
    i = rng.randint(0, 15)
    j = rng.randint(0, 15)
    if i != j:
        p[i] = [p[i][0] + rng.randint(-2, 2), p[i][1] + rng.randint(-2, 2)]
        p[j] = [p[j][0] + rng.randint(-2, 2), p[j][1] + rng.randint(-2, 2)]
    configs.append(p)
    
    return configs

def _local_search(points, seed, time_limit=2.0):
    """Simulated annealing local search with line-completion heuristic."""
    rng = random.Random(seed)
    start_time = time.time()
    
    current = [p[:] for p in points]
    current_score = _score(current)
    
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.25:
            # Shift one point by small amount
            i = rng.randint(0, 15)
            dx = rng.randint(-4, 4)
            dy = rng.randint(-4, 4)
            if dx == 0 and dy == 0:
                continue
            new_pts = [p[:] for p in current]
            new_pts[i][0] += dx
            new_pts[i][1] += dy
            if abs(new_pts[i][0]) > 1000 or abs(new_pts[i][1]) > 1000:
                continue
            if len(set(tuple(p) for p in new_pts)) != 16:
                continue
            
        elif move_type < 0.45:
            # Move one point to a new location (swap-like)
            i, j = rng.sample(range(16), 2)
            dx = rng.randint(-4, 4)
            dy = rng.randint(-4, 4)
            if dx == 0 and dy == 0:
                continue
            new_pts = [p[:] for p in current]
            new_pts[i] = [current[j][0] + dx, current[j][1] + dy]
            new_pts[j] = [current[i][0], current[i][1]]
            if abs(new_pts[i][0]) > 1000 or abs(new_pts[i][1]) > 1000:
                continue
            if len(set(tuple(p) for p in new_pts)) != 16:
                continue
                
        elif move_type < 0.65:
            # Move two points
            i, j = rng.sample(range(16), 2)
            dx1 = rng.randint(-4, 4)
            dy1 = rng.randint(-4, 4)
            dx2 = rng.randint(-4, 4)
            dy2 = rng.randint(-4, 4)
            if (dx1 == 0 and dy1 == 0) or (dx2 == 0 and dy2 == 0):
                continue
            new_pts = [p[:] for p in current]
            new_pts[i][0] += dx1
            new_pts[i][1] += dy1
            new_pts[j][0] += dx2
            new_pts[j][1] += dy2
            if abs(new_pts[i][0]) > 1000 or abs(new_pts[i][1]) > 1000:
                continue
            if abs(new_pts[j][0]) > 1000 or abs(new_pts[j][1]) > 1000:
                continue
            if len(set(tuple(p) for p in new_pts)) != 16:
                continue
        elif move_type < 0.85:
            # Line-completion heuristic: try to complete a 3-point line to 4
            three_lines = _get_3_point_lines(current)
            if not three_lines:
                # Fallback to random move
                i = rng.randint(0, 15)
                dx = rng.randint(-5, 5)
                dy = rng.randint(-5, 5)
                if dx == 0 and dy == 0:
                    continue
                new_pts = [p[:] for p in current]
                new_pts[i][0] += dx
                new_pts[i][1] += dy
                if abs(new_pts[i][0]) > 1000 or abs(new_pts[i][1]) > 1000:
                    continue
                if len(set(tuple(p) for p in new_pts)) != 16:
                    continue
            else:
                key, pts3 = rng.choice(three_lines)
                # Get the three points on this line
                pts3_list = [current[p] for p in pts3]
                # Compute the line direction
                dx = pts3_list[1][0] - pts3_list[0][0]
                dy = pts3_list[1][1] - pts3_list[0][1]
                g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
                dx_n = dx // g
                dy_n = dy // g
                # Try moving a non-line point onto this line
                non_line_pts = [k for k in range(16) if k not in pts3]
                if non_line_pts:
                    i = rng.choice(non_line_pts)
                    # Try to find a position on the line that's valid
                    # Parametric: point = p0 + t*(dx_n, dy_n)
                    p0 = pts3_list[0]
                    # Try several t values
                    found = False
                    for t in range(-10, 11):
                        if t == 0:
                            continue
                        nx = p0[0] + t * dx_n
                        ny = p0[1] + t * dy_n
                        if abs(nx) > 1000 or abs(ny) > 1000:
                            continue
                        # Check if this position is already occupied
                        new_pos = [nx, ny]
                        if any(p == new_pos for p in current if p not in pts3_list):
                            continue
                        # Check it's not one of the existing 3 points
                        if new_pos in pts3_list:
                            continue
                        new_pts = [p[:] for p in current]
                        new_pts[i] = new_pos
                        if len(set(tuple(p) for p in new_pts)) != 16:
                            continue
                        # Check no 5-point lines created
                        s, has_5 = _count_4_lines_fast(new_pts)
                        if has_5:
                            continue
                        # Check it actually creates a 4-point line
                        s2, _ = _count_4_lines_fast(new_pts)
                        if s2 >= current_score:
                            break
                    else:
                        # No good position found, do a small random move
                        i = rng.randint(0, 15)
                        dx = rng.randint(-3, 3)
                        dy = rng.randint(-3, 3)
                        if dx == 0 and dy == 0:
                            continue
                        new_pts = [p[:] for p in current]
                        new_pts[i][0] += dx
                        new_pts[i][1] += dy
                        if abs(new_pts[i][0]) > 1000 or abs(new_pts[i][1]) > 1000:
                            continue
                        if len(set(tuple(p) for p in new_pts)) != 16:
                            continue
                else:
                    continue
        else:
            # Larger jump: move one point significantly
            i = rng.randint(0, 15)
            new_pts = [p[:] for p in current]
            new_pts[i] = [rng.randint(-40, 40), rng.randint(-40, 40)]
            if abs(new_pts[i][0]) > 1000 or abs(new_pts[i][1]) > 1000:
                continue
            if len(set(tuple(p) for p in new_pts)) != 16:
                continue
        
        new_score = _score(new_pts)
        
        # Acceptance criterion
        if new_score >= current_score:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = [p[:] for p in current]
                best_score = current_score
        else:
            delta = new_score - current_score
            if delta > 0:
                current = new_pts
                current_score = new_score
                if current_score > best_score:
                    best = [p[:] for p in current]
                    best_score = current_score
            else:
                # delta is negative
                if rng.random() < math.exp(delta / T):
                    current = new_pts
                    current_score = new_score
        
        T *= cooling
    
    return best, best_score

def _refine(points, time_limit=1.0):
    """Final refinement with smaller steps and line-completion."""
    rng = random.Random(12345)
    start_time = time.time()
    
    current = [p[:] for p in points]
    current_score = _score(current)
    best = [p[:] for p in current]
    best_score = current_score
    
    for _ in range(3000):
        if time.time() - start_time > time_limit:
            break
        
        # Mix of small moves and line-completion attempts
        if rng.random() < 0.7:
            i = rng.randint(0, 15)
            dx = rng.randint(-3, 3)
            dy = rng.randint(-3, 3)
            if dx == 0 and dy == 0:
                continue
            new_pts = [p[:] for p in current]
            new_pts[i][0] += dx
            new_pts[i][1] += dy
            if abs(new_pts[i][0]) > 1000 or abs(new_pts[i][1]) > 1000:
                continue
            if len(set(tuple(p) for p in new_pts)) != 16:
                continue
        else:
            # Try to complete a 3-point line
            three_lines = _get_3_point_lines(current)
            if not three_lines:
                continue
            key, pts3 = rng.choice(three_lines)
            pts3_list = [current[p] for p in pts3]
            dx = pts3_list[1][0] - pts3_list[0][0]
            dy = pts3_list[1][1] - pts3_list[0][1]
            g = math.gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
            dx_n = dx // g
            dy_n = dy // g
            non_line_pts = [k for k in range(16) if k not in pts3]
            if not non_line_pts:
                continue
            i = rng.choice(non_line_pts)
            p0 = pts3_list[0]
            found = False
            for t in range(-12, 13):
                if t == 0:
                    continue
                nx = p0[0] + t * dx_n
                ny = p0[1] + t * dy_n
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                new_pos = [nx, ny]
                if any(p == new_pos for p in current if p not in pts3_list):
                    continue
                if new_pos in pts3_list:
                    continue
                new_pts = [p[:] for p in current]
                new_pts[i] = new_pos
                if len(set(tuple(p) for p in new_pts)) != 16:
                    continue
                s, has_5 = _count_4_lines_fast(new_pts)
                if has_5:
                    continue
                if s > current_score:
                    break
            else:
                continue
        
        new_score = _score(new_pts)
        if new_score > current_score:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = [p[:] for p in current]
                best_score = current_score
    
    return best, best_score

def construct():
    deadline = time.time() + 140  # Safety margin from 150s
    
    best_points = None
    best_score = -1
    
    # Try multiple initial configurations with local search
    initial_configs = _generate_initial_config(42)
    
    for idx, config in enumerate(initial_configs):
        if time.time() > deadline - 25:
            break
        
        # Validate initial config
        s, has_5 = _count_4_lines_fast(config)
        if has_5:
            continue
        
        # Run local search with adaptive time
        remaining = deadline - time.time()
        time_alloc = min(8.0, remaining * 0.5)
        result, score = _local_search(config, seed=1000 + idx, time_limit=time_alloc)
        
        if score > best_score:
            best_score = score
            best_points = result
        
        # Refine
        if time.time() > deadline - 12:
            break
        refined, refined_score = _refine(result, time_limit=2.5)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined
    
    # Additional random restarts
    if best_points is None:
        # Fallback: 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = _score(best_points)
    
    # Try more diverse random restarts
    for trial in range(25):
        if time.time() > deadline - 18:
            break
        
        rng = random.Random(10000 + trial)
        pts = []
        while len(pts) < 16:
            p = [rng.randint(-20, 20), rng.randint(-20, 20)]
            if p not in pts:
                pts.append(p)
        
        s, has_5 = _count_4_lines_fast(pts)
        if has_5:
            continue
        
        remaining = deadline - time.time()
        time_alloc = min(4.0, remaining * 0.4)
        result, score = _local_search(pts, seed=20000 + trial, time_limit=time_alloc)
        if score > best_score:
            best_score = score
            best_points = result
        
        if time.time() > deadline - 8:
            break
        refined, refined_score = _refine(result, time_limit=1.5)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined
    
    # Final polish
    if time.time() < deadline - 4 and best_points is not None:
        final, final_score = _refine(best_points, time_limit=4.0)
        if final_score > best_score:
            best_points = final
            best_score = final_score
    
    # Ensure validity
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    # Final validation
    s, has_5 = _count_4_lines_fast(best_points)
    if has_5:
        # Fallback to 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points
# EVOLVE-BLOCK-END
