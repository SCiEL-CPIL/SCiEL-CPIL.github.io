"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _count_four_point_lines(points):
    """Count lines containing exactly 4 of the points."""
    n = len(points)
    count = 0
    # Use a set of normalized line keys to avoid double counting
    seen_lines = set()
    for i, j, k, l in combinations(range(n), 4):
        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
        # Check if all four are collinear
        # Cross product of (p2-p1) and (p3-p1) must be 0, and (p2-p1) and (p4-p1) must be 0
        dx12 = p2[0] - p1[0]
        dy12 = p2[1] - p1[1]
        dx13 = p3[0] - p1[0]
        dy13 = p3[1] - p1[1]
        dx14 = p4[0] - p1[0]
        dy14 = p4[1] - p1[1]
        if dx12 * dy13 - dy12 * dx13 == 0 and dx12 * dy14 - dy12 * dx14 == 0:
            # Normalize the line key: use direction vector and a point
            # Direction: gcd-normalized
            import math
            g1 = math.gcd(dx12, dy12)
            if g1 == 0:
                g1 = 1
            dxn = dx12 // g1
            dyn = dy12 // g1
            # Ensure canonical direction
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn = -dxn
                dyn = -dyn
            # Line key: (dxn, dyn, p1) but we need to be careful about which point
            # Use the line equation ax + by = c where a=dy, b=-dx, normalized
            a = dyn
            b = -dxn
            g = math.gcd(abs(a), abs(b))
            if g > 0:
                a //= g
                b //= g
            c = a * p1[0] + b * p1[1]
            line_key = (a, b, c)
            if line_key not in seen_lines:
                seen_lines.add(line_key)
                # Verify no 5th point is on this line
                has_five = False
                for m in range(n):
                    if m in (i, j, k, l):
                        continue
                    pm = points[m]
                    if a * pm[0] + b * pm[1] == c:
                        has_five = True
                        break
                if not has_five:
                    count += 1
    return count

def _is_valid(points):
    """Check that no five points are collinear."""
    n = len(points)
    for i, j, k, l, m in combinations(range(n), 5):
        p1, p2, p3, p4, p5 = points[i], points[j], points[k], points[l], points[m]
        dx12 = p2[0] - p1[0]
        dy12 = p2[1] - p1[1]
        dx13 = p3[0] - p1[0]
        dy13 = p3[1] - p1[1]
        dx14 = p4[0] - p1[0]
        dy14 = p4[1] - p1[1]
        dx15 = p5[0] - p1[0]
        dy15 = p5[1] - p1[1]
        if (dx12 * dy13 - dy12 * dx13 == 0 and 
            dx12 * dy14 - dy12 * dx14 == 0 and
            dx12 * dy15 - dy12 * dx15 == 0):
            return False
    return True

def _local_search(points, max_iter=5000, time_limit=10):
    """Local search to improve the number of 4-point lines."""
    start_time = time.time()
    best_points = [p[:] for p in points]
    best_score = _count_four_point_lines(best_points)
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        # Try moving a random point
        i = __import__('random').randint(0, 15)
        old_x, old_y = best_points[i]
        new_x = __import__('random').randint(-1000, 1000)
        new_y = __import__('random').randint(-1000, 1000)
        best_points[i] = [new_x, new_y]
        
        # Check validity
        valid = True
        # Check no five collinear
        n = 16
        for combo in combinations(range(n), 5):
            if i in combo:
                p1 = best_points[combo[0]]
                p2 = best_points[combo[1]]
                p3 = best_points[combo[2]]
                p4 = best_points[combo[3]]
                p5 = best_points[combo[4]]
                dx12 = p2[0] - p1[0]
                dy12 = p2[1] - p1[1]
                dx13 = p3[0] - p1[0]
                dy13 = p3[1] - p1[1]
                dx14 = p4[0] - p1[0]
                dy14 = p4[1] - p1[1]
                dx15 = p5[0] - p1[0]
                dy15 = p5[1] - p1[1]
                if (dx12 * dy13 - dy12 * dx13 == 0 and 
                    dx12 * dy14 - dy12 * dx14 == 0 and
                    dx12 * dy15 - dy12 * dx15 == 0):
                    valid = False
                    break
        if not valid:
            best_points[i] = [old_x, old_y]
            continue
        
        # Check distinctness
        distinct = True
        for j in range(16):
            if j != i and best_points[j] == best_points[i]:
                distinct = False
                break
        if not distinct:
            best_points[i] = [old_x, old_y]
            continue
        
        score = _count_four_point_lines(best_points)
        if score >= best_score:
            best_score = score
        else:
            best_points[i] = [old_x, old_y]
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with 4x4 grid
    points = [[x, y] for x in range(4) for y in range(4)]
    best_points = [p[:] for p in points]
    best_score = _count_four_point_lines(best_points)
    
    # Local search from multiple initial configurations
    import random
    
    initial_configs = [
        # 4x4 grid
        [[x, y] for x in range(4) for y in range(4)],
        # 5x5 grid subset
        [[x, y] for x in range(5) for y in range(4)][:16],
        # 6x3 grid
        [[x, y] for x in range(6) for y in range(3)][:16],
        # Random points
        [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)],
        [[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)],
    ]
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        # Ensure distinct points
        seen = set()
        valid_config = []
        for p in config:
            if (p[0], p[1]) not in seen:
                seen.add((p[0], p[1]))
                valid_config.append(p)
        while len(valid_config) < 16:
            valid_config.append([random.randint(-1000, 1000), random.randint(-1000, 1000)])
        
        result_points, result_score = _local_search(valid_config, max_iter=3000, time_limit=20)
        if result_score > best_score:
            best_score = result_score
            best_points = [p[:] for p in result_points]
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 5:
        result_points, result_score = _local_search(best_points, max_iter=10000, time_limit=30)
        if result_score > best_score:
            best_score = result_score
            best_points = [p[:] for p in result_points]
    
    return best_points
# EVOLVE-BLOCK-END
