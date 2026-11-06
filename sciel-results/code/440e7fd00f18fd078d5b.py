"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from math import gcd

_DEADLINE = 140.0  # safety margin before 150s kill

def _line_key(p1, p2):
    """Return canonical (dx, dy, cx, cy) for the line through p1, p2.
    The line is represented as: dx*x + dy*y = c, with (dx,dy) normalized."""
    x1, y1 = p1
    x2, y2 = p2
    dx = y2 - y1
    dy = -(x2 - x1)
    # Normalize: make gcd positive, and dx > 0 or (dx==0 and dy > 0)
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dx * x1 + dy * y1
    return (dx, dy, c)

def _count_exact4_lines(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    
    count = 0
    seen_lines = set()
    for key, pts_in_line in line_map.items():
        if len(pts_in_line) == 4:
            count += 1
    return count

def _score(points):
    """Score: number of lines with exactly 4 points. Returns -inf if invalid."""
    n = len(points)
    if n != 16:
        return -10**9
    # Check distinctness
    if len(set(tuple(p) for p in points)) != 16:
        return -10**9
    # Check bounds
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -10**9
    # Check no five collinear
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for key, pts_in_line in line_map.items():
        if len(pts_in_line) >= 5:
            return -10**9
    return _count_exact4_lines(points)

def _score_fast(points):
    """Faster scoring that also checks constraints. Returns (score, valid)."""
    n = len(points)
    if n != 16:
        return (-10**9, False)
    if len(set(tuple(p) for p in points)) != 16:
        return (-10**9, False)
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return (-10**9, False)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    exact4 = 0
    for key, pts_in_line in line_map.items():
        if len(pts_in_line) >= 5:
            return (-10**9, False)
        if len(pts_in_line) == 4:
            exact4 += 1
    return (exact4, True)

def _random_point():
    return [random.randint(-1000, 1000), random.randint(-1000, 1000)]

def _grid_seed():
    """4x4 grid as starting point."""
    return [[x, y] for x in range(4) for y in range(4)]

def _mutate(points, strength=1):
    """Perturb points by small integer offsets."""
    new_points = []
    for p in points:
        x = p[0] + random.randint(-strength, strength)
        y = p[1] + random.randint(-strength, strength)
        new_points.append([x, y])
    return new_points

def _hill_climb(start_points, max_steps=2000):
    """Local search: move one point at a time to improve score."""
    current = [list(p) for p in start_points]
    current_score, _ = _score_fast(current)
    if current_score < 0:
        return current, current_score
    
    for step in range(max_steps):
        if time.time() > _DEADLINE:
            break
        # Pick a random point to move
        idx = random.randint(0, 15)
        old = list(current[idx])
        
        # Try several random perturbations
        best_neighbor = None
        best_neighbor_score = current_score
        for _ in range(20):
            new_pt = [old[0] + random.randint(-5, 5), old[1] + random.randint(-5, 5)]
            if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                continue
            trial = [list(p) for p in current]
            trial[idx] = new_pt
            s, valid = _score_fast(trial)
            if valid and s > best_neighbor_score:
                best_neighbor_score = s
                best_neighbor = trial
        
        if best_neighbor is not None:
            current = best_neighbor
            current_score = best_neighbor_score
        else:
            # Try larger moves occasionally
            new_pt = [old[0] + random.randint(-50, 50), old[1] + random.randint(-50, 50)]
            if abs(new_pt[0]) <= 1000 and abs(new_pt[1]) <= 1000:
                trial = [list(p) for p in current]
                trial[idx] = new_pt
                s, valid = _score_fast(trial)
                if valid and s > current_score:
                    current = trial
                    current_score = s
    
    return current, current_score

def _line_completion_move(points):
    """Try to place a point to complete a 4-collinear set."""
    n = len(points)
    # Find all pairs and check if adding a third point on that line would help
    best_move = None
    best_gain = 0
    
    # For each pair of points, find the line, check how many other points are on it
    line_counts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key not in line_counts:
                line_counts[key] = set()
            line_counts[key].add(i)
            line_counts[key].add(j)
    
    # For lines with exactly 3 points, try to find a 4th position
    for key, pts_on_line in line_counts.items():
        if len(pts_on_line) == 3:
            # Find the line equation: dx*x + dy*y = c
            dx, dy, c = key
            # Try to find an integer point on this line not already used
            # The line has points at indices pts_on_line
            p1 = points[list(pts_on_line)[0]]
            # Parametrize: move along the line direction
            # Direction vector is (dy, -dx) normalized... actually the direction is perpendicular to normal (dx,dy)
            # Direction: (dy, -dx) or (-dy, dx)
            dirx, diry = dy, -dx
            g = gcd(abs(dirx), abs(diry))
            if g > 0:
                dirx //= g
                diry //= g
            
            # Try positions near existing points on the line
            for base_pt in points[list(pts_on_line)]:
                for t in range(-3, 4):
                    if t == 0:
                        continue
                    nx = base_pt[0] + t * dirx
                    ny = base_pt[1] + t * diry
                    if abs(nx) > 1000 or abs(ny) > 1000:
                        continue
                    new_pt = [nx, ny]
                    # Check it's not already a point
                    if any(new_pt[0] == p[0] and new_pt[1] == p[1] for p in points):
                        continue
                    # Check this doesn't create 5-collinear
                    # Quick check: count points on the line through new_pt and each existing point
                    trial = list(points) + [new_pt]
                    s, valid = _score_fast(trial)
                    if valid:
                        # Count how many 4-lines this creates
                        if s > best_gain + 0:  # just track best
                            pass
                    # This is expensive; do a lighter check
                    # Just verify no 5-collinear with the new point
                    five_found = False
                    for k in range(n):
                        if k in pts_on_line:
                            continue
                        key2 = _line_key(new_pt, points[k])
                        # Count points on this line
                        cnt = 0
                        for m in range(n):
                            if _line_key(points[m], points[k]) == key2:
                                cnt += 1
                        if cnt >= 3:  # new_pt + k + 3 others = 5
                            five_found = True
                            break
                    if not five_found:
                        best_move = new_pt
                        break
            if best_move is not None:
                break
    
    return best_move

def construct():
    start_time = time.time()
    
    # Start with the 4x4 grid
    best_points = _grid_seed()
    best_score, _ = _score_fast(best_points)
    
    # Also try a few other structured starts
    alt_starts = [
        # Shifted grid
        [[x + 3, y + 3] for x in range(4) for y in range(4)],
        # Non-uniform grid
        [[x * 3, y * 3] for x in range(4) for y in range(4)],
        # Mixed
        [[x * 2 + y, x + y * 2] for x in range(4) for y in range(4)],
    ]
    
    for alt in alt_starts:
        s, valid = _score_fast(alt)
        if valid and s > best_score:
            best_score = s
            best_points = [list(p) for p in alt]
    
    # Hill climbing from best start
    result, result_score = _hill_climb(best_points, max_steps=3000)
    if result_score > best_score:
        best_points = result
        best_score = result_score
    
    # Multiple restarts with random perturbations of the best
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > _DEADLINE:
            break
        
        # Create a perturbed version of the best
        perturbed = _mutate(best_points, strength=random.randint(1, 30))
        
        # Quick validation
        s, valid = _score_fast(perturbed)
        if not valid:
            continue
        
        # Hill climb from here
        result, result_score = _hill_climb(perturbed, max_steps=1500)
        if result_score > best_score:
            best_score = result_score
            best_points = result
    
    # Final line-completion attempts
    if time.time() < _DEADLINE - 10:
        for _ in range(100):
            if time.time() > _DEADLINE:
                break
            move = _line_completion_move(best_points)
            if move is not None:
                # Replace the least useful point with the new one
                # Find which point to remove: the one on the fewest 4-lines
                # Simple heuristic: remove a random point and add the new one
                trial = [list(p) for p in best_points]
                # Try removing each point and adding the new one
                for idx in range(16):
                    if trial[idx][0] == move[0] and trial[idx][1] == move[1]:
                        continue
                    trial2 = [p for i, p in enumerate(trial) if i != idx] + [move]
                    s, valid = _score_fast(trial2)
                    if valid and s > best_score:
                        best_score = s
                        best_points = trial2
                        break
    
    return [list(p) for p in best_points]
# EVOLVE-BLOCK-END
