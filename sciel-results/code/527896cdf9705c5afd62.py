import time
import random
from math import gcd
from collections import Counter

def _line_key(x1, y1, x2, y2):
    """Return a canonical key for the line through (x1,y1) and (x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g > 0:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _fast_score(points):
    """
    Return (exact4_count, has_five) for a set of 16 points.
    """
    n = len(points)
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g > 0:
                dx //= g
                dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            c = dy * x1 - dx * y1
            key = (dx, dy, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    exact4 = 0
    has_five = False
    for key, pc in line_counts.items():
        if pc == 6:
            exact4 += 1
        elif pc >= 10:
            has_five = True
            break
    return exact4, has_five

def _find_lines(points):
    """
    Return dict: line_key -> list of point indices on that line.
    Only for lines with >= 2 points.
    """
    n = len(points)
    line_points = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            key = _line_key(x1, y1, x2, y2)
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    return {k: list(v) for k, v in line_points.items()}

def _is_valid(points):
    """Check that points are 16 distinct points within bounds and no 5 collinear."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    if not all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in points):
        return False
    _, has_five = _fast_score(points)
    return not has_five

def _line_completion_moves(points, lines):
    """
    Generate candidate moves that complete a 4-point line.
    """
    moves = []
    for key, pts_on_line in lines.items():
        if len(pts_on_line) == 3:
            p1 = points[pts_on_line[0]]
            p2 = points[pts_on_line[1]]
            p3 = points[pts_on_line[2]]
            dx, dy, c = key
            x1, y1 = p1
            used_coords = set(tuple(p) for p in points)
            
            t_values = []
            for p in [p1, p2, p3]:
                if dx != 0:
                    if (p[0] - x1) % dx == 0:
                        t = (p[0] - x1) // dx
                        if dy != 0:
                            if (p[1] - y1) % dy != 0 or (p[1] - y1) // dy != t:
                                continue
                    else:
                        continue
                else:
                    if (p[1] - y1) % dy == 0:
                        t = (p[1] - y1) // dy
                    else:
                        continue
                t_values.append(t)
            
            for t in range(min(t_values) - 3, max(t_values) + 4):
                if t in t_values:
                    continue
                nx = x1 + t * dx
                ny = y1 + t * dy
                if abs(nx) > 1000 or abs(ny) > 1000:
                    continue
                if (nx, ny) in used_coords:
                    continue
                moves.append((nx, ny))
    
    return list(set(moves))

def _local_search(points, deadline, max_no_improve=200):
    """
    Hill-climbing local search.
    Returns best points found and their score.
    """
    cur = [p[:] for p in points]
    cur_score, cur_five = _fast_score(cur)
    if cur_five:
        return cur, cur_score
    
    best = [p[:] for p in cur]
    best_score = cur_score
    no_improve = 0
    rng = random.Random(hash(tuple(tuple(p) for p in cur)) % 2**31)
    
    while time.time() < deadline and no_improve < max_no_improve:
        lines = _find_lines(cur)
        candidates = []
        
        comp_moves = _line_completion_moves(cur, lines)
        for (nx, ny) in comp_moves:
            candidates.append((nx, ny))
        
        for _ in range(8):
            idx = rng.randint(0, 15)
            dx = rng.randint(-6, 6)
            dy = rng.randint(-6, 6)
            nx = cur[idx][0] + dx
            ny = cur[idx][1] + dy
            if abs(nx) <= 1000 and abs(ny) <= 1000:
                candidates.append((nx, ny))
        
        found_improvement = False
        for (nx, ny) in candidates:
            if time.time() > deadline:
                break
            used = set(tuple(p) for p in cur)
            if (nx, ny) in used:
                continue
            
            for idx in range(16):
                new_pts = [p[:] for p in cur]
                new_pts[idx] = [nx, ny]
                if len(set(tuple(p) for p in new_pts)) != 16:
                    continue
                s, f = _fast_score(new_pts)
                if f == 0 and s > cur_score:
                    cur = new_pts
                    cur_score = s
                    no_improve = 0
                    found_improvement = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best = [p[:] for p in cur]
                    break
            if found_improvement:
                break
        
        if not found_improvement:
            no_improve += 1
    
    return best, best_score

def _genetic_repair(points, deadline, rng):
    """
    Try to repair a configuration by breaking 5-collinear sets.
    Returns repaired points or None if failed.
    """
    cur = [p[:] for p in points]
    s, f = _fast_score(cur)
    if not f:
        return cur
    
    for _ in range(50):
        if time.time() > deadline:
            return None
        
        # Find a 5-collinear line
        lines = _find_lines(cur)
        five_line = None
        for key, pts in lines.items():
            if len(pts) >= 5:
                five_line = key
                break
        
        if five_line is None:
            return cur
        
        # Move one point from the 5-line to a random location
        pts_on_line = lines[five_line]
        idx = rng.choice(pts_on_line)
        cur[idx][0] = rng.randint(-200, 200)
        cur[idx][1] = rng.randint(-200, 200)
        
        if len(set(tuple(p) for p in cur)) != 16:
            cur[idx][0] += 1
            cur[idx][1] += 1
        
        s, f = _fast_score(cur)
        if not f:
            return cur
    
    return None

def _crossover(parent1, parent2, rng):
    """
    Select a random subset of points from parent1 and the rest from parent2.
    Returns a new configuration.
    """
    child = []
    for i in range(16):
        if rng.random() < 0.5:
            child.append(parent1[i][:])
        else:
            child.append(parent2[i][:])
    
    # Ensure distinctness
    for attempt in range(20):
        if len(set(tuple(p) for p in child)) == 16:
            break
        dup = rng.randint(0, 15)
        child[dup][0] += rng.randint(-10, 10)
        child[dup][1] += rng.randint(-10, 10)
        if abs(child[dup][0]) > 1000:
            child[dup][0] = rng.randint(-500, 500)
        if abs(child[dup][1]) > 1000:
            child[dup][1] = rng.randint(-500, 500)
    
    return child

def _genetic_algorithm(pop_size, deadline, rng):
    """
    Genetic algorithm with population-based exploration.
    """
    # Initialize population
    population = []
    
    # Add some structured starting points
    grid = [[x, y] for x in range(4) for y in range(4)]
    population.append([p[:] for p in grid])
    
    for _ in range(pop_size - 1):
        # Random configuration
        pts = [[rng.randint(-100, 100), rng.randint(-100, 100)] for _ in range(16)]
        for attempt in range(50):
            if len(set(tuple(p) for p in pts)) == 16:
                break
            dup = rng.randint(0, 15)
            pts[dup][0] += rng.randint(-20, 20)
            pts[dup][1] += rng.randint(-20, 20)
        population.append(pts)
    
    # Evaluate population
    scores = []
    for pts in population:
        s, f = _fast_score(pts)
        scores.append(s if not f else -1)
    
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_pts = [p[:] for p in population[best_idx]]
    
    # Evolutionary loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Selection: keep top half
        ranked = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        survivors = [population[i][:] for i in ranked[:pop_size//2]]
        
        # Generate new generation
        new_pop = list(survivors)
        while len(new_pop) < pop_size:
            if time.time() > deadline:
                break
            
            p1_idx = rng.randint(0, len(survivors) - 1)
            p2_idx = rng.randint(0, len(survivors) - 1)
            if p1_idx == p2_idx:
                p2_idx = (p2_idx + 1) % len(survivors)
            
            child = _crossover(survivors[p1_idx], survivors[p2_idx], rng)
            
            # Mutation
            for _ in range(rng.randint(1, 3)):
                idx = rng.randint(0, 15)
                child[idx][0] += rng.randint(-5, 5)
                child[idx][1] += rng.randint(-5, 5)
                if abs(child[idx][0]) > 1000:
                    child[idx][0] = rng.randint(-500, 500)
                if abs(child[idx][1]) > 1000:
                    child[idx][1] = rng.randint(-500, 500)
            
            # Repair if needed
            s, f = _fast_score(child)
            if f:
                repaired = _genetic_repair(child, deadline, rng)
                if repaired is not None:
                    child = repaired
                    s, f = _fast_score(child)
                    if f:
                        continue
            
            new_pop.append(child)
        
        population = new_pop
        
        # Evaluate new population
        scores = []
        for pts in population:
            s, f = _fast_score(pts)
            scores.append(s if not f else -1)
        
        best_idx = max(range(pop_size), key=lambda i: scores[i])
        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_pts = [p[:] for p in population[best_idx]]
        
        # Local search on best
        if time.time() < deadline - 10:
            improved, improved_score = _local_search(best_pts, deadline - 5, max_no_improve=50)
            if improved_score > best_score:
                best_score = improved_score
                best_pts = [p[:] for p in improved]
        
        # Elitism: ensure best is in next generation
        if best_pts not in population:
            population[0] = [p[:] for p in best_pts]
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = 0
    
    # Initial candidate: 4x4 grid
    grid = [[x, y] for x in range(4) for y in range(4)]
    s, f = _fast_score(grid)
    if f == 0:
        best_score = s
        best_points = [p[:] for p in grid]
    
    # Try several structured starting points
    starting_points = [
        [[x, y] for x in range(4) for y in range(4)],
        [[x*2, y*2] for x in range(4) for y in range(4)],
        [[x*3, y*3] for x in range(4) for y in range(4)],
        [[x + (y % 2), y] for x in range(4) for y in range(4)],
        [[x, y + (x % 2)] for x in range(4) for y in range(4)],
        [[x*2 + (y % 2), y*2] for x in range(4) for y in range(4)],
    ]
    
    for sp in starting_points:
        if time.time() > deadline:
            break
        if len(set(tuple(p) for p in sp)) != 16:
            continue
        if not all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in sp):
            continue
        s, f = _fast_score(sp)
        if f == 0 and s > best_score:
            best_score = s
            best_points = [p[:] for p in sp]
    
    # Local search with multiple restarts
    rng = random.Random(12345)
    
    # First, do local search from best
    if time.time() < deadline - 60:
        result, score = _local_search(best_points, deadline - 30, max_no_improve=150)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in result]
    
    # Genetic algorithm
    if time.time() < deadline - 30:
        ga_points, ga_score = _genetic_algorithm(20, deadline - 10, rng)
        if ga_score > best_score:
            best_score = ga_score
            best_points = [p[:] for p in ga_points]
    
    # Final local search polish
    if time.time() < deadline - 5:
        result, score = _local_search(best_points, deadline, max_no_improve=100)
        if score > best_score:
            best_score = score
            best_points = [p[:] for p in result]
    
    # Final validation
    final = [p[:] for p in best_points]
    assert len(set(tuple(p) for p in final)) == 16
    assert all(abs(p[0]) <= 1000 and abs(p[1]) <= 1000 for p in final)
    s, f = _fast_score(final)
    assert f == 0, f"Five collinear found: {f}"
    
    return final
