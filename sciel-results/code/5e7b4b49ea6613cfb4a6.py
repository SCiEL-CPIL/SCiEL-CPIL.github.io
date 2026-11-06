"""Genetic Algorithm approach for 16 integer points, maximize exact-4 lines."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict

def _line_key(p1, p2):
    """Compute normalized line key (a, b, c) for line through p1, p2: a*x + b*y = c."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx_n, dy_n = dx // g, dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n, dy_n = -dx_n, -dy_n
    a, b = dy_n, -dx_n
    c = a * p1[0] + b * p1[1]
    g2 = math.gcd(abs(a), abs(b))
    if g2 > 1:
        a, b, c = a // g2, b // g2, c // g2
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _score_lines_fast(points):
    """Return (num_exact_4_lines, has_5_collinear) for 16 points."""
    n = 16
    line_count = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is not None:
                line_count[key] += 1
    
    exact_4 = 0
    has_5 = False
    for key, pair_count in line_count.items():
        # Find k such that C(k,2) = pair_count
        k = 0
        # Max k is 16, C(16,2)=120. Pair counts are small.
        # We can just check k=4,5,6...
        # C(4,2)=6, C(5,2)=10, C(6,2)=15
        if pair_count == 6:
            exact_4 += 1
        elif pair_count >= 10:
            has_5 = True
    return exact_4, has_5

def _build_line_map(points):
    """Build line_pts mapping: line_key -> set of point indices."""
    n = len(points)
    line_pts = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_pts:
                line_pts[key] = set()
            line_pts[key].add(i)
            line_pts[key].add(j)
    return line_pts

def _find_5_collinear(line_pts):
    """Find a line with 5 or more points. Returns (key, set_of_indices) or (None, None)."""
    for key, pts in line_pts.items():
        if len(pts) >= 5:
            return key, pts
    return None, None

def _repair_5_collinear(points, line_pts, rng):
    """
    If there is a 5-collinear set, move one of the points to break it.
    Try to move it to a position that creates a new 4-collinear line if possible.
    """
    key, pts_5 = _find_5_collinear(line_pts)
    if key is None:
        return points, line_pts, False
    
    # Pick a random point from the 5-collinear set to move
    idx_to_move = list(pts_5)[rng.randint(0, len(pts_5)-1)]
    old_pt = points[idx_to_move][:]
    
    # Try to find a new position for this point
    # 1. Try completing a 3-line to a 4-line
    # 2. If not, random small shift
    
    # Identify 3-lines that do NOT contain the moved point
    three_lines = []
    for k, pts in line_pts.items():
        if len(pts) == 3 and idx_to_move not in pts:
            three_lines.append((k, pts))
    
    new_pt = None
    if three_lines:
        # Pick a random 3-line
        k_target, pts_target = three_lines[rng.randint(0, len(three_lines)-1)]
        a, b, c = k_target
        # Find integer solution on this line near the old point
        # a*x + b*y = c
        # Try x near old_pt[0]
        base_x = old_pt[0]
        base_y = old_pt[1]
        found = False
        for delta_x in range(-10, 11):
            x_try = base_x + delta_x
            if abs(x_try) > 1000: continue
            if b != 0:
                num = c - a * x_try
                if num % b == 0:
                    y_try = num // b
                    if abs(y_try) <= 1000:
                        pt = [x_try, y_try]
                        # Check if this point is already in the set
                        if not any(pt == p for p in points):
                            # Check if this creates a new 5-collinear with OTHER points
                            # We are removing idx_to_move, so we check against points except idx_to_move
                            # But we also need to ensure it doesn't land on another line with 3 points (making 4 is good, 5 is bad)
                            # Actually, if it lands on a line with 3 points, it becomes 4.
                            # If it lands on a line with 4 points, it becomes 5.
                            # So we need to check if pt is on any line with 4 points (excluding the line we are completing? No, including)
                            # Let's just do a quick check:
                            bad = False
                            for k2, pts2 in line_pts.items():
                                if k2 == k_target: continue
                                if len(pts2) >= 4:
                                    # Check if pt is on this line
                                    # a2*x + b2*y = c2
                                    a2, b2, c2 = k2
                                    if a2 * pt[0] + b2 * pt[1] == c2:
                                        bad = True
                                        break
                            if not bad:
                                new_pt = pt
                                found = True
                                break
        if found:
            points[idx_to_move] = new_pt
            # Rebuild line map
            line_pts = _build_line_map(points)
            return points, line_pts, True
    
    # If no good 4-line completion found, just do a random small shift
    # Ensure we break the 5-collinear
    for _ in range(20):
        new_x = old_pt[0] + rng.randint(-3, 3)
        new_y = old_pt[1] + rng.randint(-3, 3)
        pt = [new_x, new_y]
        if abs(new_x) > 1000 or abs(new_y) > 1000: continue
        if any(pt == p for p in points): continue
        # Check if this breaks the 5-collinear
        # It will break it if it's not on the line a*x+b*y=c
        if a * pt[0] + b * pt[1] != c:
             # Check if it creates a new 5-collinear
             # Quick check: is it on any existing line with 4 points?
             bad = False
             for k2, pts2 in line_pts.items():
                 if k2 == key: continue
                 if len(pts2) >= 4:
                     a2, b2, c2 = k2
                     if a2 * pt[0] + b2 * pt[1] == c2:
                         bad = True
                         break
             if not bad:
                 points[idx_to_move] = pt
                 line_pts = _build_line_map(points)
                 return points, line_pts, True
             
    # Fallback: random shift that definitely breaks it (might create 5-collinear elsewhere, handled by next iteration)
    new_x = old_pt[0] + rng.randint(-5, 5)
    new_y = old_pt[1] + rng.randint(-5, 5)
    points[idx_to_move] = [new_x, new_y]
    line_pts = _build_line_map(points)
    return points, line_pts, True

def _local_search(points, deadline, rng, max_steps=50):
    """Local search to improve the number of exact-4 lines."""
    current = [p[:] for p in points]
    current_score, h5 = _score_lines_fast(current)
    if h5:
        # Repair first
        line_pts = _build_line_map(current)
        current, line_pts, _ = _repair_5_collinear(current, line_pts, rng)
        current_score, h5 = _score_lines_fast(current)
        if h5:
            return current, 0
    
    best = [p[:] for p in current]
    best_score = current_score
    
    T = 0.5
    no_improve = 0
    steps = 0
    
    while time.time() < deadline - 5 and no_improve < max_steps and steps < 200:
        idx = rng.randint(0, 15)
        old = current[idx][:]
        
        # Mutation: small shift
        new_x = old[0] + rng.randint(-4, 4)
        new_y = old[1] + rng.randint(-4, 4)
        new_pt = [new_x, new_y]
        
        if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
            continue
        if any(new_pt == p for p in current):
            continue
        
        current[idx] = new_pt
        s_new, h5_new = _score_lines_fast(current)
        steps += 1
        
        if h5_new:
            # Try to repair
            line_pts = _build_line_map(current)
            current, line_pts, _ = _repair_5_collinear(current, line_pts, rng)
            s_new, h5_new = _score_lines_fast(current)
            if h5_new:
                current[idx] = old
                no_improve += 1
                continue
        
        if s_new > current_score or (T > 0.05 and rng.random() < math.exp((s_new - current_score) / T)):
            if s_new > current_score:
                no_improve = 0
                if s_new > best_score:
                    best_score = s_new
                    best = [p[:] for p in current]
            else:
                no_improve += 1
            current_score = s_new
        else:
            current[idx] = old
            no_improve += 1
        
        T = max(0.05, T * 0.99)
    
    return best, best_score

def _crossover(p1, p2, rng):
    """Uniform crossover: pick random subset of points from p1, rest from p2."""
    child = []
    for i in range(16):
        if rng.random() < 0.5:
            child.append(p1[i][:])
        else:
            child.append(p2[i][:])
    # Ensure distinctness? Crossover might create duplicates.
    # If duplicates, replace with random point.
    for i in range(16):
        for j in range(i+1, 16):
            if child[i] == child[j]:
                child[j] = [rng.randint(-100, 100), rng.randint(-100, 100)]
    return child

def _mutate(points, rng):
    """Small random mutation."""
    new_points = [p[:] for p in points]
    for _ in range(rng.randint(1, 3)):
        idx = rng.randint(0, 15)
        new_x = new_points[idx][0] + rng.randint(-5, 5)
        new_y = new_points[idx][1] + rng.randint(-5, 5)
        if abs(new_x) <= 1000 and abs(new_y) <= 1000:
             if not any([new_x, new_y] == p for p in new_points):
                 new_points[idx] = [new_x, new_y]
    return new_points

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    rng = random.Random(42)
    
    # Initialize population
    population = []
    pop_size = 20
    
    # 1. 4x4 Grid variants
    base_grid = [[x, y] for x in range(4) for y in range(4)]
    for _ in range(5):
        p = [pt[:] for pt in base_grid]
        # Light perturbation
        for i in range(rng.randint(1, 3)):
            idx = rng.randint(0, 15)
            p[idx] = [p[idx][0] + rng.randint(-2, 2), p[idx][1] + rng.randint(-2, 2)]
        population.append(p)
        
    # 2. Random points
    for _ in range(5):
        p = []
        while len(p) < 16:
            pt = [rng.randint(-10, 10), rng.randint(-10, 10)]
            if pt not in p:
                p.append(pt)
        population.append(p)
        
    # 3. Structured random (e.g. on a circle or other lattice)
    for _ in range(5):
        p = []
        # Points on a circle of radius 5
        angles = [rng.uniform(0, 2*math.pi) for _ in range(16)]
        for a in angles:
            x = int(round(5 * math.cos(a)))
            y = int(round(5 * math.sin(a)))
            pt = [x, y]
            if pt not in p:
                p.append(pt)
        # Fill if needed
        while len(p) < 16:
            pt = [rng.randint(-10, 10), rng.randint(-10, 10)]
            if pt not in p:
                p.append(pt)
        population.append(p)
        
    # 4. Another structured: 5x5 grid subset
    for _ in range(5):
        p = []
        grid_5x5 = [[x, y] for x in range(5) for y in range(5)]
        rng.shuffle(grid_5x5)
        p = [pt[:] for pt in grid_5x5[:16]]
        population.append(p)

    # Evaluate initial population
    scored_pop = []
    for p in population:
        s, h5 = _score_lines_fast(p)
        if h5:
            s = -10  # Penalty for invalid
        scored_pop.append((s, p))
    
    scored_pop.sort(key=lambda x: x[0], reverse=True)
    best_score = scored_pop[0][0]
    best_points = scored_pop[0][1]
    
    # GA Loop
    generation = 0
    while time.time() < deadline - 20:
        generation += 1
        
        # Selection: Tournament
        parents = []
        for _ in range(pop_size // 2):
            # Pick 3 random, take best
            cands = [scored_pop[rng.randint(0, pop_size-1)] for _ in range(3)]
            parent = max(cands, key=lambda x: x[0])
            parents.append(parent[1])
            
        # Crossover and Mutation
        new_population = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[i+1]
            
            # Crossover
            child = _crossover(p1, p2, rng)
            # Mutation
            child = _mutate(child, rng)
            
            # Repair if invalid
            s, h5 = _score_lines_fast(child)
            if h5:
                line_pts = _build_line_map(child)
                child, line_pts, _ = _repair_5_collinear(child, line_pts, rng)
                s, h5 = _score_lines_fast(child)
                if h5:
                    s = -10
            
            new_population.append((s, child))
            
        # Add some mutated elites
        for s, p in scored_pop[:5]:
            mp = _mutate(p, rng)
            s_mp, h5_mp = _score_lines_fast(mp)
            if h5_mp:
                line_pts = _build_line_map(mp)
                mp, line_pts, _ = _repair_5_collinear(mp, line_pts, rng)
                s_mp, h5_mp = _score_lines_fast(mp)
                if h5_mp:
                    s_mp = -10
            new_population.append((s_mp, mp))
            
        # Combine and select top pop_size
        # Ensure we have enough
        while len(new_population) < pop_size:
            p = _mutate(scored_pop[0][1], rng)
            s, h5 = _score_lines_fast(p)
            if h5: s = -10
            new_population.append((s, p))
            
        # Sort and keep best
        new_population.sort(key=lambda x: x[0], reverse=True)
        scored_pop = new_population[:pop_size]
        
        # Update best
        if scored_pop[0][0] > best_score:
            best_score = scored_pop[0][0]
            best_points = scored_pop[0][1]
            
        # Local search on the best individual
        if time.time() < deadline - 10:
            # Run local search on the current best
            ls_pts, ls_score = _local_search(best_points, deadline, rng, max_steps=100)
            if ls_score > best_score:
                best_score = ls_score
                best_points = ls_pts
                # Update population with this improved best
                scored_pop[0] = (best_score, best_points)
                scored_pop.sort(key=lambda x: x[0], reverse=True)

    # Final verification
    final_score, final_h5 = _score_lines_fast(best_points)
    if final_h5:
        # Fallback to 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = 10
    else:
        best_score = final_score

    return best_points
# EVOLVE-BLOCK-END
