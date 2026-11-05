"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _count_collinear_triples(points):
    """Count the number of collinear triples among the points."""
    n = len(points)
    triples = 0
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    triples += 1
    return triples

def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                if (x2-x1)*(y3-y1) != (y2-y1)*(x3-x1):
                    continue
                for l in range(k+1, n):
                    x4, y4 = points[l]
                    if (x2-x1)*(y4-y1) == (y2-y1)*(x4-x1):
                        return True
    return False

def _max_3free_subset(points, deadline=None):
    """Branch and bound to find the largest subset with no three collinear."""
    n = len(points)
    best = [0]
    best_set = [set()]
    
    # Precompute collinearity
    collinear = [[[False]*n for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            for k in range(j+1, n):
                x3, y3 = points[k]
                if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                    collinear[i][j][k] = True
                    collinear[j][i][k] = True
                    collinear[k][i][j] = True
                    collinear[i][k][j] = True
                    collinear[j][k][i] = True
                    collinear[k][j][i] = True
    
    def dfs(idx, current, current_size):
        if deadline and time.time() > deadline:
            return
        if idx >= n:
            if current_size > best[0]:
                best[0] = current_size
                best_set[0] = set(current)
            return
        
        if current_size + (n - idx) <= best[0]:
            return
        
        dfs(idx + 1, current, current_size)
        
        can_add = True
        for i, ci in enumerate(current):
            for j in range(i+1, len(current)):
                cj = current[j]
                if collinear[idx][ci][cj]:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            current.append(idx)
            dfs(idx + 1, current, current_size + 1)
            current.pop()
    
    dfs(0, [], 0)
    return best[0]

def _greedy_hypergraph_construct(seed, deadline):
    """
    Structural approach: model as 3-uniform hypergraph.
    Vertices = 20 point slots. Edges = desired collinear triples.
    Greedy set-cover: iteratively select the point slot with highest degree
    (participation in most potential triples) and fix its position to
    maximize line incidences while enforcing no-4-collinear.
    
    Key inversion: choose WHICH triples to form first, then solve for positions.
    """
    rng = random.Random(seed)
    n = 20
    
    # Phase 1: Determine target triple structure via greedy degree selection
    # Start with all pairs as potential "lines" that could host a third point
    # Represent hypergraph: each potential edge is a pair (i,j) that we want
    # to extend to a triple (i,j,k)
    
    # Build a list of potential triples (i, j, k) - all C(20,3) = 1140
    all_triples = list(combinations(range(n), 3))
    
    # Greedy: select triples that share vertices (high degree) to create
    # dense line structures. Each line can have exactly 3 points.
    # Strategy: partition 20 points into groups that form lines.
    # With 20 points, we can have up to floor(20/3) = 6 disjoint triples,
    # but better: overlapping triples on shared lines.
    
    # Better approach: build a line arrangement.
    # Each line has exactly 3 points. Points can be on multiple lines.
    # We want to maximize total collinear triples = number of lines.
    
    # Greedy line construction:
    # 1. Start with empty assignment
    # 2. Repeatedly pick a pair of unassigned or partially-assigned points
    #    and add a third point on their line (if possible)
    
    # Actually, let's use a different greedy: 
    # Build a "line incidence" structure.
    # Each point can be on at most ~4 lines (to keep score low).
    # Target: ~10-12 lines of 3 points each.
    
    # Phase 2: Assign coordinates to satisfy the line structure
    # Use a constructive geometry: place points on a set of predefined lines.
    
    # Strategy: Use a finite field / modular arithmetic construction
    # or explicit line arrangement.
    
    # Let's try: pick a set of lines (slopes and intercepts), then assign
    # 3 points to each line, ensuring no 4 are collinear.
    
    # Predefined line templates (slope, intercept) - use rational slopes
    # to keep coordinates integer and bounded.
    
    line_templates = []
    # Horizontal lines
    for y in [-80, -40, 0, 40, 80]:
        line_templates.append((0, 1, y))  # 0*x + 1*y = y
    # Vertical lines
    for x in [-80, -40, 0, 40, 80]:
        line_templates.append((1, 0, x))  # 1*x + 0*y = x
    # Diagonal lines with various slopes
    for slope_num in [1, 2, 3, -1, -2, -3]:
        for slope_den in [1, 2, 3]:
            if gcd(slope_num, slope_den) == 1:
                for c in [-60, -30, 0, 30, 60]:
                    # line: slope_num * x - slope_den * y = c
                    A = slope_num
                    B = -slope_den
                    C = -c
                    g = gcd(abs(A), gcd(abs(B), abs(C))) if (A != 0 or B != 0) else 1
                    if g > 0:
                        A //= g
                        B //= g
                        C //= g
                    if A < 0 or (A == 0 and B < 0):
                        A, B, C = -A, -B, -C
                    line_templates.append((A, B, C))
    
    # Deduplicate lines
    unique_lines = []
    seen_lines = set()
    for line in line_templates:
        if line not in seen_lines:
            seen_lines.add(line)
            unique_lines.append(line)
    
    rng.shuffle(unique_lines)
    
    # Greedy: select lines and assign 3 points to each
    # Track which point slots are used and which lines they're on
    point_lines = [set() for _ in range(n)]  # point_lines[i] = set of line indices
    line_points = []  # list of (line, [point_indices])
    used_points = set()
    
    # For each candidate line, check if we can assign 3 new points to it
    # such that no point ends up on 4+ lines (to keep no-4-collinear)
    # and no 4 points end up on the same line.
    
    # Actually, the no-4-collinear constraint means: for any line, at most 3
    # of our 20 points lie on it. Since we're assigning exactly 3 points per
    # selected line, we just need to ensure that no OTHER line (not in our
    # selected set) accidentally has 4 of our points.
    
    # This is complex. Let's use a simpler constructive approach:
    # Place points one by one, each on a line with 2 previously placed points.
    
    points = []
    used_coords = set()
    # Track lines: line_key -> list of point indices on that line
    lines_map = {}
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        g = gcd(abs(A), gcd(abs(B), abs(C))) if (A != 0 or B != 0) else 1
        if g > 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    def find_third_point_on_line(line_key, existing_points, exclude_idx):
        """Find an integer point on the given line that is not already used."""
        A, B, C = line_key
        candidates = []
        if B != 0:
            for x in range(-100, 101):
                num = -A * x - C
                if num % B == 0:
                    y = num // B
                    if -100 <= y <= 100:
                        candidates.append((x, y))
        elif A != 0:
            for y in range(-100, 101):
                num = -B * y - C
                if num % A == 0:
                    x = num // A
                    if -100 <= x <= 100:
                        candidates.append((x, y))
        
        rng.shuffle(candidates)
        for cand in candidates:
            if cand not in used_coords:
                return cand
        return None
    
    # Greedy construction:
    # 1. Place first 2 points (seed)
    # 2. Repeatedly: pick a pair of existing points, find a third on their line,
    #    add it. This creates one new collinear triple per step.
    # 3. Continue until 20 points or no more valid third points.
    # 4. Fill remaining with random points not on any existing line with 2+ points.
    
    # Initial seed points
    seed_points = []
    for _ in range(2):
        while True:
            x = rng.randint(-90, 90)
            y = rng.randint(-90, 90)
            if (x, y) not in used_coords:
                used_coords.add((x, y))
                seed_points.append([x, y])
                break
    
    points = seed_points
    
    # Greedy: add points to form new triples
    max_triples = 12  # Target number of collinear triples
    attempts = 0
    max_attempts = 2000
    
    while len(points) < n and attempts < max_attempts and time.time() < deadline - 10:
        attempts += 1
        
        # Pick a random pair of existing points
        if len(points) < 2:
            # Need to add a random point
            while True:
                x = rng.randint(-90, 90)
                y = rng.randint(-90, 90)
                if (x, y) not in used_coords:
                    used_coords.add((x, y))
                    points.append([x, y])
                    break
            continue
        
        i, j = rng.sample(range(len(points)), 2)
        line_key = get_line_key(points[i], points[j])
        
        # Check: how many points are already on this line?
        line_count = 0
        for p in points:
            if line_key[0] * p[0] + line_key[1] * p[1] + line_key[2] == 0:
                line_count += 1
        
        if line_count >= 2:
            # This line already has 2+ points. Adding a third creates a new triple.
            # But if line_count == 2, adding makes it 3 (good).
            # If line_count > 2, we'd create a 4-collinear (bad).
            if line_count > 2:
                continue
            
            # Find a third point on this line
            third = find_third_point_on_line(line_key, points, None)
            if third is None:
                continue
            
            # Check: does adding this point create any 4-collinear with other lines?
            # For each other pair of existing points, check if the new point lies
            # on their line AND that line already has 2+ points.
            valid = True
            new_point_idx = len(points)
            for a in range(len(points)):
                for b in range(a + 1, len(points)):
                    if a == i and b == j:
                        continue
                    lk = get_line_key(points[a], points[b])
                    if lk[0] * third[0] + lk[1] * third[1] + lk[2] == 0:
                        # New point is on line (a,b). Check how many points on that line.
                        cnt = 0
                        for p in points:
                            if lk[0] * p[0] + lk[1] * p[1] + lk[2] == 0:
                                cnt += 1
                        # cnt includes points a and b. If cnt >= 2, adding makes it 3.
                        # But if cnt >= 3, adding makes it 4 (bad).
                        if cnt >= 3:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            used_coords.add(third)
            points.append(list(third))
        
        # Occasionally add a random point to avoid getting stuck
        elif rng.random() < 0.3:
            while True:
                x = rng.randint(-90, 90)
                y = rng.randint(-90, 90)
                if (x, y) not in used_coords:
                    used_coords.add((x, y))
                    points.append([x, y])
                    break
    
    # Fill remaining with random points
    while len(points) < n and time.time() < deadline - 10:
        while True:
            x = rng.randint(-90, 90)
            y = rng.randint(-90, 90)
            if (x, y) not in used_coords:
                used_coords.add((x, y))
                points.append([x, y])
                break
    
    if len(points) != n:
        return None
    
    # Validate: no four collinear
    if _has_four_collinear(points):
        return None
    
    return points

def _local_search_triple_max(points, deadline):
    """
    Local search: try to increase the number of collinear triples by
    moving individual points onto lines defined by pairs of other points.
    """
    points = [p[:] for p in points]
    current_triples = _count_collinear_triples(points)
    
    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        g = gcd(abs(A), gcd(abs(B), abs(C))) if (A != 0 or B != 0) else 1
        if g > 0:
            A //= g
            B //= g
            C //= g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    def find_point_on_line(line_key, used_set):
        A, B, C = line_key
        candidates = []
        if B != 0:
            for x in range(-100, 101):
                num = -A * x - C
                if num % B == 0:
                    y = num // B
                    if -100 <= y <= 100:
                        candidates.append((x, y))
        elif A != 0:
            for y in range(-100, 101):
                num = -B * y - C
                if num % A == 0:
                    x = num // A
                    if -100 <= x <= 100:
                        candidates.append((x, y))
        rng_local = random.Random(hash((A, B, C, time.time_ns())))
        rng_local.shuffle(candidates)
        for cand in candidates:
            if cand not in used_set:
                return cand
        return None
    
    rng = random.Random(hash(tuple(map(tuple, points))))
    
    while time.time() < deadline:
        improved = False
        n = len(points)
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            orig = points[i][:]
            used_set = set(tuple(p) for p in points if p != orig)
            
            # Try moving point i onto a line defined by pair (j, k)
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    
                    line_key = get_line_key(points[j], points[k])
                    
                    # Check if point i is already on this line
                    if line_key[0] * orig[0] + line_key[1] * orig[1] + line_key[2] == 0:
                        continue
                    
                    # Find a position on this line
                    new_pos = find_point_on_line(line_key, used_set)
                    if new_pos is None:
                        continue
                    
                    # Check: does this create 4-collinear?
                    # The line (j,k) now has points j, k, and new_pos.
                    # Check if any other point is also on this line.
                    four_collinear = False
                    for m in range(n):
                        if m == i or m == j or m == k:
                            if line_key[0] * points[m][0] + line_key[1] * points[m][1] + line_key[2] == 0:
                                four_collinear = True
                                break
                    if four_collinear:
                        continue
                    
                    # Check: does new_pos create 4-collinear on any OTHER line?
                    # For each pair (a, b) not involving i, j, k:
                    valid = True
                    for a in range(n):
                        if a == i or a == j or a == k:
                            continue
                        for b in range(a + 1, n):
                            if b == i or b == j or b == k:
                                continue
                            lk = get_line_key(points[a], points[b])
                            if lk[0] * new_pos[0] + lk[1] * new_pos[1] + lk[2] == 0:
                                # Check how many points on this line
                                cnt = 0
                                for m in range(n):
                                    if m == i:
                                        continue
                                    if lk[0] * points[m][0] + lk[1] * points[m][1] + lk[2] == 0:
                                        cnt += 1
                                if cnt >= 3:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Try the move
                    points[i] = list(new_pos)
                    new_triples = _count_collinear_triples(points)
                    if new_triples > current_triples:
                        current_triples = new_triples
                        improved = True
                        break
                    else:
                        points[i] = orig[:]
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s limit
    
    best_points = None
    best_triples = -1
    
    # Phase 1: Greedy hypergraph construction with multiple seeds
    for seed in range(80):
        if time.time() > deadline - 30:
            break
        
        points = _greedy_hypergraph_construct(seed, deadline)
        if points is None:
            continue
        
        triples = _count_collinear_triples(points)
        if triples > best_triples:
            best_triples = triples
            best_points = [p[:] for p in points]
    
    # Phase 2: Local search to maximize triples
    if best_points is not None and time.time() < deadline - 15:
        improved = _local_search_triple_max(best_points, deadline - 5)
        if len(set(tuple(p) for p in improved)) == 20 and not _has_four_collinear(improved):
            improved_triples = _count_collinear_triples(improved)
            if improved_triples > best_triples:
                best_triples = improved_triples
                best_points = [p[:] for p in improved]
    
    # Phase 3: Additional local search passes on top configurations
    if best_points is not None and time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            improved = _local_search_triple_max(best_points, deadline - 3)
            if len(set(tuple(p) for p in improved)) == 20 and not _has_four_collinear(improved):
                improved_triples = _count_collinear_triples(improved)
                if improved_triples > best_triples:
                    best_triples = improved_triples
                    best_points = [p[:] for p in improved]
                else:
                    break
    
    # Return the best points found
    if best_points is not None:
        return best_points
    else:
        # Fallback: parabola (no three collinear, score = 20)
        return [[x, x * x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
