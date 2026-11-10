# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _count_incidences(points, lines):
    """
    Count incidences between points and lines.
    points: (N, 2) array
    lines: (M, 3) array
    Returns total incidences.
    """
    px = points[:, 0:1] # (N, 1)
    py = points[:, 1:2] # (N, 1)
    lA = lines[:, 0:1].T # (1, M)
    lB = lines[:, 1:2].T # (1, M)
    lC = lines[:, 2:3].T # (1, M)
    
    vals = px * lA + py * lB + lC
    incs = np.sum(vals == 0)
    return incs

def _generate_line_pool(max_coord=15):
    """
    Generate a pool of candidate lines with small integer coefficients.
    These are lines likely to form dense configurations.
    """
    lines = set()
    # Horizontal, Vertical, Diagonals
    for c in range(-max_coord, max_coord + 1):
        lines.add(_normalize_line(1, 0, -c)) # x = c
        lines.add(_normalize_line(0, 1, -c)) # y = c
        lines.add(_normalize_line(1, 1, -c)) # x + y = c
        lines.add(_normalize_line(1, -1, -c)) # x - y = c
        
    # Other simple slopes: 2, -2, 1/2, -1/2
    for c in range(-max_coord * 2, max_coord * 2 + 1):
        lines.add(_normalize_line(2, 1, -c)) # 2x + y = c
        lines.add(_normalize_line(1, 2, -c)) # x + 2y = c
        lines.add(_normalize_line(2, -1, -c)) # 2x - y = c
        lines.add(_normalize_line(1, -2, -c)) # x - 2y = c
        
    # Additional slopes: 3, -3, 1/3, -1/3
    for c in range(-max_coord * 3, max_coord * 3 + 1):
        lines.add(_normalize_line(3, 1, -c)) # 3x + y = c
        lines.add(_normalize_line(1, 3, -c)) # x + 3y = c
        lines.add(_normalize_line(3, -1, -c)) # 3x - y = c
        lines.add(_normalize_line(1, -3, -c)) # x - 3y = c
        
    return list(lines)

def _evaluate_lines(lines_set, num_points=54):
    """
    Evaluate a set of lines: find all integer intersection points,
    count incidences for each, select top num_points, return score.
    """
    candidates = set()
    ls = list(lines_set)
    n_l = len(ls)
    
    for i in range(n_l):
        A1, B1, C1 = ls[i]
        for j in range(i + 1, n_l):
            A2, B2, C2 = ls[j]
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue # Parallel
            # x = (B1*C2 - B2*C1) / det
            # y = (A2*C1 - A1*C2) / det
            num_x = B1 * C2 - B2 * C1
            num_y = A2 * C1 - A1 * C2
            
            if num_x % det == 0 and num_y % det == 0:
                x = num_x // det
                y = num_y // det
                # Limit coordinate range
                if -200 <= x <= 200 and -200 <= y <= 200:
                    candidates.add((x, y))
    
    if not candidates:
        return 0, []
        
    # Count incidences for each candidate point
    point_degrees = {}
    for p in candidates:
        x, y = p
        deg = 0
        for l in ls:
            A, B, C = l
            if A * x + B * y + C == 0:
                deg += 1
        if deg > 0:
            point_degrees[p] = deg
            
    # Select top num_points points
    sorted_points = sorted(point_degrees.items(), key=lambda item: item[1], reverse=True)
    selected_points = [p for p, d in sorted_points[:num_points]]
    score = sum(d for p, d in sorted_points[:num_points])
    
    return score, selected_points

def _find_best_lines_and_points(pool_lines, num_lines=27, num_points=54, time_limit=580):
    """
    1. Select num_lines from pool_lines to maximize intersection potential.
    2. Select num_points from intersections to maximize incidences.
    """
    start_time = time.time()
    
    # Strategy: 
    # 1. Pick a base set of lines (e.g., grid lines).
    # 2. Compute all integer intersection points of these lines.
    # 3. For each intersection point, count how many of the selected lines pass through it.
    # 4. Select the top 54 points with highest degree.
    # 5. The score is the sum of degrees of selected points.
    
    # Initial set: Use a smart initialization
    # Group lines by their "type" (slope) to ensure diversity
    # Then pick lines that create many integer intersections
    
    # First, let's create an initial set with good diversity
    # Take lines from different slope families
    pool_set = set(pool_lines)
    
    # Initialize with a mix of different line types
    # Prefer lines with small coefficients
    def line_complexity(l):
        A, B, C = l
        return abs(A) + abs(B) + abs(C)
    
    sorted_pool = sorted(pool_set, key=line_complexity)
    
    # Greedy initialization: start with a few lines, add the one that maximizes new high-degree points
    current_set = set()
    
    # Start with some promising lines (small coefficients)
    for l in sorted_pool[:10]:
        current_set.add(l)
    
    # Now greedily add lines that maximize the score
    remaining = list(pool_set - current_set)
    random.shuffle(remaining)
    
    while len(current_set) < num_lines and remaining:
        best_line = None
        best_score = -1
        
        # Try a subset of remaining lines for speed
        candidates = remaining[:min(50, len(remaining))]
        random.shuffle(candidates)
        
        for l in candidates:
            test_set = current_set | {l}
            score, _ = _evaluate_lines(test_set, num_points)
            if score > best_score:
                best_score = score
                best_line = l
        
        if best_line is not None:
            current_set.add(best_line)
            remaining.remove(best_line)
        else:
            # Just add a random line
            l = remaining.pop()
            current_set.add(l)
    
    current_score, current_points = _evaluate_lines(current_set, num_points)
    best_score = current_score
    best_lines = list(current_set)
    best_points = current_points
    
    # Local search: single swap and double swap
    remaining_lines = list(pool_set - current_set)
    
    steps = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        # Check time more frequently
        if time.time() - start_time > time_limit - 5:
            break
            
        if no_improve_count > max_no_improve:
            # Restart from best with perturbation
            current_set = set(best_lines)
            # Perturb: swap 2-3 lines
            for _ in range(random.randint(2, 3)):
                if remaining_lines:
                    l_out = random.choice(list(current_set))
                    l_in = random.choice(remaining_lines)
                    current_set.remove(l_out)
                    current_set.add(l_in)
                    if l_in in remaining_lines:
                        remaining_lines.remove(l_in)
            no_improve_count = 0
            current_score, current_points = _evaluate_lines(current_set, num_points)
            continue
        
        # Try single swap
        l_out = random.choice(list(current_set))
        
        # Try multiple candidate lines in for speed
        num_candidates = min(20, len(remaining_lines))
        if num_candidates == 0:
            break
        candidates = random.sample(remaining_lines, num_candidates)
        
        found_improvement = False
        for l_in in candidates:
            new_set = (current_set - {l_out}) | {l_in}
            
            new_score, new_points = _evaluate_lines(new_set, num_points)
            
            if new_score > best_score:
                best_score = new_score
                best_lines = list(new_set)
                best_points = new_points
                current_set = new_set
                # Update remaining
                if l_out in remaining_lines:
                    remaining_lines.append(l_out)
                if l_in in remaining_lines:
                    remaining_lines.remove(l_in)
                found_improvement = True
                no_improve_count = 0
                break
            elif new_score > current_score:
                # Accept move (hill climbing)
                current_score = new_score
                current_set = new_set
                current_lines = list(current_set)
                current_points = new_points
                if l_out in remaining_lines:
                    remaining_lines.append(l_out)
                if l_in in remaining_lines:
                    remaining_lines.remove(l_in)
                found_improvement = True
                no_improve_count = 0
                break
        
        if not found_improvement:
            no_improve_count += 1
        
        steps += 1
    
    # Convert best_points to array
    if len(best_points) < num_points:
        # Pad with random integer points if needed
        pts_set = set(best_points)
        while len(pts_set) < num_points:
            p = (random.randint(-20, 20), random.randint(-20, 20))
            if p not in pts_set:
                pts_set.add(p)
        best_points = list(pts_set)
    
    points_arr = np.array(best_points[:num_points], dtype=np.int64)
    lines_arr = np.array(best_lines, dtype=np.int64)
    
    # Verify shapes
    if points_arr.shape != (54, 2):
        # Pad with dummy points if not enough
        while points_arr.shape[0] < 54:
            p = [random.randint(-20, 20), random.randint(-20, 20)]
            if tuple(p) not in [tuple(pt) for pt in points_arr]:
                points_arr = np.vstack([points_arr, np.array(p)])
        points_arr = points_arr[:54]
        
    if lines_arr.shape != (27, 3):
        # Should not happen, but handle it
        pass
        
    return points_arr, lines_arr, best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses a line-based constructive strategy with greedy initialization and local search.
    """
    start_time = time.time()
    timeout = 590  # Safety margin
    
    # Generate pool of candidate lines
    pool = _generate_line_pool(max_coord=15)
    
    # Run the line-based search
    points, lines, score = _find_best_lines_and_points(pool, 27, 54, timeout - 10)
    
    # Final validation
    points = points.astype(np.int64)
    lines = lines.astype(np.int64)
    
    assert points.shape == (54, 2)
    assert lines.shape == (27, 3)
    
    # Check duplicates
    pts_set = set([tuple(p) for p in points])
    if len(pts_set) < 54:
        # Fallback: regenerate unique points
        pts_list = list(pts_set)
        while len(pts_list) < 54:
            p = (random.randint(-20, 20), random.randint(-20, 20))
            if p not in pts_set:
                pts_list.append(p)
                pts_set.add(p)
        points = np.array(pts_list[:54], dtype=np.int64)
        
    lines_set = set([tuple(l) for l in lines])
    if len(lines_set) < 27:
        # Fallback: add random valid lines
        lines_list = list(lines_set)
        while len(lines_list) < 27:
            A = random.randint(-5, 5)
            B = random.randint(-5, 5)
            if A == 0 and B == 0:
                continue
            C = random.randint(-20, 20)
            nl = _normalize_line(A, B, C)
            if nl not in lines_set:
                lines_list.append(nl)
                lines_set.add(nl)
        lines = np.array(lines_list[:27], dtype=np.int64)
        
    return points, lines

# EVOLVE-BLOCK-END
