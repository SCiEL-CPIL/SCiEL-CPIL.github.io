# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd
from itertools import combinations

def _normalize_line(A, B, C):
    """Normalize line coefficients (A, B, C) to a canonical form."""
    if A == 0 and B == 0:
        return (0, 0, 0)
    
    # Find GCD of A, B, C
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    
    if g > 1:
        A //= g
        B //= g
        C //= g
        
    # Ensure first non-zero entry is positive
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
    Count total incidences between points and lines.
    points: (N, 2) array
    lines: (K, 3) array
    Returns: total incidences (int)
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    # Vectorized calculation
    # A*x + B*y + C == 0
    # points is (N, 2), lines is (K, 3)
    # We need sum over all pairs
    # Efficient way: for each line, count points.
    total = 0
    for i in range(len(lines)):
        A, B, C = lines[i]
        vals = A * points[:, 0] + B * points[:, 1] + C
        total += np.sum(vals == 0)
    return total

def _get_best_points_for_lines(lines, n_points=54):
    """
    Given a set of lines, find the best n_points distinct integer points
    that maximize total incidences.
    Strategy:
    1. Generate all intersection points of pairs of lines.
    2. Filter for integer coordinates.
    3. Calculate incidence count for each candidate point.
    4. Select the top n_points.
    5. If we don't have enough, add random points or points on single lines.
    """
    n_lines = len(lines)
    candidate_points = {} # point -> count
    
    # 1. Intersections of pairs
    for i in range(n_lines):
        A1, B1, C1 = lines[i]
        for j in range(i + 1, n_lines):
            A2, B2, C2 = lines[j]
            
            # A1 x + B1 y = -C1
            # A2 x + B2 y = -C2
            det = A1 * B2 - A2 * B1
            if det == 0:
                continue # Parallel or identical
            
            # x = (B1*C2 - B2*C1) / det  ... wait
            # Cramer's rule:
            # x = (B1*(-C2) - B2*(-C1)) / det = (B2*C1 - B1*C2) / det
            # y = (A1*(-C2) - A2*(-C1)) / det = (A2*C1 - A1*C2) / det
            
            num_x = B2 * C1 - B1 * C2
            num_y = A2 * C1 - A1 * C2
            
            if num_x % det != 0 or num_y % det != 0:
                continue # Not integer
            
            x = num_x // det
            y = num_y // det
            
            # Check bounds to avoid huge coordinates (optional, but good for sanity)
            # The problem doesn't restrict coordinate size, but huge numbers might cause issues?
            # int64 handles up to 9e18.
            
            pt = (x, y)
            
            # Count incidences for this point
            count = 0
            for k in range(n_lines):
                Ak, Bk, Ck = lines[k]
                if Ak * x + Bk * y + Ck == 0:
                    count += 1
            
            if pt in candidate_points:
                # Should not happen if lines are distinct, but just in case
                if count > candidate_points[pt]:
                    candidate_points[pt] = count
            else:
                candidate_points[pt] = count

    # If we have fewer than n_points candidates, we need to add more.
    # Points on a single line (not intersections) have count 1.
    # We can generate random integer points on the lines.
    if len(candidate_points) < n_points:
        # Add random points on lines
        for _ in range(1000):
            line_idx = random.randint(0, n_lines - 1)
            A, B, C = lines[line_idx]
            # Generate a random integer point on the line
            # A x + B y = -C
            if A == 0:
                # B y = -C
                if B != 0 and -C % B == 0:
                    y = -C // B
                    x = random.randint(-100, 100)
                    pt = (x, y)
                    if pt not in candidate_points:
                        # Recalculate count (might be on other lines too)
                        count = 0
                        for k in range(n_lines):
                            Ak, Bk, Ck = lines[k]
                            if Ak * x + Bk * y + Ck == 0:
                                count += 1
                        candidate_points[pt] = count
            elif B == 0:
                # A x = -C
                if A != 0 and -C % A == 0:
                    x = -C // A
                    y = random.randint(-100, 100)
                    pt = (x, y)
                    if pt not in candidate_points:
                        count = 0
                        for k in range(n_lines):
                            Ak, Bk, Ck = lines[k]
                            if Ak * x + Bk * y + Ck == 0:
                                count += 1
                        candidate_points[pt] = count
            else:
                # Parametric form
                # Let x be a parameter.
                # A x + B y = -C => y = (-C - A x) / B
                # We need (-C - A x) % B == 0
                # A x ≡ -C (mod B)
                # Find a solution for x using extended Euclidean?
                # Simpler: just try random x in a range
                found = False
                for _ in range(20):
                    x = random.randint(-50, 50)
                    rem = (-C - A * x) % B
                    if rem == 0:
                        y = (-C - A * x) // B
                        pt = (x, y)
                        if pt not in candidate_points:
                            count = 0
                            for k in range(n_lines):
                                Ak, Bk, Ck = lines[k]
                                if Ak * x + Bk * y + Ck == 0:
                                    count += 1
                            candidate_points[pt] = count
                            found = True
                            break
                if not found:
                    continue

    # Sort points by count descending
    sorted_points = sorted(candidate_points.items(), key=lambda x: x[1], reverse=True)
    
    # Select top n_points
    selected = [pt for pt, _ in sorted_points[:n_points]]
    
    # If still not enough (should be rare), pad with random distinct points
    if len(selected) < n_points:
        used = set(selected)
        while len(selected) < n_points:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                selected.append((x, y))
                used.add((x, y))
                
    return np.array(selected, dtype=np.int64)

def _evaluate_line_set(lines):
    """
    Evaluate a set of 27 lines by finding the best 54 points and counting incidences.
    """
    # Ensure lines are normalized and unique
    unique_lines = set()
    for l in lines:
        nl = _normalize_line(l[0], l[1], l[2])
        if nl != (0, 0, 0):
            unique_lines.add(nl)
            
    if len(unique_lines) < 27:
        # Pad with random lines if we have duplicates
        while len(unique_lines) < 27:
            A = random.randint(-10, 10)
            B = random.randint(-10, 10)
            C = random.randint(-50, 50)
            nl = _normalize_line(A, B, C)
            if nl != (0, 0, 0):
                unique_lines.add(nl)
                
    lines_arr = np.array(list(unique_lines)[:27], dtype=np.int64)
    
    # Get best points
    points = _get_best_points_for_lines(lines_arr, 54)
    
    # Count incidences
    score = _count_incidences(points, lines_arr)
    
    return score, points, lines_arr

def _mutate_lines(lines, mutation_rate=0.1):
    """
    Mutate a set of lines.
    """
    new_lines = []
    for l in lines:
        if random.random() < mutation_rate:
            # Random new line
            A = random.randint(-10, 10)
            B = random.randint(-10, 10)
            C = random.randint(-50, 50)
            nl = _normalize_line(A, B, C)
            if nl == (0, 0, 0):
                nl = _normalize_line(1, 0, 0)
            new_lines.append(nl)
        else:
            # Perturb coefficients
            A, B, C = l
            dA = random.randint(-2, 2)
            dB = random.randint(-2, 2)
            dC = random.randint(-5, 5)
            nl = _normalize_line(A + dA, B + dB, C + dC)
            if nl == (0, 0, 0):
                nl = _normalize_line(A, B, C) # revert
            new_lines.append(nl)
            
    return new_lines

def _crossover(lines1, lines2):
    """
    Crossover two sets of lines.
    """
    set1 = set(lines1)
    set2 = set(lines2)
    
    # Randomly pick 13 from each
    l1_list = list(set1)
    l2_list = list(set2)
    random.shuffle(l1_list)
    random.shuffle(l2_list)
    
    child = l1_list[:13] + l2_list[:13]
    
    # Ensure 27 unique lines
    unique = set(child)
    while len(unique) < 27:
        A = random.randint(-10, 10)
        B = random.randint(-10, 10)
        C = random.randint(-50, 50)
        nl = _normalize_line(A, B, C)
        if nl != (0, 0, 0):
            unique.add(nl)
            
    return list(unique)[:27]

def _evolutionary_search(max_time=580):
    """
    Evolve the set of 27 lines to maximize incidences with the best 54 points.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    # Initial population
    pop_size = 20
    population = []
    
    # Seed 1: Grid-like lines (Verticals, Horizontals, Diagonals)
    # 6 verticals, 9 horizontals, 12 diagonals?
    seed_lines = []
    for c in range(1, 7):
        seed_lines.append(_normalize_line(1, 0, -c))
    for c in range(1, 10):
        seed_lines.append(_normalize_line(0, 1, -c))
    # Diagonals
    for c in range(-5, 6):
        seed_lines.append(_normalize_line(1, 1, -c))
    # Pad to 27
    while len(seed_lines) < 27:
        seed_lines.append(_normalize_line(1, -1, random.randint(-10, 10)))
        
    population.append(seed_lines[:27])
    
    # Other seeds: Random
    for _ in range(pop_size - 1):
        lines = []
        for _ in range(27):
            A = random.randint(-5, 5)
            B = random.randint(-5, 5)
            C = random.randint(-20, 20)
            nl = _normalize_line(A, B, C)
            if nl == (0, 0, 0):
                nl = _normalize_line(1, 0, random.randint(-10, 10))
            lines.append(nl)
        population.append(lines)
        
    # Evaluate initial population
    scores = []
    best_points_per_line = []
    for lines in population:
        score, points, lines_arr = _evaluate_line_set(lines)
        scores.append(score)
        best_points_per_line.append((points, lines_arr))
        
    best_idx = np.argmax(scores)
    global_best_score = scores[best_idx]
    global_best_points = best_points_per_line[best_idx][0]
    global_best_lines = best_points_per_line[best_idx][1]
    
    T = 10.0
    cooling = 0.95
    
    while time.time() < deadline - 10:
        new_population = []
        new_scores = []
        new_points = []
        
        # Elitism: Keep the best
        new_population.append(population[best_idx])
        new_scores.append(scores[best_idx])
        new_points.append(best_points_per_line[best_idx])
        
        # Fill the rest
        while len(new_population) < pop_size:
            # Selection
            # Tournament selection
            i1 = random.randint(0, len(population) - 1)
            i2 = random.randint(0, len(population) - 1)
            if scores[i1] > scores[i2]:
                parent1 = population[i1]
            else:
                parent1 = population[i2]
                
            i3 = random.randint(0, len(population) - 1)
            i4 = random.randint(0, len(population) - 1)
            if scores[i3] > scores[i4]:
                parent2 = population[i3]
            else:
                parent2 = population[i4]
                
            # Crossover or Mutation
            if random.random() < 0.5:
                child_lines = _crossover(parent1, parent2)
            else:
                child_lines = _mutate_lines(parent1, mutation_rate=0.2)
                
            # Evaluate
            score, points, lines_arr = _evaluate_line_set(child_lines)
            
            # Acceptance (Simulated Annealing style for population update?)
            # Just keep if it's good, or with probability if bad
            if new_scores:
                current_best_in_new = max(new_scores)
                if score > current_best_in_new:
                    new_population.append(child_lines)
                    new_scores.append(score)
                    new_points.append((points, lines_arr))
                else:
                    # Accept with probability exp((score - current_best)/T)
                    if random.random() < np.exp((score - current_best_in_new) / T):
                        new_population.append(child_lines)
                        new_scores.append(score)
                        new_points.append((points, lines_arr))
            else:
                new_population.append(child_lines)
                new_scores.append(score)
                new_points.append((points, lines_arr))
                
        # Update population
        population = new_population
        scores = new_scores
        best_points_per_line = new_points
        
        # Update global best
        current_best_idx = np.argmax(scores)
        if scores[current_best_idx] > global_best_score:
            global_best_score = scores[current_best_idx]
            global_best_points = best_points_per_line[current_best_idx][0]
            global_best_lines = best_points_per_line[current_best_idx][1]
            
        T *= cooling
        
    return global_best_points, global_best_lines, global_best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses Evolutionary Strategy on the set of Lines.
    """
    start_time = time.time()
    timeout = 590  # Safety margin
    
    try:
        best_points, best_lines, best_score = _evolutionary_search(max_time=timeout - 10)
    except Exception:
        # Fallback to simple grid if something goes wrong
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Find lines for fallback
        lines_list = []
        for c in range(1, 7):
            lines_list.append(_normalize_line(1, 0, -c))
        for c in range(1, 10):
            lines_list.append(_normalize_line(0, 1, -c))
        for c in range(-5, 6):
            lines_list.append(_normalize_line(1, 1, -c))
        while len(lines_list) < 27:
            lines_list.append(_normalize_line(1, -1, random.randint(-10, 10)))
        best_lines = np.array(lines_list[:27], dtype=np.int64)
        
    # Final validation and formatting
    points_out = best_points.astype(np.int64)
    lines_out = best_lines.astype(np.int64)
    
    # Ensure shapes
    if points_out.shape != (54, 2):
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    if lines_out.shape != (27, 3):
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    # Check for duplicates in points
    pts_set = set([tuple(p) for p in points_out])
    if len(pts_set) < 54:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
        
    # Check for invalid lines
    for l in lines_out:
        if l[0] == 0 and l[1] == 0:
            xs = np.arange(1, 7)
            ys = np.arange(1, 10)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
            lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)
            break
            
    # Check for duplicate lines
    lines_set = set([tuple(l) for l in lines_out])
    if len(lines_set) < 27:
        xs = np.arange(1, 7)
        ys = np.arange(1, 10)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_out = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        lines_out = np.array(_get_top_lines(points_out, 27), dtype=np.int64)

    return points_out, lines_out

# Helper for fallback (defined in current program, keep it)
def _get_top_lines(points, k=27):
    """
    Find the k lines with the most incidences among all lines defined by pairs of points
    and some heuristic lines.
    """
    n = len(points)
    
    # Candidate lines
    candidates = set()
    
    # Lines from pairs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            key = _normalize_line(A, B, C)
            if key != (0, 0, 0):
                candidates.add(key)
                
    # Add grid lines
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    for c in range(min_x - 2, max_x + 3):
        candidates.add(_normalize_line(1, 0, -c))
        candidates.add(_normalize_line(0, 1, -c))
    for c in range(min_x + min_y - 4, max_x + max_y + 5):
        candidates.add(_normalize_line(1, 1, -c))
        candidates.add(_normalize_line(1, -1, c))
        
    # Count incidences
    counts = {}
    for line in candidates:
        A, B, C = line
        vals = A * points[:, 0] + B * points[:, 1] + C
        counts[line] = np.sum(vals == 0)
        
    # Sort and take top k
    sorted_lines = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_keys = [key for key, _ in sorted_lines[:k]]
    
    # Pad if necessary
    if len(top_keys) < k:
        existing = set(top_keys)
        # Add random lines
        while len(top_keys) < k:
            A = random.randint(-5, 5)
            B = random.randint(-5, 5)
            C = random.randint(-50, 50)
            key = _normalize_line(A, B, C)
            if key not in existing and key != (0,0,0):
                top_keys.append(key)
                existing.add(key)
                
    return top_keys

# EVOLVE-BLOCK-END
