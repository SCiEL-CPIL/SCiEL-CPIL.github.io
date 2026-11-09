import numpy as np
import time
import random
import math
from math import gcd

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = lines[0:1, :]
    B = lines[1:2, :]
    C = lines[2:3, :]
    vals = px * A + py * B + C
    return np.sum(vals == 0)

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return None
    g = gcd(abs(A), abs(B))
    g = gcd(g, abs(C))
    if g > 1:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _generate_grid_points(rows, cols):
    xs = np.arange(1, cols + 1)
    ys = np.arange(1, rows + 1)
    px, py = np.meshgrid(xs, ys, indexing="xy")
    p = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    return p

def _generate_initial_lines(points, num_lines=27):
    """Generate a diverse set of initial lines, prioritizing high incidence."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    candidates = {}
    
    # 1. Lines through pairs of points
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n):
            xj, yj = x[j], y[j]
            A = yi - yj
            B = xj - xi
            C = xi * yj - xj * yi
            key = _normalize_line(A, B, C)
            if key is not None:
                candidates[key] = None
                
    # 2. Add some random lines to ensure diversity
    # Random lines might not pass through any points initially, but could be part of a better global config?
    # Actually, for integer points, random integer lines are unlikely to hit many points.
    # Better to add lines that are "near" grid lines.
    # Let's stick to pair lines for now, but ensure we have enough.
    
    if len(candidates) < num_lines:
        # Pad with arbitrary lines
        for i in range(num_lines - len(candidates)):
            key = _normalize_line(1, i, 0)
            if key not in candidates:
                candidates[key] = None
    
    # Score all candidates
    scored = []
    for key in candidates:
        A, B, C = key
        vals = A * x + B * y + C
        count = np.sum(vals == 0)
        scored.append((count, key))
        
    # Sort by count descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Take top num_lines
    best_lines = [k for _, k in scored[:num_lines]]
    
    # If we still don't have enough (should not happen), pad
    while len(best_lines) < num_lines:
        best_lines.append((1, 0, 0)) # Placeholder, will be replaced in SA
        
    return np.array(best_lines, dtype=np.int64)

def _get_candidate_lines(points, max_candidates=500):
    """
    Generate a pool of candidate lines to swap in during SA.
    Includes all pair lines and some random perturbations.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    candidates = set()
    
    # All pair lines
    for i in range(n):
        xi, yi = x[i], y[i]
        for j in range(i + 1, n):
            xj, yj = x[j], y[j]
            A = yi - yj
            B = xj - xi
            C = xi * yj - xj * yi
            key = _normalize_line(A, B, C)
            if key is not None:
                candidates.add(key)
                
    # If we have too many, sample them?
    # 54 points -> ~1400 lines. This is small enough to keep all.
    
    return list(candidates)

def _simulated_annealing(points, initial_lines, time_limit):
    """
    Optimize the set of lines for fixed points using Simulated Annealing.
    """
    current_lines = initial_lines.copy()
    current_score = _incidence_count(points, current_lines)
    
    # Generate candidate pool
    candidate_pool = _get_candidate_lines(points)
    if not candidate_pool:
        return current_lines, current_score
        
    # Map line tuple to index for quick lookup?
    # Or just keep a list of tuples.
    
    best_lines = current_lines.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 100.0 # Initial temperature
    cooling_rate = 0.995
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Pre-compute line values for current lines?
    # We will update incrementally or recompute if fast enough.
    # n=54, m=27. Recompute is 54*27 ~ 1458 ops. Very fast.
    
    while time.time() - start_time < time_limit:
        if T < 1e-4:
            break
            
        # Pick a random line index to replace
        idx = random.randint(0, len(current_lines) - 1)
        
        # Pick a random candidate line from the pool
        # Ensure it's not the same as the current one
        new_line_key = random.choice(candidate_pool)
        if tuple(current_lines[idx]) == new_line_key:
            continue
            
        # Create new lines array
        new_lines = current_lines.copy()
        new_lines[idx] = new_line_key
        
        # Check for duplicate lines in new_lines
        # If duplicate, skip (or penalize). The task forbids duplicate lines.
        # We need to ensure the 27 lines are distinct.
        # Since candidate_pool has unique keys, and we replace one,
        # we just need to check if new_line_key is already in current_lines (at a different index).
        is_dup = False
        for k in range(len(new_lines)):
            if k == idx: continue
            if tuple(new_lines[k]) == new_line_key:
                is_dup = True
                break
        if is_dup:
            continue
            
        new_score = _incidence_count(points, new_lines)
        
        delta = new_score - current_score
        
        if delta > 0:
            current_lines = new_lines
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_lines = new_lines.copy()
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            prob = math.exp(delta / T)
            if random.random() < prob:
                current_lines = new_lines
                current_score = new_score
                
        T *= cooling_rate
        
    return best_lines, best_score

def _simulated_annealing_points_and_lines(initial_points, initial_lines, time_limit):
    """
    Simulated Annealing on both points and lines.
    Moves:
    1. Change one line to a random candidate line.
    2. Move one point to a random nearby integer coordinate.
    """
    current_points = initial_points.copy()
    current_lines = initial_lines.copy()
    current_score = _incidence_count(current_points, current_lines)
    
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 50.0
    cooling_rate = 0.99
    
    # Candidate lines need to be regenerated if points change?
    # Yes, pair lines depend on points.
    # To keep it simple, we will only do line swaps in the first phase,
    # and point moves in the second phase, or mixed.
    # Given 600s budget, we can afford some recomputation.
    
    # Let's do a hybrid:
    # Phase 1: Optimize lines for fixed points (fast)
    # Phase 2: Optimize points and lines together (slower)
    
    # Phase 1
    t1_limit = time_limit * 0.5
    current_lines, current_score = _simulated_annealing(current_points, current_lines, t1_limit)
    best_lines = current_lines.copy()
    best_score = current_score
    best_points = current_points.copy()
    
    # Phase 2
    t2_limit = time_limit * 0.4
    start_time2 = time.time()
    
    # We need a way to generate candidate lines dynamically or use a large static pool?
    # If points move, the "pair lines" change.
    # Generating all pair lines for 54 points is ~1400 lines.
    # We can regenerate the pool every K steps.
    
    pool_refresh_interval = 50
    steps = 0
    
    while time.time() - start_time2 < t2_limit:
        if T < 1e-4:
            break
            
        steps += 1
        if steps % pool_refresh_interval == 0:
            candidate_pool = _get_candidate_lines(current_points)
            if not candidate_pool:
                break
        else:
            # Use existing pool? We need to define it.
            if 'candidate_pool' not in locals() or candidate_pool is None:
                candidate_pool = _get_candidate_lines(current_points)
                if not candidate_pool:
                    break
                    
        move_type = random.random()
        
        if move_type < 0.7:
            # Line move
            idx = random.randint(0, len(current_lines) - 1)
            new_line_key = random.choice(candidate_pool)
            if tuple(current_lines[idx]) == new_line_key:
                continue
                
            new_lines = current_lines.copy()
            new_lines[idx] = new_line_key
            
            # Check dup
            is_dup = False
            for k in range(len(new_lines)):
                if k == idx: continue
                if tuple(new_lines[k]) == new_line_key:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            new_score = _incidence_count(current_points, new_lines)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_lines = new_lines
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_lines = new_lines.copy()
                    best_points = current_points.copy()
                    
        else:
            # Point move
            idx = random.randint(0, len(current_points) - 1)
            # Move to a random nearby point
            # Avoid collisions with other points
            new_p = current_points[idx].copy()
            for _ in range(10): # Try 10 times to find a unique spot
                dx = random.randint(-5, 5)
                dy = random.randint(-5, 5)
                candidate_p = (new_p[0] + dx, new_p[1] + dy)
                # Check if candidate_p is in current_points (excluding idx)
                is_unique = True
                for k in range(len(current_points)):
                    if k == idx: continue
                    if current_points[k][0] == candidate_p[0] and current_points[k][1] == candidate_p[1]:
                        is_unique = False
                        break
                if is_unique:
                    new_p = np.array(candidate_p, dtype=np.int64)
                    break
            else:
                continue # Couldn't find a unique spot
            
            new_points = current_points.copy()
            new_points[idx] = new_p
            
            new_score = _incidence_count(new_points, current_lines)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_lines = current_lines.copy()
                    best_points = new_points.copy()
                    
        T *= cooling_rate
        
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    # Candidate point configurations
    # 1. 6x9 Grid (Proven good)
    p_6x9 = _generate_grid_points(9, 6) # 9 rows, 6 cols -> 54 points
    # 2. 3x18 Grid
    p_3x18 = _generate_grid_points(18, 3)
    # 3. 2x27 Grid
    p_2x27 = _generate_grid_points(27, 2)
    # 4. 9x6 Grid
    p_9x6 = _generate_grid_points(6, 9)
    
    candidates = [p_6x9, p_3x18, p_2x27, p_9x6]
    
    best_global_points = None
    best_global_lines = None
    best_global_score = -1
    
    # Time allocation per candidate
    # We have ~590s. 4 candidates. ~140s each?
    # But SA might converge faster.
    # Let's run SA for each candidate.
    
    for p in candidates:
        if time.time() + safety_margin > start_time + deadline:
            break
            
        # Initialize lines
        init_lines = _generate_initial_lines(p, 27)
        
        # Run SA
        try:
            res_p, res_l, res_s = _simulated_annealing_points_and_lines(p, init_lines, 100.0)
            
            # Validate result
            if res_p.shape != (54, 2) or res_l.shape != (27, 3):
                continue
            if len(np.unique(res_p, axis=0)) != 54:
                continue
            
            # Check line validity and duplicates
            seen_lines = set()
            valid = True
            for i in range(27):
                A, B, C = res_l[i]
                if A == 0 and B == 0:
                    valid = False
                    break
                key = _normalize_line(A, B, C)
                if key in seen_lines:
                    valid = False
                    break
                seen_lines.add(key)
            if not valid:
                continue
                
            if res_s > best_global_score:
                best_global_score = res_s
                best_global_points = res_p
                best_global_lines = res_l
                
        except Exception as e:
            continue
            
    if best_global_points is None:
        # Fallback
        best_global_points = p_6x9
        best_global_lines = _generate_initial_lines(p_6x9, 27)
        
    return best_global_points, best_global_lines
