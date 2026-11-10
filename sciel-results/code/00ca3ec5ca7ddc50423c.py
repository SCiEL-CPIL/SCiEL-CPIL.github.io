import numpy as np
import time
import random
from math import gcd
from itertools import combinations, product

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    
    # Handle zero C case for gcd
    abs_g = gcd(abs(A), abs(B))
    if C != 0:
        abs_g = gcd(abs_g, abs(C))
    if abs_g == 0:
        abs_g = 1
        
    A //= abs_g
    B //= abs_g
    C //= abs_g
    
    # Make first non-zero entry positive
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def structured_search(candidate_points, n_lines=27, time_limit=60):
    """
    Search for the best 27 lines from the families:
    Vertical: (1, 0, -x)
    Horizontal: (0, 1, -y)
    Diag+: (1, -1, -c)  => x - y = c
    Diag-: (1, 1, -c)   => x + y = c
    
    We fix the counts to 9 V, 9 H, 5 D+, 4 D- based on the insight that this balance
    allows for 4-incidence points (intersection of V, H, D+, D-).
    """
    start_time = time.time()
    
    # Determine range of intercepts to consider
    # Points are roughly in [-50, 50]. 
    # V/H intercepts: -50 to 50
    # D+ (x-y=c): range -100 to 100
    # D- (x+y=c): range -100 to 100
    
    # To speed up, we can precompute incidence vectors for each candidate line in these families.
    # But 100*4 = 400 lines. 400 * 54 points is small.
    
    # Let's define the candidate pools for each family
    v_intercepts = list(range(-40, 41))
    h_intercepts = list(range(-40, 41))
    d_plus_intercepts = list(range(-80, 81))
    d_minus_intercepts = list(range(-80, 81))
    
    # Precompute incidence counts for each candidate line against ALL candidate points
    # This allows us to quickly evaluate the sum of counts for a selected set of lines,
    # BUT we must be careful: a point is only counted once per line.
    # The objective is sum over lines of (number of points on that line).
    # Since lines are distinct, a point can be on multiple lines.
    # So Total Incidences = Sum_{l in L} |P_l|.
    # This is additive! We don't need to worry about double counting points across lines for the *sum*.
    # Wait, the task defines I as number of (point, line) pairs.
    # Yes, I = sum_{l} count(l).
    # So we just need to pick the 27 lines with the highest individual counts, subject to the constraint
    # that they are distinct and from the allowed families with specific counts per family?
    # No, the constraint is just that we pick 27 distinct lines.
    # However, the "insight" suggests that a specific *combination* is better because it allows points to be on 4 lines.
    # But since I is the sum of counts, picking the top 27 lines by individual count *should* be optimal
    # IF the lines are independent. But they are not independent in terms of the *points* we choose.
    # The points are chosen AFTER the lines are fixed, to maximize the sum of counts.
    # Actually, `get_best_points_for_lines` picks the 54 points that maximize the sum of counts for the *fixed* lines.
    # So for a fixed set of lines L, the max I is sum of top 54 values in the array of (point, line) incidence sums?
    # No.
    # Let P be the set of 54 points.
    # I(P, L) = sum_{p in P, l in L} [p on l].
    # This is equivalent to: For each point p, let k_p be the number of lines in L that pass through p.
    # I = sum_{p in P} k_p.
    # To maximize I, we should pick the 54 points with the highest k_p.
    # So the problem reduces to: Choose 27 lines L from the candidate pool to maximize the sum of the top 54 values of k_p(L).
    
    # This is still hard if we search over all combinations.
    # But if we restrict to the 4 families, we can be smarter.
    
    # Let's generate all candidate lines from the 4 families.
    candidates = []
    for c in v_intercepts:
        candidates.append((1, 0, -c, 'V'))
    for c in h_intercepts:
        candidates.append((0, 1, -c, 'H'))
    for c in d_plus_intercepts:
        candidates.append((1, -1, -c, 'D+'))
    for c in d_minus_intercepts:
        candidates.append((1, 1, -c, 'D-'))
        
    # We want to pick 27 lines.
    # Heuristic: The best lines will have high incidence.
    # Let's calculate the incidence count for each candidate line against the *grid* of candidate points.
    # Note: The optimal points will be a subset of these.
    
    P = candidate_points.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    line_counts = np.zeros(len(candidates), dtype=int)
    for i, (A, B, C, fam) in enumerate(candidates):
        vals = A * px + B * py + C
        line_counts[i] = np.sum(vals == 0)
        
    # Sort candidates by count descending
    sorted_indices = np.argsort(line_counts)[::-1]
    
    # Greedily pick lines, ensuring we don't pick too many from one family?
    # The insight says 9V, 9H, 5D+, 4D- is good.
    # Let's try to enforce these counts.
    
    target_counts = {'V': 9, 'H': 9, 'D+': 5, 'D-': 4}
    remaining_counts = target_counts.copy()
    selected_indices = []
    
    # We iterate through sorted candidates. If a candidate's family still has quota, we pick it.
    # If not, we skip it.
    # If we run out of candidates for a family, we might need to adjust.
    
    # To be more robust, let's try a few variations of target counts.
    variations = [
        {'V': 9, 'H': 9, 'D+': 5, 'D-': 4},
        {'V': 8, 'H': 8, 'D+': 6, 'D-': 5},
        {'V': 10, 'H': 10, 'D+': 4, 'D-': 3},
        {'V': 9, 'H': 9, 'D+': 4, 'D-': 5},
        {'V': 9, 'H': 9, 'D+': 6, 'D-': 3},
    ]
    
    best_I = -1
    best_lines = None
    best_points = None
    
    for var in variations:
        if time.time() - start_time > time_limit:
            break
            
        rem = var.copy()
        sel_idx = []
        
        for idx in sorted_indices:
            if len(sel_idx) == 27:
                break
            fam = candidates[idx][3]
            if rem.get(fam, 0) > 0:
                sel_idx.append(idx)
                rem[fam] -= 1
        
        if len(sel_idx) < 27:
            # Fill with any remaining best lines
            for idx in sorted_indices:
                if idx not in sel_idx:
                    sel_idx.append(idx)
                if len(sel_idx) == 27:
                    break
                    
        lines = np.array([candidates[i][:3] for i in sel_idx], dtype=np.int64)
        
        # Check for duplicates (shouldn't happen if intercepts are unique and families distinct)
        unique_lines = []
        seen = set()
        for l in lines:
            nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
            if nl and nl not in seen:
                seen.add(nl)
                unique_lines.append(nl)
                
        if len(unique_lines) < 27:
            continue
            
        lines = np.array(unique_lines[:27], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I:
            best_I = I
            best_lines = lines
            best_points = points
            
    return best_lines, best_points, best_I

def count_incidences(points, lines):
    """
    points: (n, 2) array
    lines: (m, 3) array (A, B, C)
    Returns number of (p, l) such that A*x + B*y + C == 0
    """
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    count = np.sum(vals == 0)
    return int(count)

def get_best_points_for_lines(lines, candidate_points, k=54):
    """
    Given a set of lines and a pool of candidate points,
    find the top k points that maximize incidences.
    """
    if len(candidate_points) < k:
        return candidate_points.copy(), 0
    
    L = lines.astype(np.int64)
    P = candidate_points.astype(np.int64)
    
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    
    vals = A * px + B * py + C
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k indices
    top_indices = np.argsort(counts)[::-1][:k]
    best_points = candidate_points[top_indices]
    
    return best_points, int(counts[top_indices].max())

def generate_candidate_points(size=50, extra=2000):
    """Generate a grid of candidate points plus some random perturbations."""
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Add some random points in a slightly larger range for diversity
    rng = np.random.RandomState(123)
    rand_pts = rng.randint(-size-5, size+6, size=(extra, 2))
    
    all_pts = np.vstack([grid_pts, rand_pts])
    # Remove duplicates
    all_pts, _ = np.unique(all_pts, axis=0, return_index=True)
    return all_pts

def greedy_select_lines_from_points(points, n_lines=27, candidate_pool=None):
    """
    Greedily select lines that maximize total incidences with the given points.
    Uses a smarter approach: prioritize lines through 3+ points.
    """
    P = points.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    
    n_pts = len(points)
    candidate_lines = []
    seen_lines = set()
    
    # Generate lines from pairs
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            dx = points[j, 0] - points[i, 0]
            dy = points[j, 1] - points[i, 1]
            A = dy
            B = -dx
            C = dx * points[i, 1] - dy * points[i, 0]
            nl = normalize_line(int(A), int(B), int(C))
            if nl and nl not in seen_lines:
                seen_lines.add(nl)
                candidate_lines.append(nl)
    
    if not candidate_lines:
        for i in range(27):
            candidate_lines.append((1, 0, -i))
        return np.array(candidate_lines[:27], dtype=np.int64)
    
    # Pre-compute incidence counts for all candidate lines
    line_counts = {}
    for nl in candidate_lines:
        A, B, C = nl
        vals = A * px + B * py + C
        cnt = int(np.sum(vals == 0))
        line_counts[nl] = cnt
    
    # Sort candidates by count descending for faster greedy
    sorted_candidates = sorted(candidate_lines, key=lambda l: line_counts[l], reverse=True)
    
    # Greedy selection with marginal gain consideration
    selected_lines = []
    selected_set = set()
    
    for _ in range(n_lines):
        best_line = None
        best_score = -1
        for nl in sorted_candidates:
            if nl in selected_set:
                continue
            score = line_counts[nl]
            if score > best_score:
                best_score = score
                best_line = nl
        if best_line is None:
            break
        selected_lines.append(best_line)
        selected_set.add(best_line)
    
    # Fill if needed with random lines
    rng = random.Random(42)
    while len(selected_lines) < n_lines:
        A = rng.randint(-7, 7)
        B = rng.randint(-7, 7)
        C = rng.randint(-35, 35)
        nl = normalize_line(A, B, C)
        if nl and nl not in selected_set:
            selected_lines.append(nl)
            selected_set.add(nl)
    
    return np.array(selected_lines[:n_lines], dtype=np.int64)

def generate_initial_lines(config_id):
    """Generate diverse initial line configurations."""
    lines = []
    if config_id == 0:
        # Grid + diagonals
        for i in range(9):
            lines.append((1, 0, -i))
        for j in range(6):
            lines.append((0, 1, -j))
        for c in range(-8, 4):
            lines.append((1, -1, -c))
    elif config_id == 1:
        # Various slopes
        a_vals = [0, 1, -1, 2, -2]
        b_vals = [0, 1, -1, 2, -2, 3, -3]
        for a in a_vals:
            for b in b_vals:
                lines.append((a, -1, b))
                if len(lines) >= 27:
                    break
            if len(lines) >= 27:
                break
        while len(lines) < 27:
            lines.append((1, 0, len(lines)))
    elif config_id == 2:
        # Pencil through origin
        pairs = [(1,0), (0,1), (1,1), (1,-1), (2,1), (1,2), (3,1), (1,3), (3,2), (2,3), (3,4), (4,3), (5,1), (1,5)]
        for A, B in pairs:
            lines.append((A, -B, 0))
        for i in range(27 - len(lines)):
            lines.append((1, 1, i+1))
    elif config_id == 3:
        # Random
        random.seed(42)
        seen = set()
        while len(lines) < 27:
            A = random.randint(-5, 5)
            B = random.randint(-5, 5)
            C = random.randint(-10, 10)
            if A == 0 and B == 0:
                continue
            nl = normalize_line(A, B, C)
            if nl and nl not in seen:
                seen.add(nl)
                lines.append(nl)
                
    return np.array(lines[:27], dtype=np.int64)

def improve_configuration(lines, candidate_points, time_limit, seed=42, structured_pool=None):
    """
    Joint optimization of points and lines using simulated annealing.
    Improved: better move generation, faster evaluation, smarter restarts.
    Uses structured_pool to bias line replacements towards high-yield families.
    """
    random.seed(seed)
    current_lines = lines.copy()
    current_points, _ = get_best_points_for_lines(current_lines, candidate_points)
    current_I = count_incidences(current_points, current_lines)
    
    best_lines = current_lines.copy()
    best_points = current_points.copy()
    best_I = current_I
    
    start_time = time.time()
    T = 15.0
    cooling = 0.999
    min_T = 0.3
    
    no_improve = 0
    n_cand_pts = len(candidate_points)
    
    # Precompute a set of current line tuples for fast duplicate check
    current_line_set = set(tuple(l) for l in current_lines)
    
    while time.time() - start_time < time_limit:
        if no_improve > 300:
            T = max(T, 15.0)
            no_improve = 0
            
        move_type = random.random()
        
        if move_type < 0.30:
            # Replace a line with a line through a pair of current points
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            p1_idx, p2_idx = random.sample(range(54), 2)
            p1 = current_points[p1_idx]
            p2 = current_points[p2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            
            new_line = normalize_line(int(A), int(B), int(C))
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if new_line_tup in current_line_set:
                continue
                
            current_lines[idx] = new_line
            current_line_set.discard(tuple(old_line))
            current_line_set.add(new_line_tup)
            
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                current_line_set.discard(new_line_tup)
                current_line_set.add(tuple(old_line))
                no_improve += 1
                
        elif move_type < 0.50:
            # Replace a line with a random line from the structured pool (V, H, D+, D-)
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            if structured_pool and len(structured_pool) > 0:
                # Pick a random line from the structured pool
                new_line = structured_pool[random.randint(0, len(structured_pool)-1)].copy()
            else:
                # Fallback to random small coefficients
                A = random.randint(-8, 8)
                B = random.randint(-8, 8)
                C = random.randint(-40, 40)
                new_line = normalize_line(A, B, C)
                if new_line is None:
                    continue
                new_line = np.array(new_line, dtype=np.int64)
            
            new_line_tup = tuple(new_line)
            if new_line_tup in current_line_set:
                continue
                
            current_lines[idx] = new_line
            current_line_set.discard(tuple(old_line))
            current_line_set.add(new_line_tup)
            
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                current_points = new_points
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_lines[idx] = old_line
                current_line_set.discard(new_line_tup)
                current_line_set.add(tuple(old_line))
                no_improve += 1
                
        elif move_type < 0.80:
            # Swap a point - try multiple candidates and pick best
            pt_idx = random.randint(0, 53)
            old_pt = current_points[pt_idx].copy()
            
            best_new_I = current_I
            best_new_pt = None
            for _ in range(5):
                cand_idx = random.randint(0, n_cand_pts-1)
                new_pt = candidate_points[cand_idx].copy()
                
                if any(np.array_equal(current_points[i], new_pt) for i in range(54)):
                    continue
                
                current_points[pt_idx] = new_pt
                trial_I = count_incidences(current_points, current_lines)
                if trial_I > best_new_I:
                    best_new_I = trial_I
                    best_new_pt = new_pt.copy()
                current_points[pt_idx] = old_pt
            
            if best_new_pt is not None:
                current_points[pt_idx] = best_new_pt
                new_I = best_new_I
            else:
                new_I = current_I
            
            delta_I = new_I - current_I
            
            if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / T)):
                current_I = new_I
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
        else:
            # Rebuild lines greedily from current points
            new_lines = greedy_select_lines_from_points(current_points, 27)
            
            unique_lines = []
            seen = set()
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and tuple(nl) not in seen:
                    seen.add(tuple(nl))
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = get_best_points_for_lines(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if delta_I > 0 or (T > 0 and random.random() < np.exp(delta_I / (T * 0.5))):
                    current_I = new_I
                    current_lines = new_lines
                    current_points = new_points
                    current_line_set = set(tuple(l) for l in current_lines)
                    if current_I > best_I:
                        best_I = current_I
                        best_lines = current_lines.copy()
                        best_points = current_points.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
        T *= cooling
        T = max(T, min_T)
        
    return best_lines, best_points, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 580
    
    # Generate candidate points - larger grid for more diversity
    candidate_points = generate_candidate_points(size=50, extra=2000)
    
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # 1. Run Structured Search first to find a strong baseline
    if time.time() < deadline - 100:
        struct_lines, struct_points, struct_I = structured_search(candidate_points, time_limit=60)
        if struct_lines is not None and struct_I > best_I_global:
            best_I_global = struct_I
            best_points_global = struct_points.copy()
            best_lines_global = struct_lines.copy()
            
            # Refine the structured result with local search
            if time.time() < deadline - 50:
                # Create a pool of structured lines for the local search to sample from
                # We reuse the candidates from structured_search logic implicitly by just passing the best lines found
                # Actually, let's generate a pool of V/H/D lines for the local search to use as "random" replacements
                pool_lines = []
                for c in range(-40, 41):
                    pool_lines.append(np.array([1, 0, -c], dtype=np.int64))
                    pool_lines.append(np.array([0, 1, -c], dtype=np.int64))
                for c in range(-80, 81):
                    pool_lines.append(np.array([1, -1, -c], dtype=np.int64))
                    pool_lines.append(np.array([1, 1, -c], dtype=np.int64))
                
                bl, bp, bi = improve_configuration(struct_lines, candidate_points, 
                                                   time_limit=min(120, deadline - time.time() - 40), 
                                                   seed=100, 
                                                   structured_pool=pool_lines)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()

    # 2. Try other initial configurations if time permits
    for config_id in range(4):
        if time.time() > deadline - 40:
            break
            
        lines = generate_initial_lines(config_id)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        I = count_incidences(points, lines)
        
        if I > best_I_global:
            best_I_global = I
            best_points_global = points.copy()
            best_lines_global = lines.copy()
            
        if time.time() < deadline - 50:
            remaining_time = min(100, deadline - time.time() - 40)
            if remaining_time > 15:
                bl, bp, bi = improve_configuration(lines, candidate_points, 
                                                   time_limit=remaining_time, 
                                                   seed=200 + config_id * 17)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Final refinement: do one more round of greedy line selection on best points
    if best_points_global is not None and time.time() < deadline - 30:
        refined_lines = greedy_select_lines_from_points(best_points_global, 27)
        unique_refined = []
        seen = set()
        for l in refined_lines:
            nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
            if nl and tuple(nl) not in seen:
                seen.add(tuple(nl))
                unique_refined.append(nl)
        if len(unique_refined) >= 27:
            refined_lines = np.array(unique_refined[:27], dtype=np.int64)
            refined_points, _ = get_best_points_for_lines(refined_lines, candidate_points)
            refined_I = count_incidences(refined_points, refined_lines)
            if refined_I > best_I_global:
                best_I_global = refined_I
                best_points_global = refined_points.copy()
                best_lines_global = refined_lines.copy()
    
    # Validation
    if best_points_global is None or best_lines_global is None:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
        
    # Ensure unique points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_points_global = unique_points[indices]
        
    # Ensure unique lines
    unique_lines = []
    seen = set()
    for line in best_lines_global:
        n_line = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if n_line and tuple(n_line) not in seen:
            seen.add(tuple(n_line))
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(6)] + [(1, -1, -c) for c in range(-8, 4)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
        
    return best_points_global, best_lines_global

# EVOLVE-BLOCK-END
