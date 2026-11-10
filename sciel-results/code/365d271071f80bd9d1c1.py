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
    
    # Compute incidence count for each candidate point
    n_pts = len(P)
    n_lines = len(L)
    
    # Process in chunks to avoid memory issues
    chunk_size = 5000
    all_counts = np.zeros(n_pts, dtype=np.int32)
    
    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        P_chunk = P[start:end]
        
        px = P_chunk[:, 0].reshape(-1, 1)
        py = P_chunk[:, 1].reshape(-1, 1)
        
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        
        vals = A * px + B * py + C
        counts = np.sum(vals == 0, axis=1)
        all_counts[start:end] = counts
    
    # Get top k indices (stable sort for reproducibility)
    top_indices = np.argsort(-all_counts, kind='stable')[:k]
    best_points = P[top_indices]
    
    return best_points, int(all_counts[top_indices].max())

def generate_candidate_points(size=50):
    """Generate a grid of candidate points."""
    xs = np.arange(-size, size + 1)
    ys = np.arange(-size, size + 1)
    XX, YY = np.meshgrid(xs, ys)
    return np.column_stack([XX.ravel(), YY.ravel()])

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

def refine_configuration(lines, candidate_points, time_limit, seed=42):
    """
    Joint optimization of points and lines using simulated annealing.
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
    cooling = 0.9992
    min_T = 1.0
    
    no_improve = 0
    n_cand_pts = len(candidate_points)
    
    def compute_point_counts(points, lines):
        """Compute how many lines each point lies on."""
        P = points.astype(np.int64)
        L = lines.astype(np.int64)
        px = P[:, 0].reshape(-1, 1)
        py = P[:, 1].reshape(-1, 1)
        A = L[:, 0].reshape(1, -1)
        B = L[:, 1].reshape(1, -1)
        C = L[:, 2].reshape(1, -1)
        vals = A * px + B * py + C
        return np.sum(vals == 0, axis=1)
    
    def try_accept(delta_I, T):
        """Accept/reject based on SA criterion."""
        if delta_I > 0:
            return True
        if T <= 0:
            return False
        if delta_I < -50:  # Very bad moves always rejected
            return False
        return random.random() < np.exp(delta_I / T)
    
    while time.time() - start_time < time_limit:
        if no_improve > 600:
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
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if try_accept(delta_I, T):
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
                no_improve += 1
                
        elif move_type < 0.50:
            # Replace a line with a line through a random pair from candidate pool
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            c1_idx, c2_idx = random.sample(range(n_cand_pts), 2)
            p1 = candidate_points[c1_idx]
            p2 = candidate_points[c2_idx]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * p1[1] - dy * p1[0]
            
            new_line = normalize_line(int(A), int(B), int(C))
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if try_accept(delta_I, T):
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
                no_improve += 1
                
        elif move_type < 0.65:
            # Replace a line with a random line from small coefficients
            idx = random.randint(0, 26)
            old_line = current_lines[idx].copy()
            
            # Bias toward low-slope lines that create grid-like structures
            slope_choice = random.random()
            if slope_choice < 0.25:
                A, B = 1, 0  # vertical
                C = random.randint(-20, 20)
            elif slope_choice < 0.50:
                A, B = 0, 1  # horizontal
                C = random.randint(-20, 20)
            elif slope_choice < 0.75:
                A, B = 1, -1  # diagonal slope 1
                C = random.randint(-20, 20)
            else:
                A, B = 1, 1  # diagonal slope -1
                C = random.randint(-20, 20)
            
            new_line = normalize_line(A, B, C)
            if new_line is None:
                continue
            
            new_line_tup = tuple(new_line)
            if any(tuple(l) == new_line_tup for l in current_lines):
                continue
                
            current_lines[idx] = new_line
            new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
            new_I = count_incidences(new_points, current_lines)
            
            delta_I = new_I - current_I
            
            if try_accept(delta_I, T):
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
                no_improve += 1
                
        elif move_type < 0.85:
            # Swap a point with a candidate that has high potential
            pt_idx = random.randint(0, 53)
            old_pt = current_points[pt_idx].copy()
            
            # Find candidates not already in use
            used_set = set(map(tuple, current_points))
            candidates = [c for c in candidate_points if tuple(c) not in used_set]
            
            if not candidates:
                continue
                
            # Weight toward candidates that lie on more current lines
            cand_counts = compute_point_counts(np.array(candidates), current_lines)
            # Soft-max sampling
            probs = np.exp(cand_counts / 5.0)
            probs = probs / probs.sum()
            cand_idx = random.choices(range(len(candidates)), weights=probs)[0]
            new_pt = candidates[cand_idx].copy()
            
            current_points[pt_idx] = new_pt
            new_I = count_incidences(current_points, current_lines)
            
            delta_I = new_I - current_I
            
            if try_accept(delta_I, T):
                current_I = new_I
                if current_I > best_I:
                    best_I = current_I
                    best_lines = current_lines.copy()
                    best_points = current_points.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                current_points[pt_idx] = old_pt
                no_improve += 1
                
        else:
            # Use MWIS-inspired greedy line selection
            new_lines = greedy_mwis_line_selection(current_points, candidate_points, 27)
            
            unique_lines = []
            seen = set()
            for l in new_lines:
                nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
                if nl and nl not in seen:
                    seen.add(nl)
                    unique_lines.append(nl)
            
            if len(unique_lines) >= 27:
                new_lines = np.array(unique_lines[:27], dtype=np.int64)
                new_points, _ = get_best_points_for_lines(new_lines, candidate_points)
                new_I = count_incidences(new_points, new_lines)
                
                delta_I = new_I - current_I
                
                if try_accept(delta_I, T * 0.5):
                    current_I = new_I
                    current_lines = new_lines
                    current_points = new_points
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

def greedy_mwis_line_selection(points, candidate_points, n_lines=27):
    """
    MWIS-inspired greedy: select lines by marginal gain.
    """
    P = points.astype(np.int64)
    CP = candidate_points.astype(np.int64)
    
    n_pts = len(P)
    n_cand = len(CP)
    
    seen_lines = set()
    all_lines = []
    
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            dx = P[j, 0] - P[i, 0]
            dy = P[j, 1] - P[i, 1]
            A = dy
            B = -dx
            C = dx * P[i, 1] - dy * P[i, 0]
            nl = normalize_line(int(A), int(B), int(C))
            if nl and nl not in seen_lines:
                seen_lines.add(nl)
                all_lines.append(nl)
    
    if len(all_lines) < n_lines:
        while len(all_lines) < n_lines:
            for c in range(-50, 50):
                nl = normalize_line(1, 0, -c)
                if nl and nl not in seen_lines:
                    seen_lines.add(nl)
                    all_lines.append(nl)
                    if len(all_lines) >= n_lines:
                        break
                nl = normalize_line(0, 1, -c)
                if nl and nl not in seen_lines:
                    seen_lines.add(nl)
                    all_lines.append(nl)
                    if len(all_lines) >= n_lines:
                        break
    
    n_lines_total = len(all_lines)
    line_masks = []
    for nl in all_lines:
        A, B, C = nl
        mask = np.zeros(n_cand, dtype=bool)
        mask[A * CP[:, 0] + B * CP[:, 1] + C == 0] = True
        line_masks.append(mask)
    
    selected_lines = []
    selected_indices = set()
    used_points_mask = np.zeros(n_cand, dtype=bool)
    
    for _ in range(n_lines):
        best_gain = -1
        best_line_idx = -1
        
        for li in range(n_lines_total):
            if li in selected_indices:
                continue
            new_mask = line_masks[li] & ~used_points_mask
            gain = int(np.sum(new_mask))
            if gain > best_gain:
                best_gain = gain
                best_line_idx = li
        
        if best_line_idx < 0:
            break
        
        selected_lines.append(all_lines[best_line_idx])
        selected_indices.add(best_line_idx)
        used_points_mask |= line_masks[best_line_idx]
    
    while len(selected_lines) < n_lines:
        for li in range(n_lines_total):
            if li not in selected_indices:
                selected_lines.append(all_lines[li])
                break
    
    return np.array(selected_lines[:n_lines], dtype=np.int64)

def polish_local_search(points, lines, candidate_points, time_limit):
    """
    Final polish pass: try individual point swaps and line perturbations.
    """
    start_time = time.time()
    current_points = points.copy()
    current_lines = lines.copy()
    current_I = count_incidences(current_points, current_lines)
    best_I = current_I
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    
    n_cand = len(candidate_points)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try swapping each point with a better candidate
        for pt_idx in range(54):
            if time.time() - start_time > time_limit:
                break
            old_pt = current_points[pt_idx].copy()
            
            # Find best replacement
            used_set = set(map(tuple, current_points))
            used_set.discard(tuple(old_pt))
            
            best_gain = 0
            best_new_pt = None
            
            # Sample a subset of candidates for speed
            sample_size = min(200, n_cand)
            sample_indices = random.sample(range(n_cand), sample_size)
            
            for ci in sample_indices:
                new_pt = candidate_points[ci]
                if tuple(new_pt) in used_set:
                    continue
                
                # Quick check: does this point lie on more lines?
                cnt = 0
                for A, B, C in current_lines:
                    if A * new_pt[0] + B * new_pt[1] + C == 0:
                        cnt += 1
                
                # Old point's count
                old_cnt = 0
                for A, B, C in current_lines:
                    if A * old_pt[0] + B * old_pt[1] + C == 0:
                        old_cnt += 1
                
                gain = cnt - old_cnt
                if gain > best_gain:
                    best_gain = gain
                    best_new_pt = new_pt.copy()
            
            if best_new_pt is not None and best_gain > 0:
                current_points[pt_idx] = best_new_pt
                new_I = count_incidences(current_points, current_lines)
                if new_I > current_I:
                    current_I = new_I
                    improved = True
                    if new_I > best_I:
                        best_I = new_I
                        best_points = current_points.copy()
                        best_lines = current_lines.copy()
        
        # Try perturbing lines slightly
        for line_idx in range(27):
            if time.time() - start_time > time_limit:
                break
            old_line = current_lines[line_idx].copy()
            A, B, C = current_lines[line_idx]
            
            # Try small perturbations
            for dA in [-1, 0, 1]:
                for dB in [-1, 0, 1]:
                    for dC in [-2, -1, 0, 1, 2]:
                        if dA == 0 and dB == 0 and dC == 0:
                            continue
                        new_A, new_B, new_C = A + dA, B + dB, C + dC
                        if new_A == 0 and new_B == 0:
                            continue
                        
                        nl = normalize_line(new_A, new_B, new_C)
                        if nl is None:
                            continue
                        
                        # Check if line already exists
                        if any(tuple(l) == tuple(nl) for l in current_lines):
                            continue
                        
                        current_lines[line_idx] = nl
                        new_points, _ = get_best_points_for_lines(current_lines, candidate_points)
                        new_I = count_incidences(new_points, current_lines)
                        
                        if new_I > current_I:
                            current_I = new_I
                            current_points = new_points
                            improved = True
                            if new_I > best_I:
                                best_I = new_I
                                best_points = current_points.copy()
                                best_lines = current_lines.copy()
                        else:
                            current_lines[line_idx] = old_line
        
        if not improved:
            break
    
    return best_lines, best_points, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 580
    
    # Generate candidate points - larger grid for more options
    candidate_points = generate_candidate_points(size=50)
    
    best_I_global = -1
    best_points_global = None
    best_lines_global = None
    
    # 1. Structured search using V/H/D+/D- families
    if time.time() < deadline - 100:
        bl, bp, bi = structured_search(candidate_points, 27, time_limit=60)
        if bl is not None and bi > best_I_global:
            best_I_global = bi
            best_points_global = bp.copy()
            best_lines_global = bl.copy()
    
    # 2. Refine the best configuration found
    if best_lines_global is not None and time.time() < deadline - 60:
        remaining_time = min(120, deadline - time.time() - 40)
        if remaining_time > 30:
            bl, bp, bi = refine_configuration(best_lines_global, candidate_points, 
                                              time_limit=remaining_time, seed=42)
            if bi > best_I_global:
                best_I_global = bi
                best_points_global = bp.copy()
                best_lines_global = bl.copy()
    
    # 3. Polish
    if best_points_global is not None and best_lines_global is not None:
        if time.time() < deadline - 30:
            polish_time = min(40, deadline - time.time() - 25)
            if polish_time > 5:
                bl, bp, bi = polish_local_search(best_points_global, best_lines_global, 
                                                  candidate_points, polish_time)
                if bi > best_I_global:
                    best_I_global = bi
                    best_points_global = bp.copy()
                    best_lines_global = bl.copy()
    
    # Validation
    if best_points_global is None or best_lines_global is None:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(9)] + [(1, -1, -c) for c in range(9)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
        
    # Ensure unique points
    unique_points, indices = np.unique(best_points_global, axis=0, return_index=True)
    if len(unique_points) != 54:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(9)] + [(1, -1, -c) for c in range(9)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_points_global = unique_points[indices]
        
    # Ensure unique lines
    unique_lines = []
    seen_lines = set()
    for line in best_lines_global:
        n_line = normalize_line(int(line[0]), int(line[1]), int(line[2]))
        if n_line and n_line not in seen_lines:
            seen_lines.add(n_line)
            unique_lines.append(n_line)
            
    if len(unique_lines) < 27:
        lines = np.array([(1, 0, -i) for i in range(9)] + [(0, 1, -j) for j in range(9)] + [(1, -1, -c) for c in range(9)], dtype=np.int64)
        points, _ = get_best_points_for_lines(lines, candidate_points)
        best_points_global = points
        best_lines_global = lines
    else:
        best_lines_global = np.array(unique_lines[:27], dtype=np.int64)
        
    return best_points_global, best_lines_global
