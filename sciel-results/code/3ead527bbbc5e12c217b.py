import numpy as np
import time
import random
import sys

_P = 7
_Q = 49
_Q2 = 2401
_START = time.time()
_DEADLINE = _START + 620.0
_MARGIN = 15.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Precompute field operations
    mul_table = np.zeros((q, q), dtype=np.int32)
    add_table = np.zeros((q, q), dtype=np.int32)
    sub_table = np.zeros((q, q), dtype=np.int32)
    
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + 3 * b1 * b2) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add_table[i, j] = ra * p + rb
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub_table[i, j] = ra * p + rb
    
    # Build lines: 2450 lines total (49*49 non-vertical + 49 vertical)
    # Line ID: m*q + c for non-vertical, q2 + x0 for vertical
    # Each line has 49 points
    point_lines = [[] for _ in range(q2)]
    line_points = [[] for _ in range(2450)]
    
    for xi in range(q):
        for yi in range(q):
            pi = xi * q + yi
            vl = q2 + xi
            point_lines[pi].append(vl)
            line_points[vl].append(pi)
            for m in range(q):
                mc = sub_table[yi, mul_table[m, xi]]
                nl = m * q + mc
                point_lines[pi].append(nl)
                line_points[nl].append(pi)
    
    # For each line, store the set of points on it
    # For efficiency, use bitsets or just lists
    
    def check_validity(in_C):
        """Check if every point has a witness line with no other C points."""
        for i in range(q2):
            has_witness = False
            for lid in point_lines[i]:
                valid = True
                for q_idx in line_points[lid]:
                    if q_idx != i and in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    has_witness = True
                    break
            if not has_witness:
                return False
        return True
    
    def count_witness_lines(in_C):
        """Count for each point how many witness lines it has."""
        wcount = np.zeros(q2, dtype=np.int32)
        for i in range(q2):
            for lid in point_lines[i]:
                valid = True
                for q_idx in line_points[lid]:
                    if q_idx != i and in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    wcount[i] += 1
        return wcount
    
    def try_add_point(in_C, wcount, p_idx):
        """Try to add p_idx to C. Returns (success, new_wcount)."""
        # Check if adding p_idx would make any point lose all witness lines
        # A point x loses a witness line lid if:
        #   - lid passes through x and p_idx
        #   - lid was a witness for x (no other C points on lid except possibly x)
        #   - after adding p_idx, lid has p_idx (and x is not on lid... wait x is on lid)
        # Actually: lid is a witness for x if no point on lid (other than x) is in C.
        # If we add p_idx, and p_idx is on lid and p_idx != x, then lid is no longer a witness for x.
        # So for each line lid through p_idx, for each point x on lid (x != p_idx):
        #   if lid was a witness for x (i.e., no other C points on lid except x),
        #   then after adding p_idx, lid is no longer a witness for x.
        #   If wcount[x] was 1, then x loses all witnesses -> invalid.
        
        for lid in point_lines[p_idx]:
            for x_idx in line_points[lid]:
                if x_idx == p_idx:
                    continue
                # Check if lid is currently a witness for x_idx
                # lid is a witness for x_idx if no point on lid (other than x_idx) is in C
                # We need to check if p_idx is the only C point on lid other than x_idx
                # Actually, we need to check if there's any C point on lid other than x_idx
                # If p_idx is not in C yet, we check: is there any C point on lid other than x_idx?
                # If yes, lid is NOT a witness for x_idx.
                # If no, lid IS a witness for x_idx, and adding p_idx would destroy it.
                
                # Fast check: if wcount[x_idx] == 1 and lid is the only witness...
                # But we don't know which lines are witnesses. Let's just check.
                
                # Check if lid has any C point other than x_idx
                has_other_c = False
                for q_idx in line_points[lid]:
                    if q_idx != x_idx and q_idx != p_idx and in_C[q_idx]:
                        has_other_c = True
                        break
                if not has_other_c:
                    # lid is currently a witness for x_idx (assuming no other C points)
                    # Adding p_idx would destroy this witness
                    if wcount[x_idx] == 1:
                        return False, None
        
        # It's safe to add. Update wcount.
        new_wcount = wcount.copy()
        for lid in point_lines[p_idx]:
            for x_idx in line_points[lid]:
                if x_idx == p_idx:
                    continue
                # Check if lid was a witness for x_idx
                has_other_c = False
                for q_idx in line_points[lid]:
                    if q_idx != x_idx and q_idx != p_idx and in_C[q_idx]:
                        has_other_c = True
                        break
                if not has_other_c:
                    new_wcount[x_idx] -= 1
        
        return True, new_wcount
    
    def try_remove_point(in_C, wcount, p_idx):
        """Remove p_idx from C. Always safe. Update wcount."""
        new_wcount = wcount.copy()
        for lid in point_lines[p_idx]:
            for x_idx in line_points[lid]:
                if x_idx == p_idx:
                    continue
                # Check if lid becomes a witness for x_idx after removing p_idx
                # lid becomes a witness if no C point on lid other than x_idx
                has_other_c = False
                for q_idx in line_points[lid]:
                    if q_idx != x_idx and q_idx != p_idx and in_C[q_idx]:
                        has_other_c = True
                        break
                if not has_other_c:
                    new_wcount[x_idx] += 1
        return new_wcount
    
    # Strategy: Start with a random valid C, then do local search
    # to maximize |C|
    
    rng = random.Random(123)
    
    # Initialize: start with empty C, greedily add points
    in_C = np.zeros(q2, dtype=bool)
    wcount = np.zeros(q2, dtype=np.int32)
    
    # Greedy initialization: add points in random order
    order = list(range(q2))
    rng.shuffle(order)
    
    for p_idx in order:
        if time.time() > _DEADLINE - _MARGIN:
            break
        success, new_wcount = try_add_point(in_C, wcount, p_idx)
        if success:
            in_C[p_idx] = True
            wcount = new_wcount
    
    best_size = int(np.sum(in_C))
    best_in_C = in_C.copy()
    best_wcount = wcount.copy()
    
    # Local search: try to add points, and if stuck, remove some and re-add
    # Simulated annealing style
    
    temperature = 10.0
    cooling = 0.995
    
    start_ls = time.time()
    ls_budget = 500.0
    
    while time.time() - start_ls < ls_budget and temperature > 0.1:
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        # Try random additions
        attempts = 0
        max_attempts = 1000
        while attempts < max_attempts:
            if time.time() > _DEADLINE - _MARGIN:
                break
            attempts += 1
            p_idx = rng.randint(0, q2 - 1)
            if in_C[p_idx]:
                continue
            success, new_wcount = try_add_point(in_C, wcount, p_idx)
            if success:
                in_C[p_idx] = True
                wcount = new_wcount
                cur_size = int(np.sum(in_C))
                if cur_size > best_size:
                    best_size = cur_size
                    best_in_C = in_C.copy()
                    best_wcount = wcount.copy()
        
        # If no progress, try some removals to unstick
        if attempts == max_attempts:
            # Remove 1-3 random points
            num_remove = rng.randint(1, 3)
            for _ in range(num_remove):
                candidates = [i for i in range(q2) if in_C[i]]
                if not candidates:
                    break
                p_idx = rng.choice(candidates)
                in_C[p_idx] = False
                wcount = try_remove_point(in_C, wcount, p_idx)
        
        temperature *= cooling
    
    # Final: try to add more points with the best configuration
    in_C = best_in_C.copy()
    wcount = best_wcount.copy()
    
    order = list(range(q2))
    rng.shuffle(order)
    for p_idx in order:
        if time.time() > _DEADLINE - _MARGIN:
            break
        if in_C[p_idx]:
            continue
        success, new_wcount = try_add_point(in_C, wcount, p_idx)
        if success:
            in_C[p_idx] = True
            wcount = new_wcount
    
    cur_size = int(np.sum(in_C))
    if cur_size > best_size:
        best_size = cur_size
        best_in_C = in_C
    
    # Construct N = complement of C
    in_N = ~best_in_C
    
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
    
    if not N:
        # Fallback: return a single point
        N = [[[0, 0], [0, 0]]]
    
    return np.array(N, dtype=np.int64)
