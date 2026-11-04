import time
import math
import random
import numpy as np

def _score_fast(U):
    """Compute the score for a given set U using NumPy for speed."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = int(U_arr[-1])
    if max_u == 0:
        return 1.0
    
    # Compute U+U and U-U using broadcasting
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
    
    ratio = diff_set_size / sum_set_size
    if ratio <= 1:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _score(U):
    """Compute the score for a given set U (as a sorted list)."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0  # degenerate case
    
    # Compute U+U
    sum_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
    
    # Compute U-U
    diff_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def _local_search_dynamic(U, deadline, max_val_limit=500, max_n_limit=30):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = _score_fast(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = _generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = _generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:10]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = _score_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = _score_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate
                random.shuffle(candidates)
                for new_val in candidates[:10]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = _score_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = _generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def _hamming_code_sets(n, deadline):
    """
    Generate sets based on Hamming code / LDPC-like structures.
    These structures have high difference-to-sum ratios due to controlled bit-flip distances.
    """
    sets = []
    # Standard Hamming code weights
    for k in range(3, 7):
        if time.time() > deadline:
            break
        # Generate codewords of weight k for Hamming code
        # Simplified: use binary representations of numbers with exactly k bits set
        # This creates a structure where differences are "spread out"
        max_bits = 8
        vals = set()
        for i in range(1, 1 << max_bits):
            if bin(i).count('1') == k:
                vals.add(i)
        if vals:
            # Take a subset to keep size manageable
            vals_list = sorted(vals)
            if len(vals_list) > 20:
                # Randomly sample or take first N
                step = len(vals_list) // 20
                vals_list = vals_list[::step][:20]
            U = [0] + vals_list
            sets.append(sorted(set(U)))
    
    # LDPC-like: sparse binary matrices
    for trial in range(10):
        if time.time() > deadline:
            break
        # Random sparse binary vector of length 10-15
        length = random.randint(10, 15)
        density = 0.3
        bits = [1 if random.random() < density else 0 for _ in range(length)]
        val = 0
        for i, b in enumerate(bits):
            if b:
                val |= (1 << i)
        # Add multiples or shifts
        U = set([0])
        base = val
        for shift in range(0, 5):
            U.add(base << shift)
            U.add((base << shift) | (base >> 1))
        U = sorted(list(U))
        if len(U) > 1:
            sets.append(U)
    
    return sets

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = _score_fast(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    for cand in structured_candidates:
        s = _score_fast(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Hamming Code / LDPC Structures
    hamming_sets = _hamming_code_sets(5, deadline)
    for cand in hamming_sets:
        s = _score_fast(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 3. Geometric Search (from Current)
    # Quick coarse grid to find a good geometric seed
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    def geometric_set(b, m):
        U = [0]
        seen = {0}
        for k in range(1, m + 1):
            val = int(math.floor(b ** k))
            if val <= 0:
                val = 1
            if val not in seen:
                U.append(val)
                seen.add(val)
        return sorted(U)

    for b in b_candidates:
        if time.time() > deadline - 40:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = _score_fast(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 4. Dynamic Local Search
    # Use the best found so far as the primary seed, and structured ones as secondary seeds
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    if hamming_sets:
        seeds.append(hamming_sets[0])
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run local search
        local_score, local_set = _local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts? 
            # To keep it simple, we just update the primary seed list
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 5. Final Polish: Fine-grained local search on the best set
    if time.time() < deadline - 2:
        final_score, final_set = _local_search_dynamic(best_set, deadline, max_val_limit, max_n_limit)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set

    return best_set
