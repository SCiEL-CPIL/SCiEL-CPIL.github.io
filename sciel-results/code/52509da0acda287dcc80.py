import time
from math import log, floor, sqrt
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def score_set_fast(U):
    """Optimized scoring using numpy where possible, fallback to set-based."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # For small sets, the set-based approach is actually faster due to overhead
    # But for larger sets, we can be smarter
    if n <= 30:
        return score_set(U)
    
    # For larger sets, use numpy convolution-like approach for sums
    # Actually, for n up to 50, the O(n^2) set approach is still fine
    # Let's just use the standard one but with local variables for speed
    U_list = U
    n_local = n
    sums = set()
    diffs = set()
    
    for i in range(n_local):
        ui = U_list[i]
        for j in range(i, n_local):
            sums.add(ui + U_list[j])
        for j in range(i + 1, n_local):
            d = U_list[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = U_list[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def exhaustive_small_search(n_target, max_val, deadline):
    """
    Exhaustively search for the best set of size n_target (including 0) with elements up to max_val.
    """
    best_score = -1
    best_set = None
    
    if n_target == 3:
        for a in range(1, max_val + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, max_val + 1):
                if time.time() > deadline:
                    break
                U = [0, a, b]
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    elif n_target == 4:
        for a in range(1, max_val + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, max_val + 1):
                if time.time() > deadline:
                    break
                for c in range(b + 1, max_val + 1):
                    if time.time() > deadline:
                        break
                    U = [0, a, b, c]
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U
    elif n_target == 5:
        for a in range(1, max_val + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, max_val + 1):
                if time.time() > deadline:
                    break
                for c in range(b + 1, max_val + 1):
                    if time.time() > deadline:
                        break
                    for d in range(c + 1, max_val + 1):
                        if time.time() > deadline:
                            break
                        U = [0, a, b, c, d]
                        s = score_set(U)
                        if s > best_score:
                            best_score = s
                            best_set = U
    elif n_target == 6:
        for a in range(1, max_val + 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, max_val + 1):
                if time.time() > deadline:
                    break
                for c in range(b + 1, max_val + 1):
                    if time.time() > deadline:
                        break
                    for d in range(c + 1, max_val + 1):
                        if time.time() > deadline:
                            break
                        for e in range(d + 1, max_val + 1):
                            if time.time() > deadline:
                                break
                            U = [0, a, b, c, d, e]
                            s = score_set(U)
                            if s > best_score:
                                best_score = s
                                best_set = U
    return best_score, best_set

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def generate_candidates(U, limit_val):
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

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
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
            candidates = generate_candidates(U, new_limit)
        
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
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
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
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def mixed_radix_construct(radices, digits):
    """
    Construct a set from mixed-radix expansion.
    radices: list of radices [r_0, r_1, ..., r_{k-1}]
    digits: list of digit sets, digits[i] is a list of digits for position i
    U = { sum d_i * product_{j<i} r_j }
    """
    U = set()
    U.add(0)
    
    # Build all combinations
    # For efficiency, we build iteratively
    current_values = [0]
    multiplier = 1
    
    for i, (r, digit_set) in enumerate(zip(radices, digits)):
        new_values = []
        for base in current_values:
            for d in digit_set:
                new_values.append(base + d * multiplier)
        current_values = new_values
        multiplier *= r
    
    for v in current_values:
        if v >= 0:
            U.add(v)
    
    return sorted(list(U))

def optimize_mixed_radix(deadline):
    """
    Greedy optimization of mixed-radix parameters.
    We start with a small set and try to extend it.
    """
    best_score = -1
    best_set = [0]
    
    # Start with a few positions
    max_positions = 8
    max_radix = 10
    
    # We'll try to build the set position by position
    # radices = [r_0, r_1, ...]
    # digits[i] = {0, d_i} typically, or {0, 1, ..., d_i}
    
    # Let's try a simpler approach: 
    # U = { sum a_i * B^i } where a_i in {0, 1, ..., A_i}
    # This is like a mixed-radix number system with varying digit ranges.
    
    # Greedy: start with [0], try to add the next "level"
    # Level i has radix r_i and digit set {0, ..., D_i}
    
    # Let's parameterize by r_i and D_i
    # To keep it simple, let's fix r_i = r for all i, and vary D_i
    
    # Actually, let's just do a random search over small mixed-radix structures
    # and then local search.
    
    # Generate random mixed-radix structures
    for _ in range(100):
        if time.time() > deadline:
            break
        
        # Random parameters
        num_positions = random.randint(2, 6)
        radices = []
        digit_sets = []
        
        for i in range(num_positions):
            r = random.randint(2, 5)
            D = random.randint(1, 3)
            radices.append(r)
            digit_sets.append(list(range(D + 1)))
        
        U = mixed_radix_construct(radices, digit_sets)
        
        if 0 in U and len(U) > 1:
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Exhaustive search for small sets to find strong seeds
    if time.time() < deadline - 80:
        ex_score, ex_set = exhaustive_small_search(3, 30, deadline - 80)
        if ex_set and ex_score > best_score:
            best_score = ex_score
            best_set = ex_set
        
        if time.time() < deadline - 60:
            ex_score, ex_set = exhaustive_small_search(4, 20, deadline - 60)
            if ex_set and ex_score > best_score:
                best_score = ex_score
                best_set = ex_set
        
        if time.time() < deadline - 40:
            ex_score, ex_set = exhaustive_small_search(5, 15, deadline - 40)
            if ex_set and ex_score > best_score:
                best_score = ex_score
                best_set = ex_set

    # 2. Structured Candidates
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
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 3. Geometric Search
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 40:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 4. Mixed Radix Construction
    if time.time() < deadline - 30:
        mr_score, mr_set = optimize_mixed_radix(deadline - 30)
        if mr_set and mr_score > best_score:
            best_score = mr_score
            best_set = mr_set

    # 5. Dynamic Local Search
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run local search
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 6. Final Polish: Fine-grained local search on the best set
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 500:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values (small delta)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try deltas in a wider range but smaller steps
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element near the max
            if not improved and n < 25 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 10, max_val + 11):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

# Ensure 0 is in the set and it's a list of nonnegative integers
# The function construct returns a list.
# The evaluator calls construct().

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
