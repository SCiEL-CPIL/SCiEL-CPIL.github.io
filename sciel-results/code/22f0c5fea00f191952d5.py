import time
import random
from math import log, floor
import numpy as np
from itertools import combinations

def fast_score(U):
    """
    Faster scoring using numpy for larger sets.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums
    # i <= j
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences
    # i < j
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    # Unique positive diffs
    num_pos_diffs = len(np.unique(diffs))
    # Total diffs = 2 * num_pos_diffs + 1 (for 0)
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def geometric_set(base, m):
    """
    Construct a set U = {0} U { floor(base^k) for k=1..m }
    """
    U = [0]
    val = 1
    for _ in range(m):
        val = int(val * base)
        if val <= 0:
            break
        if val not in U:
            U.append(val)
    return sorted(set(U))

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    This helps explore the neighborhood more intelligently than random perturbation.
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

def dynamic_local_search(U, deadline, max_n_limit=25, max_val_limit=500):
    """
    Local search with dynamic candidate generation.
    Moves:
    1. Add an element from candidate pool.
    2. Remove an element.
    3. Replace an element with a candidate.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    # Initial candidate pool
    candidates = generate_candidates(current, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 20
    
    start_time = time.time()
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        r = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.15:
            current_max = max(current)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(current, new_limit)
        
        if r < 0.5:
            # Try to ADD an element
            if len(current) < max_n_limit and candidates:
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in current:
                        continue
                    new_U = current + [c]
                    new_U.sort()
                    new_score = fast_score(new_U)
                    if new_score > current_score:
                        current = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(current, max(max_val_limit, max(current)*2))
                        break
        elif r < 0.7:
            # Try to REMOVE an element
            if len(current) > 3:
                removable = [x for x in current if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in current if x != c]
                    new_score = fast_score(new_U)
                    if new_score > current_score:
                        current = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(current, max_val_limit)
        else:
            # Try to REPLACE an element
            if len(current) > 1 and candidates:
                idx = random.randint(1, len(current) - 1)
                old_val = current[idx]
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in current:
                        continue
                    new_U = current.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = fast_score(new_U)
                    if new_score > current_score:
                        current = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(current, max(max_val_limit, max(current)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_score:
                best_score = current_score
                best = current[:]
        else:
            no_improvement_count += 1
                
    return best, best_score

def high_precision_polish(U, deadline):
    """
    Final high-precision polish: try all single-element perturbations with adaptive steps,
    plus pair swaps and element replacement.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current[:]
    best_score = current_score
    
    n = len(current)
    max_val = max(current)
    
    # Adaptive step size based on set density
    step_range = max(3, min(15, int(max_val * 0.02)))
    
    # Phase 1: Single-element perturbations with adaptive range
    for i in range(1, n):
        if time.time() > deadline:
            break
        old_val = current[i]
        for offset in range(-step_range, step_range + 1):
            if offset == 0:
                continue
            new_val = old_val + offset
            if new_val <= 0:
                continue
            test_set = current[:]
            test_set[i] = new_val
            test_set = sorted(set(test_set))
            if len(test_set) != n:
                continue
            s = fast_score(test_set)
            if s > best_score:
                best_score = s
                best = test_set[:]
                current = test_set[:]
                old_val = new_val
    
    # Phase 2: Try replacing each element with a value in a wider range
    if time.time() < deadline - 5:
        for i in range(1, n):
            if time.time() > deadline:
                break
            old_val = current[i]
            # Try values at various fractions of max_val
            for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5]:
                new_val = max(1, int(max_val * frac))
                if new_val == old_val:
                    continue
                test_set = current[:]
                test_set[i] = new_val
                test_set = sorted(set(test_set))
                if len(test_set) != n:
                    continue
                s = fast_score(test_set)
                if s > best_score:
                    best_score = s
                    best = test_set[:]
                    current = test_set[:]
    
    # Phase 3: Pair swaps
    if time.time() < deadline - 2 and n > 2:
        for i in range(1, n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                test_set = current[:]
                test_set[i], test_set[j] = test_set[j], test_set[i]
                test_set = sorted(test_set)
                s = fast_score(test_set)
                if s > best_score:
                    best_score = s
                    best = test_set[:]
                    current = test_set[:]
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: evaluator kills at 150s, return well before
    
    best_set = [0, 1, 4, 9, 11]
    best_score = fast_score(best_set)
    
    # 1. Structured Candidates (from Inspiration)
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52], # Golomb
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67], # Golomb
    ]
    
    for cand in structured_candidates:
        s = fast_score(cand)
        if s > best_score:
            best_score = s
            best_set = cand[:]

    # 2. Geometric Search (from Current)
    # Quick coarse grid to find a good geometric seed
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
            s = fast_score(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U[:]

    # 3. Dynamic Local Search (from Inspiration)
    # Use the best found so far as the primary seed, and structured ones as secondary seeds
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 25
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = base[:]
        
        # Run local search
        local_score, local_set = dynamic_local_search(U, deadline, max_n_limit, max_val_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 4. Final Polish: Fine-grained local search on the best set
    # This addresses the "refinement machinery" requirement.
    if time.time() < deadline - 2:
        refined, refined_score = high_precision_polish(best_set, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined[:]
            
    return best_set

# Test
if __name__ == "__main__":
    print(f"Best set: {construct()}")
