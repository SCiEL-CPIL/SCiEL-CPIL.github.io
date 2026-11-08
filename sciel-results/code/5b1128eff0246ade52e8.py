import time
from math import log, floor
import random
import numpy as np
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
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

def local_search_dynamic(U, deadline, max_val_limit=2000, max_n_limit=50):
    """
    Local search with dynamic candidate generation and double swap moves.
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
        if not candidates or random.random() < 0.05:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.4:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
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
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.6:
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
        elif move_type < 0.8:
            # Try to REPLACE an element (Single Swap)
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
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
        else:
            # Try DOUBLE SWAP (Replace 2 elements simultaneously)
            if len(U) > 2 and candidates:
                idx1 = random.randint(1, len(U) - 1)
                idx2 = random.randint(1, len(U) - 1)
                if idx1 == idx2:
                    continue
                
                old1 = U[idx1]
                old2 = U[idx2]
                
                random.shuffle(candidates)
                for new_val1 in candidates[:5]:
                    if new_val1 in U:
                        continue
                    for new_val2 in candidates[:5]:
                        if new_val2 in U or new_val2 == new_val1:
                            continue
                        
                        new_U = U.copy()
                        new_U[idx1] = new_val1
                        new_U[idx2] = new_val2
                        new_U.sort()
                        
                        if len(set(new_U)) != len(U):
                            continue
                            
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                            break
                    if improved:
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def exhaustive_search(n, max_val):
    """Find best set of size n (including 0) with elements <= max_val."""
    best_s = -1
    best_c = None
    range_vals = range(1, max_val + 1)
    for comb in combinations(range_vals, n - 1):
        U = [0] + list(comb)
        s = score_set(U)
        if s > best_s:
            best_s = s
            best_c = U
    return best_s, best_c

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Aggressive Exhaustive Search for Small n
    # This is the primary structural improvement from Inspiration 2
    # We allocate specific time budgets to each n
    time_per_n = 5  # 5 seconds per n level
    
    # n=3
    if time.time() < deadline - 30:
        s, c = exhaustive_search(3, 100)
        if s > best_score:
            best_score = s
            best_set = c
            
    # n=4
    if time.time() < deadline - 25:
        s, c = exhaustive_search(4, 60)
        if s > best_score:
            best_score = s
            best_set = c
            
    # n=5
    if time.time() < deadline - 20:
        s, c = exhaustive_search(5, 40)
        if s > best_score:
            best_score = s
            best_set = c
            
    # n=6
    if time.time() < deadline - 15:
        s, c = exhaustive_search(6, 30)
        if s > best_score:
            best_score = s
            best_set = c

    # 2. Structured Seeds for Local Search
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 4, 9, 11], # Known good small set
    ]
    
    # Update best with any structured candidate that is better
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 3. Local Search Phase
    # Run local search from the best seed found so far
    if time.time() < deadline - 5:
        local_score, local_set = local_search_dynamic(best_set, deadline)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Run local search from a few other high-quality seeds if time permits
    if time.time() < deadline - 5:
        # Pick a different seed, e.g., the best structured one that isn't the current best
        alt_seed = None
        for cand in structured_candidates:
            if cand != best_set:
                alt_seed = cand
                break
        
        if alt_seed:
            local_score2, local_set2 = local_search_dynamic(alt_seed, deadline)
            if local_score2 > best_score:
                best_score = local_score2
                best_set = local_set2

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
