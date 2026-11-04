import time
from math import log, floor, ceil
import itertools
import random
import numpy as np

def score_set(U):
    """
    Computes the score for a set U.
    Score = 1 + log(|U-U| / |U+U|) / log(2*max(U)+1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    # Use numpy for efficiency on larger sets, python sets for very small
    if n <= 15:
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
    else:
        U_arr = np.array(U, dtype=np.int64)
        sums_arr = U_arr[:, None] + U_arr[None, :]
        unique_sums = np.unique(sums_arr)
        diffs_arr = U_arr[:, None] - U_arr[None, :]
        unique_diffs = np.unique(diffs_arr)
        sums = set(unique_sums.tolist())
        diffs = set(unique_diffs.tolist())
        
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
    """
    Optimized scoring.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    # Vectorized scoring for speed
    U_arr = np.array(U, dtype=np.int64)
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

# Known optimal Golomb rulers (minimal diameter for n marks)
GOLOMB_RULERS = {
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 9, 11],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 18, 23, 25],
    8: [0, 1, 4, 10, 18, 23, 29, 34],
    9: [0, 1, 4, 10, 18, 23, 29, 37, 41],
    10: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
    11: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58],
    12: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67],
    13: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67, 73],
    14: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67, 73, 82],
    15: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67, 73, 82, 89],
    16: [0, 1, 4, 10, 18, 23, 29, 37, 44, 52, 58, 67, 73, 82, 89, 99],
}

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    This explores the arithmetic neighborhood to escape local minima.
    """
    U_set = set(U)
    candidates = set()
    
    # Sums and differences of existing elements
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
    
    # Add some random small integers to ensure diversity
    for _ in range(20):
        r = random.randint(1, min(50, limit_val))
        if r not in U_set:
            candidates.add(r)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set_fast(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 20 # Lower to allow more restarts or faster exit
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
                for c in candidates[:15]: # Try a few more candidates
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set_fast(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set_fast(new_U)
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
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set_fast(new_U)
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

def construct():
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Initial scoring
    s = score_set_fast(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)

    # Phase 1: Structured seeds (Golomb rulers and other patterns)
    structured_candidates = []
    for n in range(4, 17):
        if n in GOLOMB_RULERS:
            structured_candidates.append(GOLOMB_RULERS[n])
    
    # Add other patterns
    structured_candidates.extend([
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ])
    
    for cand in structured_candidates:
        if time.time() > deadline - 10:
            break
        s = score_set_fast(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    # Phase 2: Dynamic Local Search with Restarts
    # We run multiple restarts to explore different basins of attraction
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        
        # Choose a starting set
        if restarts % 3 == 0:
            # Start from a structured set
            base = random.choice(structured_candidates)
            U = list(base)
        else:
            # Start from random
            n = random.randint(4, 15)
            if n - 1 > max_val_limit:
                n = max_val_limit
            elements = random.sample(range(1, max_val_limit + 1), n - 1)
            U = [0] + elements
            U.sort()
        
        # Run local search
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    # Phase 3: Final Polish
    # Run a final local search on the best set found, potentially with tighter constraints or more iterations
    if time.time() < deadline - 2:
        U = list(best_set)
        # Re-run dynamic local search to squeeze out last bits of improvement
        final_score, final_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
            
    return best_set
