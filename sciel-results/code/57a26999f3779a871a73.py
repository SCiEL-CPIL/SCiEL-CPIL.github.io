import time
import itertools
import random
import math
import numpy as np

def _score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    U = set(U)
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    U_arr = np.array(sorted(U), dtype=np.int64)
    U_plus = np.unique(U_arr[:, None] + U_arr[None, :])
    U_minus = np.unique(U_arr[:, None] - U_arr[None, :])
    val = len(U_minus) / len(U_plus)
    if val <= 0:
        return 0.0
    return 1.0 + math.log(val) / math.log(2 * max_u + 1)

def _score_fast(U_arr):
    """Fast score using numpy array directly."""
    U_arr = np.sort(np.unique(U_arr))
    if len(U_arr) < 2 or U_arr[0] != 0:
        return 0.0
    max_u = U_arr[-1]
    if max_u == 0:
        return 0.0
    U_plus = np.unique(U_arr[:, None] + U_arr[None, :])
    U_minus = np.unique(U_arr[:, None] - U_arr[None, :])
    val = len(U_minus) / len(U_plus)
    if val <= 0:
        return 0.0
    return 1.0 + math.log(val) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    best_set = [0, 1, 3]
    best_score = _score(best_set)
    
    # Try parametric families
    # Family 1: Small exhaustive search
    for n in range(2, 7):
        if time.time() > deadline:
            break
        for combo in itertools.combinations(range(1, 21), n - 1):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Family 2: Powers-based sets
    for base in range(2, 12):
        for k in range(2, 10):
            if time.time() > deadline:
                break
            U = [base ** i for i in range(k)]
            if max(U) > 10000:
                continue
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Family 3: Golomb ruler-like constructions
    # Known good Golomb rulers
    golomb_rulers = [
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18, 27, 31],
        [0, 1, 4, 10, 18, 27, 37, 41],
        [0, 2, 7, 15, 22, 31, 40, 48, 54, 62, 67, 73],
    ]
    for gr in golomb_rulers:
        if time.time() > deadline:
            break
        s = _score(gr)
        if s > best_score:
            best_score = s
            best_set = gr[:]
    
    # Hill climbing on best set found
    def hill_climb(U, max_iters=3000):
        U = sorted(set(U))
        current_score = _score(U)
        max_u = max(U)
        U_set = set(U)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # Try adding a new element
            candidates = list(range(1, max_u + 8))
            random.shuffle(candidates)
            for new_el in candidates:
                if new_el not in U_set:
                    U_new = U + [new_el]
                    s = _score(U_new)
                    if s > current_score:
                        U = U_new
                        U_set = set(U)
                        current_score = s
                        max_u = max(U)
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try removing an element (except 0)
            removable = [el for el in U if el != 0]
            random.shuffle(removable)
            for el in removable:
                U_new = [x for x in U if x != el]
                if len(U_new) < 2:
                    continue
                s = _score(U_new)
                if s > current_score:
                    U = U_new
                    U_set = set(U)
                    current_score = s
                    max_u = max(U)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try replacing an element
            replaceable = [el for el in U if el != 0]
            random.shuffle(replaceable)
            for el in replaceable:
                if improved:
                    break
                for new_el in range(1, max_u + 8):
                    if new_el not in U_set and new_el != el:
                        U_new = [new_el if x == el else x for x in U]
                        s = _score(U_new)
                        if s > current_score:
                            U = U_new
                            U_set = set(U)
                            current_score = s
                            max_u = max(U)
                            improved = True
                            break
            
            if not improved:
                break
        
        return U, current_score
    
    # Hill climb from best found
    U_hc, score_hc = hill_climb(best_set)
    if score_hc > best_score:
        best_set = U_hc
        best_score = score_hc
    
    # Try multiple random starts with hill climbing
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline:
            break
        # Random set containing 0 with larger range
        size = random.randint(3, 15)
        max_el = random.randint(10, 200)
        U_rand = [0] + random.sample(range(1, max_el + 1), min(size - 1, max_el))
        U_hc, score_hc = hill_climb(U_rand, max_iters=1500)
        if score_hc > best_score:
            best_set = U_hc
            best_score = score_hc
    
    # Final polish: try small modifications with more iterations
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        max_u = max(best_set)
        best_set_set = set(best_set)
        
        # Try adding
        for new_el in range(1, max_u + 5):
            if new_el not in best_set_set:
                U_new = best_set + [new_el]
                s = _score(U_new)
                if s > best_score:
                    best_set = U_new
                    best_set_set = set(U_new)
                    best_score = s
                    improved = True
                    break
        if improved:
            continue
        
        # Try removing
        for i, el in enumerate(best_set):
            if el == 0:
                continue
            U_new = best_set[:i] + best_set[i+1:]
            if len(U_new) >= 2:
                s = _score(U_new)
                if s > best_score:
                    best_set = U_new
                    best_set_set = set(U_new)
                    best_score = s
                    improved = True
                    break
        if improved:
            continue
        
        # Try replacing
        for i, el in enumerate(best_set):
            if el == 0:
                continue
            for new_el in range(1, max_u + 5):
                if new_el not in best_set_set:
                    U_new = best_set[:]
                    U_new[i] = new_el
                    s = _score(U_new)
                    if s > best_score:
                        best_set = U_new
                        best_set_set = set(U_new)
                        best_score = s
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return sorted(best_set)
