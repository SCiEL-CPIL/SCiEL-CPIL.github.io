import time
from math import log
import itertools
import numpy as np
import random

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    # Use NumPy for faster computation
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    
    # Compute all pairwise differences
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

def _incremental_score_gain(U_current, new_elem):
    """
    Compute the approximate score gain from adding new_elem to U_current.
    Uses the idea of 'collision penalty' (new sums that already exist) vs 
    'dispersion gain' (new differences that are new).
    """
    U_arr = np.array(sorted(U_current), dtype=np.int64)
    n = len(U_arr)
    
    # Current sums and diffs
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums_set = set(np.unique(sums_arr).tolist())
    
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs_set = set(np.unique(diffs_arr).tolist())
    
    n_sums = len(unique_sums_set)
    n_diffs = len(unique_diffs_set)
    
    # New sums: new_elem + u for u in U_current
    new_sums = {new_elem + u for u in U_arr.tolist()}
    new_diffs = {new_elem - u for u in U_arr.tolist()} | {u - new_elem for u in U_arr.tolist()}
    
    # Count how many are truly new
    new_sum_count = len(new_sums - unique_sums_set)
    new_diff_count = len(new_diffs - unique_diffs_set)
    
    # Total after addition
    total_sums = n_sums + new_sum_count
    total_diffs = n_diffs + new_diff_count
    
    if total_sums == 0:
        return -1
        
    ratio = total_diffs / total_sums
    if ratio <= 1:
        return -1
        
    max_u = max(max(U_arr), new_elem)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def _greedy_construct_with_weights(max_candidates=200, target_size=15):
    """
    Greedy construction using weighted vertex selection.
    At each step, pick the candidate that maximizes net score improvement.
    """
    U = [0]
    candidates = list(range(1, max_candidates + 1))
    
    for step in range(target_size - 1):
        best_elem = None
        best_score = -1
        
        for c in candidates:
            if c in U:
                continue
            s = _incremental_score_gain(U, c)
            if s > best_score:
                best_score = s
                best_elem = c
        
        if best_elem is None:
            break
        
        U.append(best_elem)
        candidates.remove(best_elem)
    
    return sorted(U)

def construct():
    deadline = time.time() + 110  # Safety margin
    
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Update best if initial is good
    s = score_set(best_set)
    if s > best_score:
        best_score = s
    
    # Phase 1: Greedy construction with weighted selection
    if time.time() < deadline - 20:
        greedy_set = _greedy_construct_with_weights(max_candidates=150, target_size=12)
        gs = score_set(greedy_set)
        if gs > best_score:
            best_score = gs
            best_set = greedy_set
        
        # Also try different target sizes
        for ts in [8, 10, 14, 16]:
            if time.time() > deadline - 20:
                break
            gset = _greedy_construct_with_weights(max_candidates=200, target_size=ts)
            gs = score_set(gset)
            if gs > best_score:
                best_score = gs
                best_set = gset
    
    # Phase 2: Exhaustive search for small sets with moderate max_val
    max_val = 40
    
    for n in range(3, 6): # Sizes 3 to 5 (keep it fast)
        if time.time() > deadline - 20:
            break
            
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline - 20:
                break
                
            U = (0,) + combo
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 3: Local search to refine the best set found
    if time.time() < deadline - 15:
        # Try adding elements using incremental scoring for speed
        current_set = set(best_set)
        
        # Try adding elements up to 400
        for new_elem in range(1, 400):
            if time.time() > deadline - 15:
                break
            if new_elem in current_set:
                continue
            # Use incremental scoring for quick filter
            inc_score = _incremental_score_gain(current_set, new_elem)
            if inc_score > best_score - 0.01:  # Only do full score if promising
                new_set = current_set | {new_elem}
                s = score_set(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
                    current_set = new_set
        
        # Try removing elements (keep at least 2 elements)
        if len(best_set) > 2:
            for elem in list(best_set):
                if time.time() > deadline - 15:
                    break
                if elem == 0:
                    continue  # Must keep 0
                new_set = set(best_set) - {elem}
                if len(new_set) < 2:
                    continue
                s = score_set(list(new_set))
                if s > best_score:
                    best_score = s
                    best_set = list(new_set)
    
    # Phase 4: Simulated annealing for better exploration
    if time.time() < deadline - 10:
        random.seed(42)
        current = set(best_set)
        current_score = best_score
        T = 0.1
        T_min = 0.001
        cooling = 0.998
        
        while T > T_min and time.time() < deadline - 5:
            # Generate a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random element
                max_elem = max(current) if current else 0
                new_elem = random.randint(1, max_elem * 2 + 100)
                if new_elem not in current:
                    candidate = current | {new_elem}
            elif move_type < 0.7:
                # Remove a random element (not 0)
                removable = [x for x in current if x != 0]
                if removable and len(current) > 2:
                    remove_elem = random.choice(removable)
                    candidate = current - {remove_elem}
                else:
                    continue
            else:
                # Swap: remove one, add another
                removable = [x for x in current if x != 0]
                if removable:
                    remove_elem = random.choice(removable)
                    max_elem = max(current)
                    new_elem = random.randint(1, max_elem * 2 + 100)
                    if new_elem not in current:
                        candidate = (current - {remove_elem}) | {new_elem}
                    else:
                        continue
                else:
                    continue
            
            candidate_score = score_set(list(candidate))
            
            if candidate_score > current_score or random.random() < np.exp((candidate_score - current_score) / T):
                current = candidate
                current_score = candidate_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current)
            
            T *= cooling
    
    # Final polish: try small perturbations around best
    if time.time() < deadline - 3:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            perturbed = list(best_set)
            if random.random() < 0.5 and len(perturbed) > 2:
                idx = random.randint(1, len(perturbed) - 1)
                perturbed[idx] += random.randint(-5, 5)
                if perturbed[idx] < 1:
                    perturbed[idx] = 1
            else:
                max_elem = max(perturbed)
                perturbed.append(random.randint(1, max_elem + 50))
            
            ps = score_set(perturbed)
            if ps > best_score:
                best_score = ps
                best_set = perturbed
    
    return best_set
