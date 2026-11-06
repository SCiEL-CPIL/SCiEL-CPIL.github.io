import time
import random
import numpy as np
from itertools import combinations

N = 16
DEADLINE = 325.0
N_TOTAL = N * N

# Precompute all points and pairwise squared distances
_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
_POINT_INDEX = {p: i for i, p in enumerate(_ALL_POINTS)}
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles masks using numpy for speed
iso_c_mask = np.zeros((N_TOTAL, N_TOTAL), dtype=object)

for i in range(N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        di_j = _D2[i, j]
        # Vectorized check: which k form isosceles with i,j?
        di_k = _D2[i, :]
        dj_k = _D2[j, :]
        # Exclude i and j themselves
        mask_bool = (di_k == dj_k) | (di_k == di_j) | (dj_k == di_j)
        mask_bool[i] = False
        mask_bool[j] = False
        # Convert to bitmask
        mask = 0
        for k in np.where(mask_bool)[0]:
            mask |= (1 << k)
        iso_c_mask[i, j] = mask
        iso_c_mask[j, i] = mask

def _check_no_isosceles(S_mask, S_list):
    """Check if set S is no-isosceles using precomputed masks."""
    n = len(S_list)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            if iso_c_mask[a, b] & S_mask:
                return False
    return True

def _check_maximal(S_mask, S_list):
    """Check if S is maximal: no point can be added without creating isosceles."""
    S_set = set(S_list)
    for p in range(N_TOTAL):
        if p in S_set:
            continue
        # p can be added iff for all a, b in S, (p, a, b) is NOT isosceles
        # i.e., p is not in iso_c_mask[a][b] for any a, b in S
        # Equivalently: for all a in S, (iso_c_mask[p][a] & S_mask) == 0
        can_add = True
        for a in S_list:
            if iso_c_mask[p, a] & S_mask:
                can_add = False
                break
        if can_add:
            return False
    return True

def _evaluate(S_mask, S_list):
    """
    Evaluate a candidate set.
    Returns (penalty, size) where:
    - penalty = number of isosceles triples + large penalty if not maximal
    - size = len(S_list)
    """
    # Count isosceles violations
    n = len(S_list)
    iso_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            iso_count += bin(iso_c_mask[a, b] & S_mask).count('1')
    
    # Check maximality
    if not _check_maximal(S_mask, S_list):
        # Not maximal - count how many points can be added
        addable_count = 0
        S_set = set(S_list)
        for p in range(N_TOTAL):
            if p in S_set:
                continue
            can_add = True
            for a in S_list:
                if iso_c_mask[p, a] & S_mask:
                    can_add = False
                    break
            if can_add:
                addable_count += 1
        penalty = iso_count * 10 + addable_count * 5
    else:
        penalty = iso_count * 10
    
    return penalty, len(S_list)

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_set = None
    best_score = float('inf')  # We want to minimize: penalty * 1000 + size
    
    # Initialize with a random valid no-isosceles set
    current_set = []
    for p in range(N_TOTAL):
        if _check_no_isosceles(0, current_set):
            current_set.append(p)
    
    # Build initial mask
    def build_mask(S_list):
        mask = 0
        for idx in S_list:
            mask |= (1 << idx)
        return mask
    
    current_mask = build_mask(current_set)
    current_penalty, current_size = _evaluate(current_mask, current_set)
    current_score = current_penalty * 1000 + current_size
    
    if current_penalty == 0:
        best_set = current_set[:]
        best_score = current_score
    
    # Simulated Annealing
    T = 100.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.9995
    
    steps = 0
    max_steps = 500000
    
    while steps < max_steps:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        steps += 1
        
        # Choose a random move: add, remove, or swap a point
        move_type = rng.random()
        
        if move_type < 0.33:
            # Add a point
            if len(current_set) >= N_TOTAL:
                continue
            candidates = [p for p in range(N_TOTAL) if p not in set(current_set)]
            if not candidates:
                continue
            p_add = rng.choice(candidates)
            new_set = current_set + [p_add]
            new_mask = current_mask | (1 << p_add)
            
        elif move_type < 0.66:
            # Remove a point
            if len(current_set) <= 2:
                continue
            p_remove = rng.choice(current_set)
            new_set = [p for p in current_set if p != p_remove]
            new_mask = current_mask & ~(1 << p_remove)
            
        else:
            # Swap: remove one, add one
            if len(current_set) == 0:
                continue
            p_remove = rng.choice(current_set)
            candidates = [p for p in range(N_TOTAL) if p != p_remove and p not in set(current_set)]
            if not candidates:
                continue
            p_add = rng.choice(candidates)
            new_set = [p for p in current_set if p != p_remove] + [p_add]
            new_mask = (current_mask & ~(1 << p_remove)) | (1 << p_add)
        
        # Evaluate new state
        new_penalty, new_size = _evaluate(new_mask, new_set)
        new_score = new_penalty * 1000 + new_size
        
        # Accept or reject
        delta = new_score - current_score
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_set = new_set
            current_mask = new_mask
            current_penalty = new_penalty
            current_size = new_size
            current_score = new_score
            
            if current_penalty == 0 and current_score < best_score:
                best_score = current_score
                best_set = current_set[:]
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # Occasionally restart from best
        if steps % 5000 == 0 and best_set is not None:
            if rng.random() < 0.1:
                current_set = best_set[:]
                current_mask = build_mask(current_set)
                current_penalty, current_size = _evaluate(current_mask, current_set)
                current_score = current_penalty * 1000 + current_size
                T = 50.0  # Reset temperature
    
    # Final: try to reduce the best set
    if best_set is not None:
        S = list(best_set)
        changed = True
        while changed:
            changed = False
            for i in range(len(S)):
                if len(S) - 1 < 2:
                    continue
                S_test = S[:i] + S[i + 1:]
                S_test_mask = build_mask(S_test)
                p, s = _evaluate(S_test_mask, S_test)
                if p == 0 and s < len(S):
                    S = S_test
                    changed = True
                    break
        best_set = S
    
    # Fallback if no valid set found
    if best_set is None:
        chosen = []
        for i in range(N_TOTAL):
            if _check_no_isosceles(0, chosen):
                chosen.append(i)
        best_set = chosen
    
    return [_ALL_POINTS[i] for i in best_set]
