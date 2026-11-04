import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    """
    Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    Uses NumPy for speed.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all sums (i <= j)
    sums_arr = U_arr[:, None] + U_arr[None, :]
    sum_vals = np.unique(sums_arr)
    sum_count = len(sum_vals)
    
    # Compute all differences (i < j, and their negatives, plus 0)
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    diff_vals = np.unique(diffs_arr)
    diff_count = len(diff_vals)
    
    if sum_count == 0:
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def generate_recurrence_set(a, b, n):
    """
    Generate a set from a linear recurrence u_{k+1} = a*u_k + b*u_{k-1}
    u_0 = 0, u_1 = 1
    """
    if n < 2:
        return [0]
    U = [0, 1]
    for k in range(2, n):
        next_val = int(a * U[-1] + b * U[-2])
        if next_val <= U[-1]:
            # Ensure strictly increasing
            next_val = U[-1] + 1
        U.append(next_val)
    return U

def local_search(U, deadline, max_val_limit=2000, max_n_limit=25):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Generate initial candidates
    U_set = set(U)
    candidates = set()
    max_u = max(U)
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= max_val_limit and s not in U_set:
                candidates.add(s)
            if 0 < d <= max_val_limit and d not in U_set:
                candidates.add(d)
    
    # Add some random candidates
    for _ in range(50):
        r = random.randint(1, max_val_limit)
        if r not in U_set:
            candidates.add(r)
            
    candidates = list(candidates)
    
    no_improvement_count = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 5:
        if no_improvement_count > max_no_improvement:
            # Restart with new candidates
            U_set = set(U)
            candidates = set()
            max_u = max(U)
            for a in U:
                for b in U:
                    s = a + b
                    d = abs(a - b)
                    if 0 < s <= max_val_limit and s not in U_set:
                        candidates.add(s)
                    if 0 < d <= max_val_limit and d not in U_set:
                        candidates.add(d)
            for _ in range(50):
                r = random.randint(1, max_val_limit)
                if r not in U_set:
                    candidates.add(r)
            candidates = list(candidates)
            no_improvement_count = 0
            
        improved = False
        move_type = random.random()
        
        if move_type < 0.4:
            # Add element
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
                        break
        elif move_type < 0.6:
            # Remove element
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
        else:
            # Replace element
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
    deadline = time.time() + 115
    best_score = -1.0
    best_set = [0, 1, 4, 9, 11]
    
    # Initial seeds
    seeds = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 20, 35, 56, 84],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 10, 18, 28, 40, 54],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 9, 27, 81, 243],
        [0, 1, 5, 14, 34, 71, 135],
        [0, 1, 6, 22, 65, 168, 370],
    ]
    
    # Add recurrence-based seeds
    for a in [1.5, 2.0, 2.5, 3.0]:
        for b in [-0.5, 0.0, 0.5]:
            U = generate_recurrence_set(a, b, 10)
            seeds.append(U)
    
    for cand in seeds:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    # Local search from best seed
    if time.time() < deadline - 10:
        local_score, local_set = local_search(best_set, deadline)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            
    # Multiple restarts
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        if restarts % 5 == 0:
            # Start from a random seed
            base = random.choice(seeds)
            U = list(base)
        else:
            # Start from a perturbed best set
            U = list(best_set)
            # Perturb
            for _ in range(3):
                if random.random() < 0.5:
                    # Add
                    if len(U) < 25:
                        new_val = random.randint(1, 2000)
                        if new_val not in U:
                            U.append(new_val)
                else:
                    # Remove
                    if len(U) > 2:
                        removable = [x for x in U if x != 0]
                        if removable:
                            U.remove(random.choice(removable))
            U = sorted(set(U))
            
        local_score, local_set = local_search(U, deadline)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            
    return best_set

# EVOLVE-BLOCK-END
