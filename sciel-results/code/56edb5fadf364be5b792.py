import math
import random
import time
from itertools import combinations

def _score(U):
    """Compute the score for a set U."""
    U = list(set(U))
    if 0 not in U:
        U.append(0)
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute |U+U| and |U-U|
    U_set = set(U)
    
    # Sumset
    sumset = set()
    for i, a in enumerate(U):
        for b in U[i:]:  # a+b = b+a
            sumset.add(a + b)
    sumset_size = len(sumset)
    
    # Difference set
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    log_ratio = math.log(ratio)
    log_base = math.log(2 * max_u + 1)
    
    if log_base <= 0:
        return 0.0
    
    return 1.0 + log_ratio / log_base

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_set = [0, 1, 3]
    best_score = _score(best_set)
    
    # Generate diverse initial seeds
    seeds = []
    
    # Seed 1: Small Sidon-like set
    seeds.append([0, 1, 3, 7, 12, 18, 25, 33, 42, 52])
    
    # Seed 2: Powers of 2 (maximizes difference set)
    seeds.append([0] + [2**i for i in range(1, 12)])
    
    # Seed 3: Quadratic growth
    seeds.append([0] + [i*i for i in range(1, 15)])
    
    # Seed 4: Mixed
    seeds.append([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
    
    # Seed 5: Sparse
    seeds.append([0, 2, 5, 11, 23, 47, 95, 191])
    
    # Seed 6: Near-AP with perturbation
    base = [0] + [5*i for i in range(1, 15)]
    random.seed(42)
    for i in range(1, len(base)):
        base[i] += random.randint(-1, 1)
    base = sorted(set(base))
    seeds.append(base)
    
    # Evaluate seeds
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        s = _score(seed)
        if s > best_score:
            best_score = s
            best_set = list(seed)
    
    # Local search with swap/add/remove moves
    current_set = list(best_set)
    current_score = best_score
    
    # Determine max value for candidate additions
    max_val = max(current_set) * 3 + 100
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5 and len(current_set) > 2:
            # Swap: replace one element with a new one
            idx = random.randint(1, len(current_set) - 1)  # Don't remove 0
            old_val = current_set[idx]
            new_val = random.randint(1, max_val)
            if new_val not in current_set:
                trial = current_set[:]
                trial[idx] = new_val
                trial = sorted(set(trial))
                if len(trial) >= 2 and 0 in trial:
                    s = _score(trial)
                    if s > current_score:
                        current_set = trial
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(current_set)
                        no_improve_count = 0
                        max_val = max(current_set) * 3 + 100
                    else:
                        no_improve_count += 1
        elif move_type < 0.8 and len(current_set) < 50:
            # Add: add a new element
            new_val = random.randint(1, max_val)
            if new_val not in current_set:
                trial = current_set + [new_val]
                trial = sorted(trial)
                s = _score(trial)
                if s > current_score:
                    current_set = trial
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = list(current_set)
                    no_improve_count = 0
                    max_val = max(current_set) * 3 + 100
                else:
                    no_improve_count += 1
        else:
            # Remove: remove a non-zero element
            if len(current_set) > 2:
                idx = random.randint(1, len(current_set) - 1)
                trial = current_set[:idx] + current_set[idx+1:]
                s = _score(trial)
                if s > current_score:
                    current_set = trial
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = list(current_set)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
    
    # Final polish: try to improve with more targeted moves
    if time.time() < deadline - 3:
        for _ in range(500):
            if time.time() > deadline - 3:
                break
            # Try removing each element and see if score improves
            improved = False
            for i in range(1, len(current_set)):
                if time.time() > deadline - 3:
                    break
                trial = current_set[:i] + current_set[i+1:]
                s = _score(trial)
                if s > current_score:
                    current_set = trial
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_set = list(current_set)
                    improved = True
                    break
            if not improved:
                break
    
    return best_set
