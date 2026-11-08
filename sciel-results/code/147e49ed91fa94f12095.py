import time
from math import log
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_set = [0, 1]
    best_score = score_set(best_set)

    # Generate diverse seeds
    seeds = []
    # Mersenne-like
    for k in range(3, 8):
        seeds.append([2**i - 1 for i in range(k+1)])
    # Perturbed Mersenne
    for k in range(3, 8):
        base = [2**i - 1 for i in range(k+1)]
        for _ in range(5):
            pert = base[:]
            if len(pert) > 1:
                idx = random.randint(1, len(pert)-1)
                pert[idx] += random.randint(-5, 5)
                if pert[idx] > 0 and pert[idx] not in pert[:idx] and pert[idx] not in pert[idx+1:]:
                    seeds.append(sorted(pert))
    # Random sets
    for _ in range(20):
        n = random.randint(4, 10)
        max_val = random.randint(20, 50)
        U = [0] + sorted(random.sample(range(1, max_val+1), n-1))
        seeds.append(U)

    # Evaluate seeds
    for cand in seeds:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Local Search
    def local_search(U, deadline):
        U = sorted(set(U))
        current_score = score_set(U)
        best_score_local = current_score
        best_set_local = list(U)
        
        T = 1.0
        cooling = 0.99
        max_iterations = 20000
        
        for i in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            new_U = list(U)
            move_type = random.random()
            
            if move_type < 0.5:
                # Replace
                if len(new_U) < 2:
                    continue
                idx = random.randint(1, len(new_U) - 1)
                old_val = new_U[idx]
                new_val = old_val + random.randint(-10, 10)
                if new_val <= 0 or new_val in U:
                    continue
                new_U[idx] = new_val
                new_U = sorted(set(new_U))
            elif move_type < 0.8:
                # Add
                if len(new_U) < 15:
                    max_val = max(new_U)
                    new_val = random.randint(max(1, max_val-10), max_val+10)
                    if new_val in U:
                        continue
                    new_U.append(new_val)
                    new_U = sorted(set(new_U))
            else:
                # Remove
                if len(new_U) > 3:
                    idx = random.randint(1, len(new_U) - 1)
                    new_U.pop(idx)
                    new_U = sorted(set(new_U))
            
            if len(new_U) < 2:
                continue
                
            new_score = score_set(new_U)
            
            if new_score > current_score:
                U = new_U
                current_score = new_score
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_set_local = list(U)
            else:
                delta = new_score - current_score
                if random.random() < np.exp(delta / T):
                    U = new_U
                    current_score = new_score
                    
            T *= cooling
            
        return best_score_local, best_set_local

    # Run local search from best seed and top seeds
    search_seeds = [best_set]
    seed_scores = [(score_set(s), s) for s in seeds]
    seed_scores.sort(reverse=True)
    for s, cand in seed_scores[:5]:
        if cand != best_set:
            search_seeds.append(cand)

    for seed in search_seeds:
        if time.time() > deadline - 10:
            break
        local_score, local_set = local_search(seed, deadline)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
