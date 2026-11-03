import time
import random
import itertools
from math import log
import numpy as np

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
    # sums_arr = U_arr[:, None] + U_arr[None, :]
    # unique_sums = np.unique(sums_arr)
    
    # Compute all pairwise differences
    # diffs_arr = U_arr[:, None] - U_arr[None, :]
    # unique_diffs = np.unique(diffs_arr)
    
    # To save time/memory, we can be smarter, but for n~10-20 this is fine.
    # Let's stick to the standard approach but ensure it's efficient.
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

def local_search(start_set, deadline, max_iter=5000):
    current = sorted(set(start_set))
    if 0 not in current:
        current = sorted(set(current + [0]))
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    # Simple simulated annealing / hill climbing
    T = 0.05
    cooling = 0.99
    
    while time.time() < deadline - 2 and max_iter > 0:
        max_iter -= 1
        if len(current) <= 1:
            break
            
        # Randomly choose an operation: move, add, remove
        op = random.random()
        
        if op < 0.7:
            # Move an element
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            
            if random.random() < 0.5:
                # Small perturbation
                delta = random.randint(-3, 3)
                new_val = old_val + delta
            else:
                # Jump
                new_val = random.randint(1, current[-1] + 10)
                
            if new_val <= 0:
                continue
                
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
            if len(new_set) != len(current):
                # Collision, skip or handle? Usually skip if we want fixed size
                continue
                
        elif op < 0.85:
            # Add an element
            if len(current) >= 15: # Cap size to keep search fast
                continue
            new_val = random.randint(1, current[-1] + 20)
            if new_val in current:
                continue
            new_set = sorted(current + [new_val])
            
        else:
            # Remove an element (not 0)
            if len(current) <= 4:
                continue
            i = random.randint(1, len(current) - 1)
            new_set = current[:i] + current[i+1:]
            
        new_score = score_set(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best_set = new_set[:]
            
        # Accept if better or with probability
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
            else:
                prob = 1.0
                
            if random.random() < prob:
                current = new_set
                current_score = new_score
                
        T *= cooling
        
    return best_set, best_score

def construct():
    deadline = time.time() + 105  # Safety margin for 120s limit
    
    best_set = [0, 1, 4]
    best_score = score_set(best_set)
    
    # Seeds: mix of known good sets and random starts
    seeds = [
        [0, 1, 3],
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 8, 13],
        [0, 2, 3, 7, 12],
        [0, 1, 4, 11, 13],
        [0, 1, 4, 9, 13],
        [0, 1, 4],
        [0, 1, 4, 16],
        [0, 1, 3, 9, 27],
    ]
    
    # Add some random seeds
    random.seed(42)
    for _ in range(10):
        n = random.randint(4, 8)
        max_val = random.randint(20, 80)
        U = [0]
        other_elements = random.sample(range(1, max_val), n - 1)
        U.extend(other_elements)
        seeds.append(sorted(U))
        
    # Local search from each seed
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        local_best, local_score = local_search(seed, deadline, max_iter=3000)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
            
    return best_set
