import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 256
    candidates = list(range(1, N + 1))
    n_cand = N
    
    # Precompute conflicts as bitsets
    conflicts = [0] * n_cand
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if gcd(candidates[i], candidates[j]) > 1:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Greedy: sort by value descending
    order = sorted(range(n_cand), key=lambda x: candidates[x], reverse=True)
    
    def greedy_init(ord):
        selected = set()
        blocked = 0
        for idx in ord:
            if not (blocked >> idx) & 1:
                selected.add(idx)
                blocked |= (1 << idx)
                blocked |= conflicts[idx]
        return selected
    
    best_set = greedy_init(order)
    best_sum = sum(candidates[i] for i in best_set)
    
    # Local search
    def local_search(initial_set, max_iter=500):
        current_set = set(initial_set)
        current_sum = sum(candidates[i] for i in current_set)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline:
            improved = False
            iters += 1
            current_mask = 0
            for i in current_set:
                current_mask |= (1 << i)
            
            # Try adding a single item
            for idx in range(n_cand):
                if idx in current_set:
                    continue
                if (conflicts[idx] & current_mask) == 0:
                    current_set.add(idx)
                    current_sum += candidates[idx]
                    current_mask |= (1 << idx)
                    improved = True
            
            # Try 1-for-1 swaps
            for idx_out in list(current_set):
                mask_without_out = current_mask & ~(1 << idx_out)
                for idx_in in range(n_cand):
                    if idx_in in current_set or idx_in == idx_out:
                        continue
                    if (conflicts[idx_in] & mask_without_out) == 0:
                        if candidates[idx_in] > candidates[idx_out]:
                            current_set.remove(idx_out)
                            current_set.add(idx_in)
                            current_sum += candidates[idx_in] - candidates[idx_out]
                            current_mask = (current_mask & ~(1 << idx_out)) | (1 << idx_in)
                            improved = True
                            
        return current_set, current_sum
    
    best_set, best_sum = local_search(best_set)
    
    # Multiple restarts with different orderings
    rng = random.Random(42)
    for restart in range(30):
        if time.time() > deadline:
            break
        # Create a perturbed ordering
        shuffled_order = list(range(n_cand))
        rng.shuffle(shuffled_order)
        # Sort by value with some randomness
        shuffled_order.sort(key=lambda x: candidates[x] + rng.random() * 80, reverse=True)
        
        trial_set = greedy_init(shuffled_order)
        trial_set, trial_sum = local_search(trial_set, max_iter=200)
        
        if trial_sum > best_sum:
            best_sum = trial_sum
            best_set = trial_set
    
    result = [candidates[i] for i in best_set]
    return [int(x) for x in result]
