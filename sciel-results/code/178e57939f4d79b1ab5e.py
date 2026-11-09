import time
import random
import math

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    # Primes powers: 2^4, 3^3, 5^2, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    L = 1
    L *= 16
    L *= 27
    L *= 25
    L *= 7
    L *= 11
    L *= 13
    L *= 17
    L *= 19
    L *= 23
    L *= 29
    L *= 31
    L *= 37
    L *= 41
    L *= 43
    L *= 47
    
    weights = [L // d for d in denominators]
    target = L - 1
    
    start_time = time.time()
    time_limit = 55.0 # 5s margin for 60s limit
    
    best_sum = 0
    best_set = set()
    
    all_indices = list(range(n))
    
    def try_improve(curr_set, curr_sum):
        nonlocal best_sum, best_set
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            curr_set_list = list(curr_set)
            
            # 1. Try adding items
            # Iterate through all items not in set
            for j in all_indices:
                if j not in curr_set:
                    new_sum = curr_sum + weights[j]
                    if new_sum <= target:
                        curr_sum = new_sum
                        curr_set.add(j)
                        improved = True
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_set = set(curr_set)
            
            # 2. Try 1-for-1 swaps
            # For each i in set, for each j not in set
            # If w_j > w_i and new_sum <= target, swap
            # We need to re-fetch list as set might have changed
            curr_set_list = list(curr_set)
            for i in curr_set_list:
                for j in all_indices:
                    if j not in curr_set:
                        if weights[j] > weights[i]:
                            new_sum = curr_sum - weights[i] + weights[j]
                            if new_sum <= target:
                                curr_set.remove(i)
                                curr_set.add(j)
                                curr_sum = new_sum
                                improved = True
                                if curr_sum > best_sum:
                                    best_sum = curr_sum
                                    best_set = set(curr_set)
                                break # Restart inner loop for this i? 
                                       # Breaking j loop is fine. 
                                       # We continue to next i. 
                                       # But curr_sum changed, so other swaps might be valid/invalid.
                                       # The outer 'while improved' loop handles re-evaluation.
            
            # 3. Try 1-for-2 swaps
            # Remove i, add j, k
            # curr_sum - w_i + w_j + w_k > curr_sum => w_j + w_k > w_i
            # curr_sum - w_i + w_j + w_k <= target
            curr_set_list = list(curr_set)
            for i in curr_set_list:
                if time.time() - start_time > time_limit:
                    break
                for j in all_indices:
                    if j in curr_set:
                        continue
                    # Optimization: if w_j <= w_i, we still might find k such that w_j+w_k > w_i
                    # But w_j + w_k > w_i implies at least one is larger or both are significant.
                    # Just iterate all j, k.
                    # To speed up, sort remaining indices? 
                    # N=47, 47*47*47 ~ 100k ops. 
                    for k in all_indices:
                        if k in curr_set or k == j:
                            continue
                        new_sum = curr_sum - weights[i] + weights[j] + weights[k]
                        if new_sum > curr_sum and new_sum <= target:
                            curr_set.remove(i)
                            curr_set.add(j)
                            curr_set.add(k)
                            curr_sum = new_sum
                            improved = True
                            if curr_sum > best_sum:
                                best_sum = curr_sum
                                best_set = set(curr_set)
                            break # Break k loop
                    if improved:
                        break # Break j loop if we found a swap for i
            
            # 4. Try 2-for-1 swaps
            # Remove i, j, add k
            # curr_sum - w_i - w_j + w_k > curr_sum => w_k > w_i + w_j
            # curr_sum - w_i - w_j + w_k <= target
            curr_set_list = list(curr_set)
            for idx_i in range(len(curr_set_list)):
                i = curr_set_list[idx_i]
                if time.time() - start_time > time_limit:
                    break
                for idx_j in range(idx_i + 1, len(curr_set_list)):
                    j = curr_set_list[idx_j]
                    for k in all_indices:
                        if k in curr_set:
                            continue
                        if weights[k] > weights[i] + weights[j]:
                            new_sum = curr_sum - weights[i] - weights[j] + weights[k]
                            if new_sum <= target:
                                curr_set.remove(i)
                                curr_set.remove(j)
                                curr_set.add(k)
                                curr_sum = new_sum
                                improved = True
                                if curr_sum > best_sum:
                                    best_sum = curr_sum
                                    best_set = set(curr_set)
                                break # Break k loop
                    if improved:
                        break # Break j loop
                if improved:
                    break # Break i loop
    
    # 1. Greedy Initialization
    indices_sorted = sorted(all_indices, key=lambda i: weights[i], reverse=True)
    curr_set = set()
    curr_sum = 0
    for i in indices_sorted:
        if curr_sum + weights[i] <= target:
            curr_sum += weights[i]
            curr_set.add(i)
    
    if curr_sum > best_sum:
        best_sum = curr_sum
        best_set = set(curr_set)
    
    try_improve(curr_set, curr_sum)
    
    # 2. Random Restarts
    # We perform random restarts with different initial sets
    # Strategy: Start with random subset, then improve
    while time.time() - start_time < time_limit:
        curr_set = set()
        curr_sum = 0
        
        # Randomly select items to start with
        # Sometimes start with large weights, sometimes small
        mode = random.random()
        if mode < 0.5:
            # Greedy on random order
            rand_indices = all_indices.copy()
            random.shuffle(rand_indices)
            for i in rand_indices:
                if curr_sum + weights[i] <= target:
                    curr_sum += weights[i]
                    curr_set.add(i)
        elif mode < 0.8:
            # Start with top k random large weights
            top_weights = sorted(all_indices, key=lambda i: weights[i], reverse=True)
            k = random.randint(5, 20)
            candidates = top_weights[:k*2] # Pick from top half
            random.shuffle(candidates)
            for i in candidates:
                if curr_sum + weights[i] <= target:
                    curr_sum += weights[i]
                    curr_set.add(i)
                    if len(curr_set) >= k:
                        break
        else:
            # Start with a random subset of size 10-20
            k = random.randint(10, 20)
            random_subset = random.sample(all_indices, k)
            # Sort by weight descending to try to pack tightly
            random_subset.sort(key=lambda i: weights[i], reverse=True)
            for i in random_subset:
                if curr_sum + weights[i] <= target:
                    curr_sum += weights[i]
                    curr_set.add(i)
        
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_set = set(curr_set)
        
        try_improve(curr_set, curr_sum)
        
        # Occasionally re-improve the best solution found so far
        # to see if we can squeeze more out of it with new randomization?
        # No, try_improve is deterministic given the set. 
        # But we can try a "shake" on the best solution:
        # Randomly remove 2 items and add 2 different ones, then improve.
        if random.random() < 0.1 and len(best_set) > 4:
            curr_set = set(best_set)
            curr_sum = best_sum
            # Shake
            items_to_remove = random.sample(list(curr_set), 2)
            for i in items_to_remove:
                curr_set.remove(i)
                curr_sum -= weights[i]
            # Try to add back or add new
            candidates = [i for i in all_indices if i not in curr_set]
            random.shuffle(candidates)
            for i in candidates:
                if curr_sum + weights[i] <= target:
                    curr_sum += weights[i]
                    curr_set.add(i)
            try_improve(curr_set, curr_sum)

    return list(best_set)
