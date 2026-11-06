def construct():
    import time
    import random
    from math import gcd # Not needed if using masks, but good for verification if needed. We will use masks.
    
    start_time = time.time()
    deadline = start_time + 115
    
    N = 511
    weights = [1.0 / (N + 1 - a) for a in range(1, N + 1)]
    
    # 1. Precompute prime factors as bitmasks
    # Primes below 512
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
            
    prime_index = {p: i for i, p in enumerate(primes)}
    masks = [0] * (N + 1)
    for a in range(1, N + 1):
        m = a
        mask = 0
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask |= (1 << prime_index[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            mask |= (1 << prime_index[m])
        masks[a] = mask
        
    sorted_indices = list(range(1, N + 1))
    sorted_indices.sort(key=lambda x: weights[x-1], reverse=True)
    
    def greedy_fill(current_mask, current_set, current_score):
        # current_set is a set of chosen numbers
        # current_mask is the OR of their masks
        # current_score is the sum of weights
        # Returns (new_set, new_mask, new_score)
        new_set = set(current_set)
        new_mask = current_mask
        new_score = current_score
        for b in sorted_indices:
            if b in new_set:
                continue
            if masks[b] & new_mask == 0:
                new_set.add(b)
                new_mask |= masks[b]
                new_score += weights[b-1]
        return new_set, new_mask, new_score

    # Initial Solution
    best_set = set()
    best_mask = 0
    best_score = 0.0
    best_set, best_mask, best_score = greedy_fill(0, set(), 0.0)
    
    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try 1-removal + greedy fill
        for a in list(best_set):
            if time.time() > deadline:
                break
            
            # Remove a
            temp_set = set(best_set)
            temp_set.remove(a)
            # Recalculate mask? 
            # We can't just remove bit from best_mask if another number has same prime.
            # So we must recalculate mask from temp_set.
            # This is O(K) where K is size of set. K is small (< 100).
            temp_mask = 0
            for x in temp_set:
                temp_mask |= masks[x]
            temp_score = best_score - weights[a-1]
            
            # Greedy Fill
            new_set, new_mask, new_score = greedy_fill(temp_mask, temp_set, temp_score)
            
            if new_score > best_score:
                best_set = new_set
                best_mask = new_mask
                best_score = new_score
                improved = True
                
        # Optional: Try 2-removals if time permits?
        # Or Randomized Perturbation
        if time.time() < deadline - 10:
             # Perturb: remove 2 random elements, refill
             if len(best_set) >= 2:
                 items = list(best_set)
                 i, j = random.sample(range(len(items)), 2)
                 a, b = items[i], items[j]
                 temp_set = set(best_set)
                 temp_set.remove(a)
                 temp_set.remove(b)
                 temp_mask = 0
                 for x in temp_set:
                     temp_mask |= masks[x]
                 temp_score = best_score - weights[a-1] - weights[b-1]
                 
                 new_set, new_mask, new_score = greedy_fill(temp_mask, temp_set, temp_score)
                 
                 if new_score > best_score:
                     best_set = new_set
                     best_mask = new_mask
                     best_score = new_score
                     improved = True

    return list(best_set)
