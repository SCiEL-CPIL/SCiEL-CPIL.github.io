import sys
import time
from math import gcd

# Increase recursion limit for B&B
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 512
    # Candidates 1 to 511
    candidates = list(range(511, 0, -1))
    
    # 1. Precompute prime factors and masks
    # Primes < 512
    primes = []
    is_prime = [True] * 512
    is_prime[0] = is_prime[1] = False
    for i in range(2, 512):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 512, i):
                is_prime[j] = False
                
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    def get_mask(n):
        mask = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                if d in prime_to_bit:
                    mask |= (1 << prime_to_bit[d])
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            if temp in prime_to_bit:
                mask |= (1 << prime_to_bit[temp])
        return mask

    # 2. Prepare candidates sorted by weight descending
    # weight = 1 / (512 - a)
    # We want to maximize sum.
    
    # Create list of tuples (weight, a, mask)
    items = []
    for a in range(511, 0, -1):
        w = 1.0 / (N - a)
        m = get_mask(a)
        items.append((w, a, m))
        
    # Sort by weight descending. If weights are equal, larger a first? 
    # Weights are distinct for distinct a in 1..511 because 1/(512-a) is strictly monotonic.
    items.sort(key=lambda x: x[0], reverse=True)
    
    # Extract sorted lists
    sorted_weights = [x[0] for x in items]
    sorted_avalues = [x[1] for x in items]
    sorted_masks = [x[2] for x in items]
    
    # Precompute suffix sums of weights for upper bound
    n_items = len(items)
    suffix_sums = [0.0] * (n_items + 1)
    for i in range(n_items - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_weights[i]
        
    best_score = 0.0
    best_set = []
    
    # Greedy initial solution to set a good lower bound
    greedy_set = []
    greedy_mask = 0
    greedy_score = 0.0
    for i in range(n_items):
        if (sorted_masks[i] & greedy_mask) == 0:
            greedy_set.append(sorted_avalues[i])
            greedy_mask |= sorted_masks[i]
            greedy_score += sorted_weights[i]
            
    if greedy_score > best_score:
        best_score = greedy_score
        best_set = list(greedy_set)
        
    # Branch and Bound
    # To avoid deep recursion overhead and potential stack issues, we can use a stack, 
    # but recursion is cleaner and 511 depth is fine.
    
    def bb(idx, current_score, used_mask, current_set):
        nonlocal best_score, best_set
        
        # Check time
        if time.time() > deadline:
            return
        
        # Pruning: Upper bound
        # Max possible score from here is current_score + sum of all remaining weights
        # This is a valid upper bound, though loose.
        if current_score + suffix_sums[idx] <= best_score:
            return
            
        if idx == n_items:
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
            return
        
        a = sorted_avalues[idx]
        m = sorted_masks[idx]
        w = sorted_weights[idx]
        
        # Option 1: Include a
        if (m & used_mask) == 0:
            current_set.append(a)
            bb(idx + 1, current_score + w, used_mask | m, current_set)
            current_set.pop()
            
        # Option 2: Exclude a
        bb(idx + 1, current_score, used_mask, current_set)
        
    bb(0, 0.0, 0, [])
    
    return best_set
