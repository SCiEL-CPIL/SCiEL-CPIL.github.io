import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Generate first 16 primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes_list = sieve(100)
    # First 15 primes
    base_primes = primes_list[:15]
    
    # Check if product fits
    prod = 1
    for p in base_primes:
        prod *= p
    if prod > 10**18:
        base_primes = base_primes[:14]

    k = len(base_primes)
    num_divs = 1 << k

    # Precompute: for each prime index, which other primes are coprime (all are, since they're distinct primes)
    # But the key insight: two divisors are coprime iff their prime support sets are disjoint
    # We can precompute the "coprime mask" for each subset
    
    # Faster scoring: build divisors once, then for a given permutation, just compute the sorted order
    # and count disjoint mask pairs
    
    # Precompute all 2^k subset masks and their values
    # For each subset (bitmask), store the product
    subset_products = np.zeros(num_divs, dtype=np.int64)
    subset_products[0] = 1
    for mask in range(1, num_divs):
        # Find the lowest set bit
        lowbit = mask & (-mask)
        idx = lowbit.bit_length() - 1
        prev_mask = mask ^ lowbit
        subset_products[mask] = subset_products[prev_mask] * base_primes[idx]
    
    # Precompute: for each pair of subsets, whether they are coprime (disjoint)
    # coprime_mask[i][j] = 1 if subset i and subset j are disjoint
    # But we only need this for adjacent pairs in sorted order, so we'll compute on the fly
    
    # Function to calculate score for a given permutation of primes
    # The permutation determines which primes go to which bit positions
    # So we need to remap the subset products
    def calc_score_fast(primes_perm):
        # Build a mapping: old_bit_position -> new_prime
        # Actually, the permutation tells us which prime is at each position
        # So for a subset mask, the product is: prod of primes_perm[i] for each bit i set in mask
        # We can precompute this for all masks
        
        # Build products for current permutation
        products = np.zeros(num_divs, dtype=np.int64)
        products[0] = 1
        for mask in range(1, num_divs):
            lowbit = mask & (-mask)
            idx = lowbit.bit_length() - 1
            prev_mask = mask ^ lowbit
            products[mask] = products[prev_mask] * primes_perm[idx]
        
        # Sort by product value
        sort_idx = np.argsort(products, kind='stable')
        sorted_masks = np.array([sort_idx[i] for i in range(num_divs)], dtype=np.int32)
        
        # Count adjacent pairs that are coprime (disjoint masks)
        if num_divs > 1:
            # For each adjacent pair, check if masks are disjoint
            left_masks = sorted_masks[:-1]
            right_masks = sorted_masks[1:]
            # Disjoint means bitwise AND is 0
            disjoint = (left_masks & right_masks) == 0
            score = np.sum(disjoint)
        else:
            score = 0
        return int(score)
    
    # Greedy set-packing heuristic for initial permutation
    # Idea: try to place primes in an order that maximizes coprime adjacencies
    # Start with a random permutation, then greedily improve
    def greedy_init():
        # Start with sorted primes
        perm = list(base_primes)
        current_score = calc_score_fast(perm)
        
        # Try to improve by swapping
        improved = True
        while improved:
            improved = False
            for i in range(k):
                if time.time() > deadline:
                    return perm, current_score
                for j in range(i + 1, k):
                    perm[i], perm[j] = perm[j], perm[i]
                    s = calc_score_fast(perm)
                    if s > current_score:
                        current_score = s
                        improved = True
                    else:
                        perm[i], perm[j] = perm[j], perm[i]
        return perm, current_score
    
    # Initial state using greedy
    current_primes, current_score = greedy_init()
    best_primes = list(current_primes)
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.9995  # Slower cooling
    step_count = 0
    
    # Main search loop
    while time.time() < deadline:
        # Try all pairs systematically
        for i in range(k):
            if time.time() > deadline:
                break
            for j in range(i + 1, k):
                if time.time() > deadline:
                    break
                
                current_primes[i], current_primes[j] = current_primes[j], current_primes[i]
                s = calc_score_fast(current_primes)
                
                if s > current_score:
                    current_score = s
                    # Accept
                else:
                    # Simulated Annealing
                    delta = current_score - s
                    if delta > 0 and random.random() < math.exp(-delta / max(T, T_min)):
                        current_score = s
                        # Accept worse move
                    else:
                        # Revert
                        current_primes[i], current_primes[j] = current_primes[j], current_primes[i]
                
                step_count += 1
                if step_count % 500 == 0:
                    T *= alpha
                    if T < T_min:
                        T = T_min
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_primes = list(current_primes)
        
        # If we've done a full pass, consider restarting from best with perturbation
        if step_count % (k * (k - 1) // 2) == 0:
            # Perturb and continue
            current_primes = list(best_primes)
            current_score = best_score
            # Apply 2-3 random swaps
            for _ in range(random.randint(2, 4)):
                a, b = random.sample(range(k), 2)
                current_primes[a], current_primes[b] = current_primes[b], current_primes[a]
            current_score = calc_score_fast(current_primes)
            T = max(T, 0.5)  # Reheat slightly
    
    # Final calculation of n
    n = 1
    for p in best_primes:
        n *= p
        
    return n
