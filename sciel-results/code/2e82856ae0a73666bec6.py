"""Maximum weight independent set on the non-coprime conflict graph.

Nodes: integers 1..511, weight w(a) = 1/(512 - a).
Edges: pairs (a, b) with gcd(a, b) > 1.
Goal: maximize sum of weights of a pairwise-coprime subset.

Solved by branch-and-bound with a fractional-knapsack upper bound:
at a node with available set S, the bound is sum_{v in S} w(v)
(a valid upper bound for any independent subset of S).
"""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    N = 512
    n = N - 1
    # w[i] corresponds to a = i + 1
    w = [1.0 / (N - (i + 1)) for i in range(n)]
    
    # 1. Efficient neighbor construction using primes
    # Find all primes <= 511
    is_prime = [True] * (N)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N, i):
                is_prime[j] = False
    primes = [i for i in range(2, N) if is_prime[i]]
    
    # Map prime -> bitmask of indices i (a=i+1) such that p divides i+1
    prime_masks = {}
    for p in primes:
        mask = 0
        # a is multiple of p: a = p, 2p, ...
        # index i = a - 1
        for a in range(p, N, p):
            i = a - 1
            mask |= (1 << i)
        prime_masks[p] = mask
        
    # For each index i, neighbors[i] is the union of masks of all primes dividing i+1
    neighbors = [0] * n
    for i in range(n):
        a = i + 1
        mask = 0
        # Factorize a using primes
        temp = a
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= prime_masks[p]
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= prime_masks[temp]
        # Exclude self
        neighbors[i] = mask & ~(1 << i)
        
    # 2. Precompute chunk sums for fast sum_weights(mask)
    CHUNK_SIZE = 16
    num_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    # chunk_sums[c][m] = sum of weights for bits in m within chunk c
    # We only need to handle the last chunk partially, but for simplicity we pad w to num_chunks*CHUNK_SIZE
    w_padded = w + [0.0] * (num_chunks * CHUNK_SIZE - n)
    
    chunk_sums = []
    for c in range(num_chunks):
        base = c * CHUNK_SIZE
        # Precompute sum for all 16-bit values for this chunk
        # We can do this iteratively
        sums = [0.0] * (1 << CHUNK_SIZE)
        # sums[0] = 0
        for m in range(1, 1 << CHUNK_SIZE):
            # Find lowest set bit
            lsb = m & (-m)
            bit_idx = lsb.bit_length() - 1
            prev_m = m ^ lsb
            sums[m] = sums[prev_m] + w_padded[base + bit_idx]
        chunk_sums.append(sums)
        
    def sum_weights(mask):
        total = 0.0
        for c in range(num_chunks):
            chunk_val = (mask >> (c * CHUNK_SIZE)) & ((1 << CHUNK_SIZE) - 1)
            total += chunk_sums[c][chunk_val]
        return total

    # 3. Branch and Bound
    all_mask = (1 << n) - 1
    best_val = 0.0
    best_set_mask = 0
    
    # Initial greedy seed
    # Pick highest weight (highest index) if coprime to chosen
    chosen_mask = 0
    current_val = 0.0
    # We iterate from high index to low index
    for i in range(n - 1, -1, -1):
        if not (chosen_mask & neighbors[i]):
            chosen_mask |= (1 << i)
            current_val += w[i]
    best_val = current_val
    best_set_mask = chosen_mask
    
    start = time.time()
    TIME_LIMIT = 110.0
    
    # Stack: (available_mask, current_value, chosen_mask)
    stack = [(all_mask, 0.0, 0)]
    
    count = 0
    while stack:
        mask, val, chosen = stack.pop()
        
        # Bound check
        avail_w = sum_weights(mask)
        if val + avail_w <= best_val + 1e-15:
            continue
            
        if mask == 0:
            if val > best_val:
                best_val = val
                best_set_mask = chosen
            continue
            
        # Time check
        if count & 0x1FF == 0:
            if time.time() - start > TIME_LIMIT:
                break
        count += 1
        
        # Select branch vertex: highest index (highest weight) in mask
        v = mask.bit_length() - 1
        
        # Include branch
        # Remove v and its neighbors
        new_mask_inc = mask & ~(1 << v) & ~neighbors[v]
        new_val_inc = val + w[v]
        new_chosen_inc = chosen | (1 << v)
        stack.append((new_mask_inc, new_val_inc, new_chosen_inc))
        
        # Exclude branch
        new_mask_exc = mask & ~(1 << v)
        new_val_exc = val
        new_chosen_exc = chosen
        stack.append((new_mask_exc, new_val_exc, new_chosen_exc))
        
    # Convert best_set_mask to list of integers a
    result = []
    for i in range(n):
        if best_set_mask & (1 << i):
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
