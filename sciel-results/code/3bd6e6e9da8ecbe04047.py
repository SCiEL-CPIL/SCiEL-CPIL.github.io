"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    num_candidates = n - 1  # 255 numbers
    
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask where bit j is set if candidates[i] and candidates[j] are INCOMPATIBLE.
    # We use Python integers as arbitrary precision bitmasks.
    conflict_masks = [0] * num_candidates
    
    # Helper to check compatibility of x, y
    # Returns True if compatible, False if incompatible
    def are_compatible(x, y):
        gx = math.gcd(x, y)
        h1 = x // gx
        h2 = y // gx
        
        # Check if x | y*z for some z in [2, 256] \ {x, y}
        if h1 > 1:
            # Multiples of h1 in [2, 256]
            # Count is 256 // h1
            # We need to exclude x and y if they are multiples of h1.
            # x is always a multiple of h1.
            # y is a multiple of h1 if y % h1 == 0.
            count = 256 // h1
            count -= 1 # x
            if y % h1 == 0:
                count -= 1
            if count > 0:
                return False
        
        # Check if y | x*z for some z in [2, 256] \ {x, y}
        if h2 > 1:
            count = 256 // h2
            count -= 1 # y
            if x % h2 == 0:
                count -= 1
            if count > 0:
                return False
        
        return True

    # Build conflict masks
    for i in range(num_candidates):
        x = candidates[i]
        for j in range(num_candidates):
            if i == j:
                continue
            y = candidates[j]
            if not are_compatible(x, y):
                conflict_masks[i] |= (1 << j)
                conflict_masks[j] |= (1 << i) # Symmetric

    # Greedy construction using bitmasks
    def greedy_construct(order):
        S_mask = 0
        S_indices = []
        for idx in order:
            if (conflict_masks[idx] & S_mask) == 0:
                S_mask |= (1 << idx)
                S_indices.append(idx)
        return S_indices

    # Local search using bitmasks
    def local_search(S_indices):
        S_mask = 0
        for idx in S_indices:
            S_mask |= (1 << idx)
        
        improved = True
        while improved:
            improved = False
            # Try to add any available element
            # Iterate through all candidates not in S
            # To speed this up, we could maintain a list of available candidates,
            # but with N=255, a simple loop with O(1) check is fast enough.
            for idx in range(num_candidates):
                if (S_mask & (1 << idx)) == 0:
                    if (conflict_masks[idx] & S_mask) == 0:
                        S_indices.append(idx)
                        S_mask |= (1 << idx)
                        improved = True
                        break
            
            if improved:
                continue

            # If no direct add, try 1-for-1 swaps
            # Remove a, add b.
            # b must be compatible with S \ {a}.
            # (conflict_masks[b] & (S_mask ^ (1 << a))) == 0
            
            # Try a few random swaps
            # We pick a random a from S, and try to find a valid b.
            # To avoid O(N) scan for b every time, we can just scan, it's 255 items.
            for _ in range(20):
                if not S_indices:
                    break
                a = random.choice(S_indices)
                a_mask = 1 << a
                temp_mask = S_mask ^ a_mask
                
                # Find b not in S such that compatible with temp_mask
                found_b = False
                # Randomize order of b candidates to escape local minima
                b_order = list(range(num_candidates))
                random.shuffle(b_order)
                for b in b_order:
                    if (S_mask & (1 << b)) == 0:
                        if (conflict_masks[b] & temp_mask) == 0:
                            # Swap a with b
                            S_indices.remove(a)
                            S_indices.append(b)
                            S_mask = temp_mask | (1 << b)
                            improved = True
                            found_b = True
                            break
                if found_b:
                    break
        
        return S_indices

    best_size = 0
    best_set = []
    
    # Baseline: Primes
    primes = [i for i in range(2, n+1) if all(i % d != 0 for d in range(2, int(math.isqrt(i)) + 1))]
    prime_indices = [candidates.index(p) for p in primes]
    best_set = greedy_construct(prime_indices)
    best_size = len(best_set)
    
    # Random restarts
    num_restarts = 200 # Increased restarts due to faster operations
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomize order
        order = list(range(num_candidates))
        random.shuffle(order)
        
        # Greedy
        S = greedy_construct(order)
        
        # Local Search
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    # Final Polish: Try to add any remaining elements to best_set
    best_mask = 0
    for idx in best_set:
        best_mask |= (1 << idx)
        
    for idx in range(num_candidates):
        if (best_mask & (1 << idx)) == 0:
            if (conflict_masks[idx] & best_mask) == 0:
                best_set.append(idx)
                best_mask |= (1 << idx)
    
    # Convert indices to values
    result = [candidates[i] for i in best_set]
    
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
