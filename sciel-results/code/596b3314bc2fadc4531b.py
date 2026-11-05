"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import random

def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0: sieve[0] = False
    if limit >= 1: sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    primes = _primes(N)
    
    # Precompute masks for small primes to speed up counting and removal
    # We use bitmask representation for S
    # S is an integer where bit i is 1 if i is in the set
    MAX_PRECOMP_P = 200
    precomputed_masks = {}
    for p in primes:
        if p > MAX_PRECOMP_P:
            break
        masks_for_p = []
        for r in range(p):
            mask = 0
            # Set bits for x in [0, N] such that x % p == r
            for x in range(r, N + 1, p):
                mask |= (1 << x)
            masks_for_p.append(mask)
        precomputed_masks[p] = masks_for_p
        
    def run_greedy(tie_breaker, prime_order='ascending'):
        """
        Run the greedy removal algorithm.
        tie_breaker: 'low', 'high', or 'random'
        prime_order: 'ascending' or 'descending'
        Returns the final set size and the set itself (as bitmask).
        """
        S = (1 << (N + 1)) - 1  # All bits 0 to N set
        
        # Determine prime processing order
        if prime_order == 'ascending':
            primes_to_process = primes
        else:
            primes_to_process = primes[::-1]
            
        for p in primes_to_process:
            if time.time() > deadline:
                break
            
            current_len = S.bit_count()
            if p > current_len:
                # If p > len(S), condition is automatically satisfied
                # because we can't cover p residues with < p elements
                break
                
            # Get counts for each residue
            if p in precomputed_masks:
                masks = precomputed_masks[p]
                counts = [0] * p
                for r in range(p):
                    counts[r] = (S & masks[r]).bit_count()
            else:
                # For larger p, count by iterating set bits
                counts = [0] * p
                x = S
                while x:
                    lsb = x & -x
                    idx = lsb.bit_length() - 1
                    counts[idx % p] += 1
                    x ^= lsb
                    
            # Check if any residue is missing
            min_count = min(counts)
            if min_count == 0:
                continue
                
            # Find target residue based on tie_breaker
            if tie_breaker == 'low':
                target_residue = counts.index(min_count)
            elif tie_breaker == 'high':
                # Find last occurrence of min_count
                for r in range(p - 1, -1, -1):
                    if counts[r] == min_count:
                        target_residue = r
                        break
            else: # random
                candidates = [r for r in range(p) if counts[r] == min_count]
                target_residue = random.choice(candidates)
                
            # Remove elements in target_residue
            if p in precomputed_masks:
                mask_to_remove = precomputed_masks[p][target_residue]
            else:
                # Construct mask on the fly
                mask_to_remove = 0
                for x in range(target_residue, N + 1, p):
                    mask_to_remove |= (1 << x)
            
            S &= ~mask_to_remove
            
        return S.bit_count(), S

    best_size = -1
    best_S = 0
    
    # Try different combinations of tie-breaking and prime order
    strategies = [
        ('low', 'ascending'),
        ('high', 'ascending'),
        ('low', 'descending'),
        ('high', 'descending'),
    ]
    
    # Add random strategies with different orders
    for _ in range(10):
        strategies.append(('random', 'ascending'))
    for _ in range(10):
        strategies.append(('random', 'descending'))
        
    for strat, order in strategies:
        if time.time() > deadline - 15: # Leave 15s for safety
            break
            
        size, S_mask = run_greedy(strat, order)
        
        if size > best_size:
            best_size = size
            best_S = S_mask
            
    # Convert best_S bitmask to list of integers
    result = []
    x = best_S
    while x:
        lsb = x & -x
        idx = lsb.bit_length() - 1
        result.append(idx)
        x ^= lsb
        
    return result

# EVOLVE-BLOCK-END
