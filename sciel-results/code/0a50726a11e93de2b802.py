"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute prime factors for 1..N using bitmasks for fast compatibility checks
    # Assign each prime a bit position
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, math.isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Map each prime to a bit index
    prime_bit = {p: i for i, p in enumerate(primes)}
    
    # Compute bitmask for each number 1..N
    num_mask = [0] * (N + 1)
    num_mask[1] = 0
    for p in primes:
        bit = 1 << prime_bit[p]
        for j in range(p, N + 1, p):
            num_mask[j] |= bit
    
    # For each prime, find the largest power p^k <= N
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    # Initial solution: 1 + all prime powers
    initial_set = [1] + list(prime_powers.values())
    
    # Verify pairwise coprime (sanity check using bitmasks)
    for i in range(len(initial_set)):
        for j in range(i + 1, len(initial_set)):
            if num_mask[initial_set[i]] & num_mask[initial_set[j]] != 0:
                return [1] + list(prime_powers.values())
    
    best_sum = sum(initial_set)
    best_set = list(initial_set)
    
    # Build candidate list: all numbers not in initial set, sorted descending by value
    in_set = set(initial_set)
    candidates = [x for x in range(N, 0, -1) if x not in in_set]
    
    def is_compatible_with_set_mask(num, current_set):
        """Check if num is coprime with all elements in current_set using bitmasks."""
        m = num_mask[num]
        for x in current_set:
            if m & num_mask[x]:
                return False
        return True
    
    # Local search: 1-for-1 swaps
    improved = True
    while improved and time.time() < deadline - 8:
        improved = False
        current_set = list(best_set)
        current_sum = best_sum
        
        for i in range(len(current_set)):
            if time.time() > deadline - 8:
                break
            removed = current_set[i]
            remaining = [x for j, x in enumerate(current_set) if j != i]
            
            # Try adding each candidate (sorted desc, so first compatible high-value one is best)
            for cand in candidates:
                if cand <= removed:
                    continue  # No improvement possible
                if is_compatible_with_set_mask(cand, remaining):
                    new_sum = current_sum - removed + cand
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_set = remaining + [cand]
                        improved = True
                        break  # Found improvement, restart outer loop
    
    # Local search: 2-for-1 swaps (remove two small, add one large)
    if time.time() < deadline - 15:
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            current_set = list(best_set)
            current_sum = best_sum
            
            for i in range(len(current_set)):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, len(current_set)):
                    if time.time() > deadline - 15:
                        break
                    removed1 = current_set[i]
                    removed2 = current_set[j]
                    remaining = [x for k, x in enumerate(current_set) if k != i and k != j]
                    
                    # Try adding a single candidate that's larger than the sum of removed
                    for cand in candidates:
                        if cand <= removed1 + removed2:
                            continue
                        if is_compatible_with_set_mask(cand, remaining):
                            new_sum = current_sum - removed1 - removed2 + cand
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_set = remaining + [cand]
                                improved = True
                                break
    
    # Local search: 1-for-2 swaps (remove one large, add two smaller but summing higher)
    if time.time() < deadline - 25:
        improved = True
        while improved and time.time() < deadline - 25:
            improved = False
            current_set = list(best_set)
            current_sum = best_sum
            
            for i in range(len(current_set)):
                if time.time() > deadline - 25:
                    break
                removed = current_set[i]
                remaining = [x for j, x in enumerate(current_set) if j != i]
                
                # Try adding pairs of candidates
                for a_idx in range(len(candidates)):
                    if time.time() > deadline - 25:
                        break
                    cand1 = candidates[a_idx]
                    if cand1 >= removed:
                        continue
                    if not is_compatible_with_set_mask(cand1, remaining):
                        continue
                    
                    for b_idx in range(a_idx + 1, len(candidates)):
                        cand2 = candidates[b_idx]
                        if cand1 + cand2 <= removed:
                            continue
                        if not is_compatible_with_set_mask(cand2, remaining):
                            continue
                        if num_mask[cand1] & num_mask[cand2]:
                            continue
                        new_sum = current_sum - removed + cand1 + cand2
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_set = remaining + [cand1, cand2]
                            improved = True
                            break
    
    # Final polish: another pass of 1-for-1 to ensure local optimality
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            current_set = list(best_set)
            current_sum = best_sum
            
            for i in range(len(current_set)):
                if time.time() > deadline - 3:
                    break
                removed = current_set[i]
                remaining = [x for j, x in enumerate(current_set) if j != i]
                
                for cand in candidates:
                    if cand <= removed:
                        continue
                    if is_compatible_with_set_mask(cand, remaining):
                        new_sum = current_sum - removed + cand
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_set = remaining + [cand]
                            improved = True
                            break
    
    # Convert to list of ints
    result = list(best_set)
    result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
