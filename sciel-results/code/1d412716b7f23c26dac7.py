import time
import math

N = 5000
DEADLINE = 315.0  # Safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

def _count_rough_numbers(N, primes):
    """Count numbers in [0, N] not divisible by any prime in primes."""
    if not primes:
        return N + 1
    # Use inclusion-exclusion for small number of primes
    # For larger sets, use a simple sieve
    marked = bytearray(N + 1)
    for p in primes:
        for multiple in range(0, N + 1, p):
            marked[multiple] = 1
    return sum(1 for i in range(N + 1) if not marked[i])

def construct():
    start_time = time.time()
    
    all_primes = _primes(N)
    
    best_set = []
    best_size = 0
    
    # Try different K values (largest prime threshold)
    # We want to find K such that the rough number count is approximately consistent
    # Let's try K values from small to large
    
    for idx, K in enumerate(all_primes):
        if time.time() - start_time > DEADLINE:
            break
        
        # Get primes up to K
        primes_up_to_K = [p for p in all_primes if p <= K]
        
        # Build the set of numbers in [0, N] not divisible by any prime in primes_up_to_K
        # This misses residue 0 mod p for each p in primes_up_to_K
        marked = bytearray(N + 1)
        for p in primes_up_to_K:
            for multiple in range(0, N + 1, p):
                marked[multiple] = 1
        
        S = set(i for i in range(N + 1) if not marked[i])
        current_size = len(S)
        
        # For this to be admissible, we need K >= current_size
        # (all primes p <= current_size must be in primes_up_to_K)
        if K < current_size:
            # Need to extend: find the largest prime <= current_size
            # But if current_size > K, we have primes between K and current_size not handled
            # We need to add those primes to the removal set
            needed_primes = [p for p in all_primes if p <= current_size]
            if len(needed_primes) > len(primes_up_to_K):
                # Rebuild with all needed primes
                marked2 = bytearray(N + 1)
                for p in needed_primes:
                    for multiple in range(0, N + 1, p):
                        marked2[multiple] = 1
                S = set(i for i in range(N + 1) if not marked2[i])
                current_size = len(S)
            
            # Now check if it's admissible
            # Verify: for every prime p <= current_size, S misses a residue mod p
            valid = True
            for p in all_primes:
                if p > current_size:
                    break
                residues = set(x % p for x in S)
                if len(residues) == p:
                    valid = False
                    break
            
            if valid:
                if current_size > best_size:
                    best_size = current_size
                    best_set = sorted(S)
            else:
                # Repair: remove elements to fix violations
                S_list = list(S)
                changed = True
                while changed and time.time() - start_time < DEADLINE:
                    changed = False
                    current_size = len(S_list)
                    for p in all_primes:
                        if p > current_size:
                            break
                        counts = {}
                        for x in S_list:
                            r = x % p
                            counts[r] = counts.get(r, 0) + 1
                        if len(counts) == p:
                            min_residue = min(counts, key=counts.get)
                            S_list = [x for x in S_list if x % p != min_residue]
                            changed = True
                            break
                    if time.time() - start_time > DEADLINE - 5:
                        break
                
                if S_list:
                    # Final validation
                    final_size = len(S_list)
                    valid = True
                    for p in all_primes:
                        if p > final_size:
                            break
                        residues = set(x % p for x in S_list)
                        if len(residues) == p:
                            valid = False
                            break
                    if valid and final_size > best_size:
                        best_size = final_size
                        best_set = sorted(S_list)
    
    # If no valid set found, fall back to a simple construction
    if not best_set:
        # Use the seed approach: {1} + primes > sqrt(N), then repair
        sqrt_N = int(math.isqrt(N))
        S_list = [1] + [p for p in all_primes if p * p > N]
        changed = True
        while changed and time.time() - start_time < DEADLINE:
            changed = False
            current_size = len(S_list)
            for p in all_primes:
                if p > current_size:
                    break
                counts = {}
                for x in S_list:
                    r = x % p
                    counts[r] = counts.get(r, 0) + 1
                if len(counts) == p:
                    min_residue = min(counts, key=counts.get)
                    S_list = [x for x in S_list if x % p != min_residue]
                    changed = True
                    break
        
        if S_list:
            best_set = sorted(S_list)
    
    return best_set
