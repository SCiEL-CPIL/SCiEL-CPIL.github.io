"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from math import isqrt

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Approach:
    Instead of precomputing LPF for all numbers, we iterate through possible
    maximum prime factor limits 'y'. For each 'y', we generate all 'y'-smooth
    numbers in the range [10000, 1000000]. We then check if any two consecutive
    integers are both in this set. The first 'y' for which such a pair exists
    gives the minimal possible largest prime factor.
    """
    MAX_N = 1000000
    MIN_N = 10000
    
    # Precompute primes up to a reasonable limit.
    # The optimal 'y' is likely small. Let's go up to 1000 to be safe.
    # If no solution is found with y <= 1000, we fall back to a sieve method
    # for the remaining range, though it's highly unlikely.
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(limit) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    
    def generate_smooth_numbers(max_val, primes_list):
        """
        Generate all numbers <= max_val that have no prime factors > max(primes_list).
        Returns a sorted list of such numbers in [MIN_N, MAX_N].
        """
        if not primes_list:
            return []
        
        # Use a set to avoid duplicates if the generation method allows, 
        # though standard DFS with non-decreasing primes avoids duplicates.
        smooths = set()
        
        def dfs(current, prime_idx):
            if current > max_val:
                return
            if current >= MIN_N:
                smooths.add(current)
            
            for i in range(prime_idx, len(primes_list)):
                p = primes_list[i]
                next_val = current * p
                if next_val > max_val:
                    break
                dfs(next_val, i) # i, not i+1, to allow powers
        
        dfs(1, 0)
        
        # Filter to range [MIN_N, MAX_N]
        result = [x for x in smooths if MIN_N <= x <= MAX_N]
        result.sort()
        return result

    # Iterate y from 2 upwards
    # We check y=2, 3, ... 
    # For each y, we take primes <= y
    # Generate smooth numbers
    # Check for consecutive pairs
    
    # Optimization: We can just increment y and add the new prime to the list
    # but regenerating is simple enough for small y.
    # Let's try up to y=1000.
    
    current_primes = []
    
    for y in range(2, 1001):
        # Update current_primes if y is prime
        # Actually, simpler: just filter the precomputed primes list
        # But that's O(P) each time. 
        # Let's just iterate through the primes list and pick those <= y.
        # Since y is small, this is fast.
        
        # To be more efficient, we can maintain a pointer.
        # But let's keep it simple first.
        
        active_primes = [p for p in primes if p <= y]
        
        if not active_primes:
            continue
            
        smooth_nums = generate_smooth_numbers(MAX_N, active_primes)
        
        # Check for consecutive integers in smooth_nums
        # Since smooth_nums is sorted, we can iterate
        found_n = None
        for i in range(len(smooth_nums) - 1):
            if smooth_nums[i+1] - smooth_nums[i] == 1:
                found_n = smooth_nums[i]
                break
        
        if found_n is not None:
            return found_n
            
    # Fallback if no solution found with y <= 1000 (very unlikely)
    # Use the sieve method from the original code as a safety net
    N = 1000001
    lpf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_n = 10000
    best_lpf_val = max(lpf[10000], lpf[10001])
    for n in range(10000, N):
        current_max = max(lpf[n], lpf[n + 1])
        if current_max < best_lpf_val:
            best_lpf_val = current_max
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
