import numpy as np
from heapq import heappush, heappop
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Approach:
    Iterate through possible maximum prime factors P in increasing order.
    For each P, determine if there exists an n in [10000, 1000000] such that
    both n and n+1 are P-smooth (i.e., all prime factors are <= P).
    The first such P found gives the minimal largest prime factor.
    """
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # 1000001
    
    # Generate primes up to MAX_VAL
    # Simple sieve
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_VAL + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # We will iterate through primes as the bound P
    # For a given P, we need to generate all P-smooth numbers <= MAX_VAL
    # A number is P-smooth if all its prime factors are in the set of primes <= P.
    
    # To efficiently check for consecutive P-smooth numbers, we can maintain a boolean array
    # indicating which numbers are P-smooth.
    # However, generating the set from scratch for each P might be slow if P is large.
    # But we expect P to be small.
    
    # Let's estimate how many P-smooth numbers there are.
    # For P=2, {1, 2, 4, ...}
    # For P=3, {1, 2, 3, 4, 6, 8, 9, ...}
    # The count is roughly (log(MAX_VAL))^k / k! where k is number of primes <= P.
    # For P=100, k=25. This might be large.
    # But we stop as soon as we find a pair.
    
    # Optimization:
    # Instead of regenerating the whole set, we can just check candidates?
    # No, we don't know which candidates are smooth.
    
    # Let's try generating the set of P-smooth numbers for increasing P.
    # We can use a dynamic approach:
    # Start with P=2. Set S = {1, 2, 4, ...}.
    # Check for consecutive pairs in S within range.
    # If none, add next prime 3 to the set of allowed primes.
    # Update S to include numbers with factor 3.
    # This is complex to update incrementally.
    
    # Simpler: For each P, generate the list of P-smooth numbers <= MAX_VAL.
    # Use a min-heap to generate them in order.
    # Or use a boolean array `is_smooth` and update it?
    # If we use a boolean array, we can update it by multiplying existing smooth numbers by the new prime.
    # Let's maintain `is_smooth` array.
    # Initially, for P=2, we mark powers of 2.
    # When we add prime p, we need to mark all numbers of form x * p^k where x was smooth before?
    # No, x must be smooth with respect to primes < p?
    # Actually, a number is P-smooth if it is composed of primes in {p1, ..., pk}.
    # If we add pk, the new smooth numbers are old_smooth * pk^j.
    # But old_smooth might already contain numbers with pk? No, we process primes in order.
    # So, `is_smooth` for P_k contains numbers formed by p1...p_k.
    # To get `is_smooth` for P_{k+1}, we take the current `is_smooth` set, and for each number x in it,
    # we can multiply by p_{k+1} repeatedly.
    # But we must be careful not to duplicate work or miss numbers.
    # Actually, if we just iterate through all numbers 1..MAX_VAL and check if they are smooth, it's O(N * number_of_primes).
    
    # Let's stick to generating the list of smooth numbers for a specific P.
    # We do this for P = 2, 3, 5, 7, ... until we find a solution.
    
    def get_smooth_numbers(prime_bound, limit):
        """
        Generate all numbers <= limit whose prime factors are all <= prime_bound.
        Uses a priority queue to generate them in sorted order.
        """
        # Find primes <= prime_bound
        allowed_primes = [p for p in primes if p <= prime_bound]
        if not allowed_primes:
            return []
        
        # We generate numbers of the form p1^a1 * p2^a2 * ...
        # We can use a heap.
        # Start with 1.
        # To avoid duplicates, we can enforce an order on exponents or use a set.
        # Given the limit is 10^6, the number of smooth numbers is manageable for small P.
        
        smooth = [1]
        # We can do a BFS/DFS multiplication.
        # Let's use a set to store generated numbers to avoid duplicates, then sort.
        # Or use the standard "ugly numbers" algorithm generalized.
        
        # Simpler method: 
        # Initialize array of 1s? No.
        # Let's just use a set and expand.
        current_smooths = {1}
        all_smooths = set()
        
        # This might be inefficient if we just expand.
        # Better: For each prime in allowed_primes, we can multiply existing numbers.
        # But we need to be careful.
        
        # Let's use the heap method.
        # h = [(1, 0)] where 0 is index of last prime used?
        # Standard Ugly Number algorithm for k primes:
        # u[0] = 1
        # i1, i2, ..., ik = 0
        # next = min(p1*u[i1], p2*u[i2], ...)
        
        # We don't know the size of the list in advance.
        # We can generate until we exceed limit.
        
        u = [1]
        indices = [0] * len(allowed_primes)
        
        while True:
            min_val = float('inf')
            min_idx = -1
            for i, p in enumerate(allowed_primes):
                val = u[indices[i]] * p
                if val < min_val:
                    min_val = val
                    min_idx = i
            
            if min_val > limit:
                break
            
            # Avoid duplicates
            if min_val != u[-1]:
                u.append(min_val)
            
            # Advance indices for all primes that produced min_val
            for i, p in enumerate(allowed_primes):
                if u[indices[i]] * p == min_val:
                    indices[i] += 1
        
        return u

    # Iterate through primes as the bound P
    # We need to check if there is n in [10000, 1000000] such that n and n+1 are in the smooth set.
    
    for P in primes:
        smooth_nums = get_smooth_numbers(P, MAX_VAL)
        
        # Convert to a set for O(1) lookup? Or sorted list for binary search?
        # The list is sorted.
        # We can iterate through smooth_nums and check if n+1 is also in the set.
        # To make lookup fast, convert to set.
        smooth_set = set(smooth_nums)
        
        found = False
        for n in smooth_nums:
            if n < 10000:
                continue
            if n > MAX_N:
                break
            
            if (n + 1) in smooth_set:
                return n
        
        # If no solution found for this P, continue to next prime
        # Note: If P is very large, smooth_nums will be large (up to 10^6).
        # But we expect to find a solution for small P.
        
    # Fallback: return the result from the original method if no small P works
    # This should not happen for this range, but just in case.
    # We can fall back to the LPF sieve if necessary, but let's assume it works.
    # To be safe, if we exhaust primes up to MAX_VAL, we should return something valid.
    # The loop above goes through all primes. If we finish, it means no pair exists?
    # Actually, if P = MAX_VAL, all numbers are P-smooth. So n=10000, n+1=10001 is a pair.
    # So it will always return.
    
    return 10000 # Should not reach here
# EVOLVE-BLOCK-END
