import time
import math
import heapq

def construct():
    deadline = time.time() + 75  # Safety margin before 90s limit
    target = 123457
    
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Compute largest prime factor for all numbers up to target-1 first (fast sieve)
    max_n = target - 1
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                lpf[j] = i
    
    # Try increasing k: generate k-smooth numbers and check if target is a sum of two
    # Use lpf to verify smoothness instead of separate generation
    for k in range(1, len(primes_list) + 1):
        if time.time() > deadline - 15:
            break
        
        primes = set(primes_list[:k])
        max_prime = primes_list[k - 1]
        
        # Find all numbers x in [1, target-1] with P(x) <= max_prime
        # i.e., all prime factors of x are in primes
        smooth_set = set()
        for x in range(1, target):
            if lpf[x] <= max_prime:
                # Verify all prime factors are in primes (lpf gives largest, but need all)
                # Actually if lpf[x] <= max_prime and all primes up to max_prime are in our set,
                # then x is k-smooth only if all its prime factors are in primes_list[:k]
                # Since primes_list is sorted and contains all primes up to max_prime,
                # and lpf[x] <= max_prime means all prime factors of x are <= max_prime,
                # they must all be in primes_list[:k] (which contains all primes <= max_prime)
                smooth_set.add(x)
        
        # Check if target can be written as sum of two k-smooth numbers
        for x in smooth_set:
            y = target - x
            if y > 0 and y in smooth_set:
                return [x, y]
        
        # If smooth set is very large, further k will be even larger - break
        if len(smooth_set) > 80000:
            break
    
    # Fallback: full scan to find best pair
    best_max = float('inf')
    best_a = 1
    
    for a in range(1, target):
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
    
    return [best_a, target - best_a]
