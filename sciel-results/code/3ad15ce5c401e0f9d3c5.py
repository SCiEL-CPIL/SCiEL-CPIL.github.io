import math
import sys

# Increase recursion limit just in case, though depth is small
sys.setrecursionlimit(10000)

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order to prioritize large values (fewer items)
    divisors.sort(reverse=True)
    N = len(divisors)
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] is the sum of divisors[i:]
    suffix_sum = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
        
    best_count = float('inf')
    best_solution = []
    
    def search(idx, rem, count, current):
        nonlocal best_count, best_solution
        
        # Pruning 1: If we already have a solution with fewer or equal items, stop
        if count >= best_count:
            return
            
        # Pruning 2: If we reached the target
        if rem == 0:
            if count < best_count:
                best_count = count
                best_solution = current[:]
            return
            
        # Pruning 3: If no more divisors to consider
        if idx == N:
            return
            
        # Pruning 4: If even taking all remaining divisors is not enough to reach target
        if suffix_sum[idx] < rem:
            return
            
        # Pruning 5: If remaining sum is negative (should be handled by check before call or here)
        if rem < 0:
            return
            
        # Option 1: Include divisors[idx]
        # Only if it doesn't exceed the remaining target
        if divisors[idx] <= rem:
            current.append(divisors[idx])
            search(idx + 1, rem - divisors[idx], count + 1, current)
            current.pop()
            
        # Option 2: Exclude divisors[idx]
        search(idx + 1, rem, count, current)
        
    search(0, target, 0, [])
    
    return best_solution
