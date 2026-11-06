import time
import heapq

def construct():
    deadline = time.time() + 85
    TARGET = 123457
    
    # Sieve of Eratosthenes to get primes
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    limit = int(TARGET**0.5)
    for i in range(2, limit + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, TARGET + 1) if is_prime[i]]
    
    # Helper to generate y-smooth numbers <= TARGET
    def get_smooth_numbers(y):
        small_primes = [p for p in primes if p <= y]
        if not small_primes:
            return []
            
        smooths = set()
        # Use a heap to generate in order, or just recursive
        # Given N is small, recursive is fine
        def gen(idx, current_val):
            if current_val > TARGET:
                return
            smooths.add(current_val)
            if idx >= len(small_primes):
                return
            p = small_primes[idx]
            # Multiply by p as many times as possible
            val = current_val
            while val <= TARGET:
                gen(idx + 1, val)
                if val > TARGET // p:
                    break
                val *= p
                
        gen(0, 1)
        return smooths

    # Binary search on the smoothness bound y
    # We want the smallest y such that there exist a, b y-smooth with a+b=TARGET
    low = 2
    high = TARGET
    best_pair = None
    best_y = TARGET
    
    # We can also just iterate y from 2 upwards until we find a pair,
    # since the optimal y is likely small (11).
    # This is faster than binary search if the answer is small.
    # Let's iterate y from 2 to 100 (or until found).
    # If not found, fall back to binary search or linear scan.
    
    found = False
    for y in range(2, 100):
        if time.time() > deadline - 5:
            break
        smooths = get_smooth_numbers(y)
        # Check for pair
        # Optimization: sort smooths? Or just use set lookup.
        # For each a in smooths, check if TARGET - a in smooths.
        for a in smooths:
            b = TARGET - a
            if b in smooths:
                # Found a pair with max prime factor <= y
                # We want to minimize y, so the first y we find is the min?
                # Yes, because we iterate y in increasing order.
                # But we need to ensure P(a) <= y and P(b) <= y.
                # By construction, all numbers in smooths have prime factors <= y.
                # So max(P(a), P(b)) <= y.
                # Is it possible that the actual max is < y?
                # Yes, but we are looking for the minimum y that *allows* a solution.
                # The value of the objective is max(P(a), P(b)).
                # If we find a solution at y, the value is at most y.
                # Could there be a solution with value < y?
                # If there were, we would have found it at a smaller y.
                # So the first y we find gives the optimal value?
                # Not necessarily. The value is max(P(a), P(b)).
                # If we find a pair at y=11, the value is <= 11.
                # It could be 11 or less.
                # But if it were less (e.g. 7), we would have found it at y=7.
                # So the first y that yields a solution corresponds to the optimal value?
                # Yes, because if a solution exists with value V, it is a solution for any y >= V.
                # So the smallest y for which a solution exists is exactly the minimum value V.
                
                best_pair = [a, b]
                best_y = y
                found = True
                break
        if found:
            break
            
    if not found:
        # Fallback: linear scan as before
        # Compute LPF for all
        lpf = [0] * (TARGET + 1)
        lpf[1] = 1
        for p in primes:
            for j in range(p, TARGET + 1, p):
                lpf[j] = p
                
        best_val = float('inf')
        best_a = 1
        best_b = TARGET - 1
        for a in range(1, TARGET):
            if time.time() > deadline - 5:
                break
            b = TARGET - a
            pa = lpf[a]
            pb = lpf[b]
            m = pa if pa > pb else pb
            if m < best_val:
                best_val = m
                best_a = a
                best_b = b
        return [best_a, best_b]
        
    return best_pair
