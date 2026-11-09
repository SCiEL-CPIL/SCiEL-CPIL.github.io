import time
import math

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Primes to check in order
    primes_to_check = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We always include 2 in the set of allowed primes because a+b is odd, 
    # so one is even (factor 2) and one is odd. 
    # The even one can have factor 2. The odd one cannot.
    # But for the set generation, we include 2 to cover the even number.
    
    current_primes = [2]
    
    for p in primes_to_check:
        if time.time() > deadline:
            break
            
        current_primes.append(p)
        
        # Generate all numbers <= N with prime factors in current_primes
        # Using a set to store them
        smooth_nums = {1}
        # We can generate by iterating through the set and multiplying by primes
        # But a recursive approach or a priority queue is better to avoid duplicates and keep it small.
        # Given N is small, a simple BFS/DFS works.
        
        # Let's use a list and a set to track visited
        # Actually, for small N, we can just use a set and expand.
        # Start with 1.
        # For each prime, multiply existing numbers by powers of that prime.
        
        # Efficient generation:
        # Sort primes
        # Use a min-heap or just a list and sort?
        # For N=123457, the number of smooth numbers is small.
        # Let's use a simple recursive generator.
        
        def gen_smooths(primes, limit):
            result = set()
            def _gen(idx, current):
                if current > limit:
                    return
                if idx == len(primes):
                    result.add(current)
                    return
                p = primes[idx]
                # Option 1: Don't use p
                _gen(idx + 1, current)
                # Option 2: Use p
                c = current
                while c <= limit:
                    c *= p
                    if c > limit:
                        break
                    _gen(idx + 1, c)
            
            _gen(0, 1)
            return result

        smooth_nums = gen_smooths(current_primes, N)
        
        # Check for pair
        for a in smooth_nums:
            b = N - a
            if b >= 1 and b in smooth_nums:
                return [a, b]
                
    # Fallback: If no small-prime solution found quickly, use the SPF method
    # This ensures we return a valid answer within time limits
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        x = i
        max_p = 1
        while x > 1:
            p = spf[x]
            if p > max_p:
                max_p = p
            while x % p == 0:
                x //= p
        lpf[i] = max_p
        
    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        val = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            if best_val == 3:
                break
    return [best_a, best_b]
