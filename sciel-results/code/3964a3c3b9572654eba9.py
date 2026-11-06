import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 123457
    
    # 1. Generate primes up to N
    # Simple sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Binary search on the index of the prime bound
    # We search for the smallest prime index k such that 
    # there exists a pair (a, b) with a+b=N, P(a) <= primes[k], P(b) <= primes[k]
    
    # Helper: Check if a solution exists with prime bound B
    def exists_smooth_pair(B):
        # Get primes <= B
        allowed_primes = [p for p in primes if p <= B]
        
        # Generate all B-smooth numbers <= N
        smooth_numbers = set()
        
        # Recursive generation
        def gen_smooth(idx, current_val):
            if current_val > N:
                return
            smooth_numbers.add(current_val)
            
            if idx >= len(allowed_primes):
                return
            
            p = allowed_primes[idx]
            next_val = current_val
            while next_val <= N:
                next_val *= p
                if next_val > N:
                    break
                gen_smooth(idx, next_val) # Note: we allow multiple powers of p, so we recurse with same idx
                # Wait, standard smooth number generation:
                # We multiply by p repeatedly, then move to next prime.
                # My recursion above is slightly wrong. 
                # Let's use a standard iterative/recursive approach:
                # For each prime p_i, we can multiply by p_i any number of times.
                # Then move to p_{i+1}.
                
        # Correct generation logic:
        smooth_numbers = {1}
        current_smooth = [1]
        for p in allowed_primes:
            new_smooth = []
            for x in current_smooth:
                val = x
                while val * p <= N:
                    val *= p
                    new_smooth.append(val)
            current_smooth.extend(new_smooth)
        
        # The set of smooth numbers is current_smooth
        # Check for pair
        for a in current_smooth:
            if a > N: continue
            b = N - a
            if b >= 1 and b in smooth_numbers:
                return True
        return False
        
    # Note: The generation logic above builds a list. 
    # 1 is smooth.
    # If allowed_primes is empty, smooth_numbers = {1}.
    # If B < 2, only 1 is smooth.
    
    # We want to find the minimal B.
    # B must be at least 1.
    # Let's binary search over the primes list indices.
    # If no prime is used, B=1.
    
    lo = 0
    hi = len(primes) - 1
    best_B = primes[-1] # Fallback
    
    # Check B=1 first? 
    # If B=1, only 1 is smooth. a=1, b=123456. P(123456) must be 1. Impossible.
    # So B must be at least 2.
    
    # We search for the smallest prime P_k such that exists_smooth_pair(P_k) is True.
    # But the bound B doesn't have to be a prime? 
    # P(a) is the largest prime factor. So if P(a) <= B, it means all prime factors of a are <= B.
    # The set of smooth numbers depends only on the set of primes <= B.
    # So the feasibility only changes when B crosses a prime value.
    # Thus, binary searching on prime indices is correct.
    
    found = False
    ans_prime = None
    
    # We can also just iterate primes from smallest to largest? 
    # The number of primes up to 123457 is ~11000.
    # Iterating 11000 times, each generating smooth numbers... might be slow if smooth numbers grow large.
    # Binary search is safer.
    
    # However, the feasibility function `exists_smooth_pair` is monotonic?
    # If a pair exists for B, it exists for B' > B.
    # Yes, because the set of smooth numbers grows.
    
    # Let's do binary search on the index of primes.
    # If index 0 (prime=2) works, return 2.
    # If index k works, check k-1.
    
    # Wait, is it possible that the optimal B is not a prime?
    # The value we minimize is max(P(a), P(b)).
    # P(a) is always a prime (or 1).
    # So the optimal value is either 1 or a prime.
    # So searching over primes is sufficient.
    
    # But we need to return the pair (a,b), not just the value.
    # The task asks to return a,b.
    # "Return two positive integers a,b ... Minimize max(P(a),P(b))"
    # The evaluator measures the raw value (lower is better).
    # So I should return the pair that achieves the minimum.
    
    # Let's modify the helper to return the pair if found.
    
    def find_smooth_pair(B):
        allowed_primes = [p for p in primes if p <= B]
        smooth_numbers = {1}
        current_smooth = [1]
        for p in allowed_primes:
            new_smooth = []
            for x in current_smooth:
                val = x
                while val * p <= N:
                    val *= p
                    new_smooth.append(val)
            current_smooth.extend(new_smooth)
        
        for a in current_smooth:
            if a > N: continue
            b = N - a
            if b >= 1 and b in smooth_numbers:
                return (a, b)
        return None

    # Binary search
    # lo = 0 (prime 2), hi = len(primes)-1
    # If find_smooth_pair(primes[0]) is not None, answer is primes[0].
    # Else, binary search.
    
    # Actually, let's just iterate? 
    # Number of primes is ~11000.
    # Generation of smooth numbers for small B is fast.
    # For large B, it's slower.
    # Binary search is O(log P) * Cost(Feasibility).
    # Cost(Feasibility) depends on B.
    # Worst case B is large.
    # Let's stick to binary search.
    
    lo = 0
    hi = len(primes) - 1
    best_pair = None
    best_prime_idx = -1
    
    # Check if any solution exists? 
    # a=1, b=123456. P(123456) is some prime.
    # So a solution always exists with B = max(P(1), P(123456)) = P(123456).
    
    while lo <= hi:
        mid = (lo + hi) // 2
        B = primes[mid]
        pair = find_smooth_pair(B)
        if pair is not None:
            best_pair = pair
            best_prime_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_pair:
        return list(best_pair)
    
    # Fallback: just return [1, N-1]
    return [1, N - 1]
