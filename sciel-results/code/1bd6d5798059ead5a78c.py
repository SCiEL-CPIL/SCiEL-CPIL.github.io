import time
import heapq

def construct():
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85
    
    # 1. Sieve for Primes up to TARGET
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(TARGET**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Iterate primes K in increasing order
    for K in primes:
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate K-smooth numbers <= TARGET
        # K-smooth means all prime factors <= K
        # We need primes <= K
        # Since primes is sorted, we can find the index of K
        # Or just iterate primes and break if p > K
        
        smooths = set()
        smooths.add(1)
        heap = [1]
        
        # Get relevant primes <= K
        # We can just iterate through the global primes list and stop when p > K
        # But doing this inside the loop is O(P) for each K.
        # Better: precompute indices? Or just slice.
        # Since primes is sorted, we can use bisect.
        import bisect
        idx = bisect.bisect_right(primes, K)
        relevant_primes = primes[:idx]
        
        # Generate
        while heap:
            x = heapq.heappop(heap)
            for p in relevant_primes:
                y = x * p
                if y > TARGET:
                    break
                if y not in smooths:
                    smooths.add(y)
                    heapq.heappush(heap, y)
        
        # Check for pair
        found_pair = None
        for a in smooths:
            b = TARGET - a
            if b in smooths:
                if a > 0 and b > 0:
                    found_pair = [a, b]
                    break
        
        if found_pair:
            return found_pair
    
    # Fallback: Full scan with LPF
    # (Implement the current program's logic here)
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:
            for j in range(i, TARGET + 1, i):
                lpf[j] = i
                
    best_max_p = TARGET
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET // 2 + 1):
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
            
    return [best_a, best_b]
