import bisect

def construct():
    from math import isqrt

    # Generate all primes <= 10000
    primes = []
    for p in range(2, 10001):
        if all(p % d for d in range(2, isqrt(p) + 1)):
            primes.append(p)
    
    n = len(primes)
    if n < 2:
        return primes

    best_seq = []
    
    # Strategy 1: Greedy Smallest Gap
    for i in range(n):
        p = primes[i]
        last_gap = 0
        seq = [p]
        next_idx = i + 1
        
        while next_idx < n:
            target = p + last_gap
            # Find the first prime >= target starting from next_idx
            j = bisect.bisect_left(primes, target, lo=next_idx)
            
            if j < n:
                gap = primes[j] - p
                seq.append(primes[j])
                last_gap = gap
                p = primes[j]
                next_idx = j + 1
            else:
                break
        
        if len(seq) > len(best_seq):
            best_seq = seq
            
    # Strategy 2: Greedy Largest Gap (might find different paths, though usually worse for length)
    # But let's try a "Lookahead" Greedy: Pick the smallest gap that allows at least 1 more step?
    # That's complex. Let's just return the best from Strategy 1.
    
    return best_seq
