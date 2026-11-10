import math

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to N
PRIMES = _primes(N)

def construct():
    valid_indices = list(range(N + 1))
    
    best_k = 0
    best_set = []
    
    for p in PRIMES:
        if not valid_indices:
            break
            
        # Group by residue
        counts = [0] * p
        for v in valid_indices:
            counts[v % p] += 1
            
        # Find min count
        min_count = min(counts)
        min_res = counts.index(min_count)
        
        # Remove all v with v % p == min_res
        valid_indices = [v for v in valid_indices if v % p != min_res]
        
        L = len(valid_indices)
        
        # The set valid_indices is formed by removing residues for primes <= p.
        # It is admissible for any size K such that:
        # 1. K <= L (we have enough elements)
        # 2. All primes q <= K are <= p (so they were processed)
        #    This implies K <= p (since p is the largest prime processed so far, 
        #    and there are no primes between p and next prime, 
        #    actually we just need p >= K to ensure all primes <= K are covered).
        #    Wait, if K <= p, then all primes <= K are <= p, so they are processed.
        #    If K > p, then there is a prime q with p < q <= K which was NOT processed.
        #    So we must have K <= p.
        
        current_max_k = min(p, L)
        
        if current_max_k > best_k:
            best_k = current_max_k
            # Store a copy of the current set
            best_set = list(valid_indices)
            
    return best_set[:best_k]
