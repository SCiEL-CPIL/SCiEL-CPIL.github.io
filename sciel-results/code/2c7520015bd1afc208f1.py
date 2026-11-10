import numpy as np
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = 280.0 # 20s margin
    
    S = np.arange(N + 1)
    all_primes = _primes(N)
    
    while True:
        if time.time() > deadline:
            break
            
        K = len(S)
        if K == 0:
            break
        
        # Find primes <= K
        # Since K decreases, we can filter all_primes
        relevant_primes = [p for p in all_primes if p <= K]
        
        if not relevant_primes:
            break
            
        best_p = None
        best_r = None
        min_cost = float('inf')
        
        for p in relevant_primes:
            # Check if all residues are covered
            residues = S % p
            # np.unique is O(K log K) or O(K) with bincount?
            # bincount is O(K + p)
            counts = np.bincount(residues, minlength=p)
            
            if np.all(counts > 0):
                # Covered
                cost = np.min(counts)
                if cost < min_cost:
                    min_cost = cost
                    best_p = p
                    best_r = np.argmin(counts)
        
        if best_p is None:
            # Admissible
            break
        
        # Remove elements
        mask = (S % best_p) != best_r
        S = S[mask]
        
    return S.tolist()
