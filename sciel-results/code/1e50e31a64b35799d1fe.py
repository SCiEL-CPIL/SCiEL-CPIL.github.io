import time
import numpy as np

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 280
    
    points = np.arange(N + 1, dtype=np.int32)
    primes = _primes(N + 1)
    
    # We will process primes in order.
    # For small primes, we use lookahead to choose the best residue class to remove.
    # For larger primes, we just pick the smallest class.
    
    idx = 0
    while idx < len(primes):
        if time.time() > deadline:
            break
            
        p = primes[idx]
        if p > len(points):
            break
            
        residues = np.mod(points, p)
        unique_res, counts = np.unique(residues, return_counts=True)
        
        if len(unique_res) < p:
            # Already admissible for p
            idx += 1
            continue
            
        # Need to remove one class
        # Sort by count (ascending)
        sorted_idx = np.argsort(counts)
        
        best_points = None
        best_score = float('-inf') # We want to maximize remaining size, or minimize cost
        
        # Lookahead depth: 1 step (next prime)
        # Try the top K smallest classes
        K = min(5, len(sorted_idx))
        
        for i in range(K):
            r = unique_res[sorted_idx[i]]
            mask = residues != r
            new_points = points[mask]
            
            # Evaluate next prime
            # Find next prime q > p such that q <= len(new_points)
            # Since we are iterating primes, the next relevant prime is primes[idx+1]
            # But only if primes[idx+1] <= len(new_points)
            
            if idx + 1 < len(primes):
                q = primes[idx+1]
                if q <= len(new_points):
                    # Check if new_points covers all residues mod q
                    q_residues = np.mod(new_points, q)
                    q_unique, q_counts = np.unique(q_residues, return_counts=True)
                    if len(q_unique) == q:
                        # We will have to remove something for q too
                        # Estimate cost: size of smallest q-class
                        min_q_size = np.min(q_counts)
                        # Total removed so far + estimated future removal
                        # We want to minimize total removal.
                        # Current removal: counts[sorted_idx[i]]
                        # Future removal: min_q_size
                        # Score = - (len(vals) + min_q_size)
                        score = -(counts[sorted_idx[i]] + min_q_size)
                    else:
                        # No removal needed for q
                        score = -counts[sorted_idx[i]]
                else:
                    # q is not active
                    score = -counts[sorted_idx[i]]
            else:
                score = -counts[sorted_idx[i]]
                
            if score > best_score:
                best_score = score
                best_points = new_points
        
        if best_points is not None:
            points = best_points
            
        idx += 1
        
    return points.tolist()
