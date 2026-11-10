import sys
import time

def construct():
    start_time = time.time()
    limit = 10000
    
    # Sieve of Eratosthenes
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            for multiple in range(p*p, limit + 1, p):
                is_prime[multiple] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    N = len(primes)
    
    # Map prime value to index
    idx = [-1] * (limit + 1)
    for i, p in enumerate(primes):
        idx[p] = i
        
    # Precompute previous indices for each prime i and gap d
    # prevs[i] is a dict {d: j}
    prevs = [{} for _ in range(N)]
    for i, p in enumerate(primes):
        # d can be at most p - 2
        # We only care about d such that p-d is prime
        # We can iterate j < i
        # But doing this for all i is O(N^2). N=1229, N^2=1.5M. This is fine.
        # Actually, we can just do it inside the main loop or precompute.
        # Precomputing prevs[i] by iterating d is O(N * Limit) which is 12M.
        # Iterating j < i is O(N^2) which is 1.5M.
        # Let's iterate j < i to build prevs[i]
        for j in range(i):
            d = p - primes[j]
            prevs[i][d] = j

    # DP Table
    # g[i][d] = max length of sequence ending at primes[i] with last gap <= d
    # Using list of lists of ints
    # Max length is small, but we use int for safety.
    # Memory: 1229 * 10001 * 8 bytes ~ 100MB. Might be high for some environments but usually OK.
    # Let's try to optimize memory if needed, but let's start with this.
    
    # Actually, we can use array('H') or similar?
    # Let's stick to lists for speed of access in pure Python loops.
    
    g = [[0] * (limit + 1) for _ in range(N)]
    
    # Base case: single element sequences
    for i in range(N):
        g[i][0] = 1
        # g[i][d] for d > 0 will be at least 1

    max_len = 1
    best_i = 0
    best_d = 0
    
    for i in range(N):
        pi = primes[i]
        gi = g[i]
        prevs_i = prevs[i]
        
        # gi[0] is already 1
        # Iterate d from 1 to limit
        # We can break early if pi - d < 2? No, we need to carry forward the max.
        # gi[d] = max(gi[d-1], cand)
        
        # Optimization: We only need to check d where pi-d is prime?
        # No, because gi[d] depends on gi[d-1].
        # But cand is only non-zero if d is in prevs_i.
        
        current_max = 1 # gi[0]
        
        # We can iterate d from 1 to pi-2 (max possible gap for this prime)
        # But we need to fill the rest with current_max?
        # Yes, if d > pi-2, no new candidate, so gi[d] = gi[d-1].
        # We can just fill up to pi-2 and then fill the rest?
        # Or just iterate to limit. 10000 iterations * 1229 = 12M. Fine.
        
        for d in range(1, limit + 1):
            # Check if there is a previous prime at pi - d
            j = prevs_i.get(d)
            if j is not None:
                # cand = g[j][d] + 1
                # g[j][d] is the max length ending at j with last gap <= d
                cand = g[j][d] + 1
                if cand > current_max:
                    current_max = cand
            # gi[d] is the max length ending at i with last gap <= d
            # This is max(gi[d-1], best candidate for gap exactly d)
            # current_max holds gi[d-1] effectively?
            # No, current_max is the running max.
            # Let's trace:
            # d=1: current_max = 1. j = prevs[1]. if valid, cand = g[j][1]+1.
            # if cand > 1, current_max = cand.
            # gi[1] = current_max.
            # d=2: current_max is gi[1]. j = prevs[2]. if valid, cand = g[j][2]+1.
            # if cand > current_max, current_max = cand.
            # gi[2] = current_max.
            
            gi[d] = current_max
            
            if current_max > max_len:
                max_len = current_max
                best_i = i
                best_d = d

    # Reconstruct sequence
    # Start from (best_i, best_d)
    # We need to trace back.
    # State (i, d) represents the best sequence ending at i with last gap <= d.
    # If gi[d] == gi[d-1], the optimal gap is < d.
    # If gi[d] > gi[d-1], the optimal gap is exactly d (or we extended from j with gap d).
    
    seq = []
    i = best_i
    d = best_d
    
    # Backtrack
    # We need to find the specific path.
    # If gi[d] > gi[d-1], it means the improvement came from a candidate with gap exactly d.
    # That candidate came from j = prevs_i[d].
    # And the length was g[j][d] + 1.
    # So the previous state is (j, d).
    # Note: The constraint for j is "last gap <= d".
    
    while i != -1:
        seq.append(primes[i])
        
        # Check if we can go to d-1
        # If gi[d] == gi[d-1], then the best sequence ending at i with gap <= d
        # is the same as the best sequence ending at i with gap <= d-1.
        # So we can just reduce d.
        if d > 0 and g[i][d] == g[i][d-1]:
            d -= 1
            continue
            
        # If we are here, g[i][d] > g[i][d-1] (or d==0)
        # If d==0, it's a single element.
        if d == 0:
            break
            
        # The improvement must have come from a transition with gap exactly d.
        # Find j
        j = prevs[i].get(d)
        if j is None:
            # This shouldn't happen if g[i][d] > g[i][d-1]
            # Unless it's a base case? No, base case is d=0.
            # Maybe numerical error?
            break
        i = j
        # The next state is (j, d). The constraint on j is last gap <= d.
        # So we keep d the same.
        # Wait, is it possible that g[j][d] came from a gap < d?
        # Yes. But that will be handled in the next iteration of the loop
        # when we check if g[j][d] == g[j][d-1].
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
