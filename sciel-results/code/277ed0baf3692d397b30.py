"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect
import numpy as np

def construct():
    start = time.time()
    deadline = start + 55
    
    n_max = 10000
    # Sieve of Eratosthenes using numpy for speed
    is_prime = np.ones(n_max + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [int(p) for p in np.where(is_prime)[0]]
    n = len(primes)
    
    if n <= 2:
        return primes
    
    # DP with Pareto frontier per prime index.
    # For each prime index i, we maintain:
    #   fgaps[i]  = sorted list of gap values (ascending)
    #   fpmax[i]  = prefix-max of lengths at each gap (non-decreasing)
    #   fprev[i]  = for each frontier entry, the previous prime index that achieved that length
    #
    # Query: max length of subsequence ending at i with last_gap <= T
    #   = fpmax[i][bisect_right(fgaps[i], T) - 1]  (or 1 if no entry)
    #
    # Build frontier[i] from all j < i:
    #   gap = primes[i] - primes[j]
    #   len = query(frontier[j], gap) + 1
    #   Then take Pareto frontier (sort by gap, prefix-max length, keep only improvements)
    
    fgaps = [None] * n
    fpmax = [None] * n
    fprev = [None] * n
    max_gap = [0] * n
    max_prev = [-1] * n
    max_len_arr = [1] * n
    
    fgaps[0] = [0]
    fpmax[0] = [1]
    fprev[0] = [-1]
    
    best_len = 1
    best_i = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        pi = primes[i]
        # Collect (gap, length, prev_j) for all j < i
        # Use a list for efficiency
        candidates = []
        
        for j in range(i):
            gap = pi - primes[j]
            # Query frontier[j]
            prev_len = 1
            fj = fgaps[j]
            if fj is not None and len(fj) > 0:
                idx = bisect.bisect_right(fj, gap) - 1
                if idx >= 0:
                    prev_len = fpmax[j][idx]
            candidates.append((gap, prev_len + 1, j))
        
        # Sort by gap
        candidates.sort(key=lambda x: x[0])
        
        # Build Pareto frontier: for each unique gap, keep max length, then prefix-max, keep only improvements
        fg = []
        fp = []
        fpj = []
        cur_max = 0
        k = 0
        nc = len(candidates)
        while k < nc:
            g = candidates[k][0]
            max_l = candidates[k][1]
            pj = candidates[k][2]
            k += 1
            while k < nc and candidates[k][0] == g:
                if candidates[k][1] > max_l:
                    max_l = candidates[k][1]
                    pj = candidates[k][2]
                k += 1
            if max_l > cur_max:
                cur_max = max_l
                fg.append(g)
                fp.append(cur_max)
                fpj.append(pj)
        
        fgaps[i] = fg
        fpmax[i] = fp
        fprev[i] = fpj
        
        # The max length for prime i is the last entry in the frontier (or 1 if empty)
        ml = fp[-1] if fp else 1
        max_len_arr[i] = ml
        if fp:
            max_gap[i] = fg[-1]
            max_prev[i] = fpj[-1]
        
        if ml > best_len:
            best_len = ml
            best_i = i
    
    if best_len <= 1:
        return primes[:2]
    
    # Reconstruct the sequence by backtracking
    seq = [primes[best_i]]
    i = best_i
    while len(seq) < best_len:
        j = max_prev[i]
        if j < 0:
            break
        seq.append(primes[j])
        i = j
    
    seq.reverse()
    
    # Validate
    if len(seq) < 2:
        return primes[:2]
    
    for p in seq:
        if p > 10000 or not is_prime[p]:
            return primes[:2]
    
    for idx in range(len(seq) - 1):
        if seq[idx] >= seq[idx+1]:
            return primes[:2]
    
    gaps = [seq[idx+1] - seq[idx] for idx in range(len(seq) - 1)]
    for idx in range(len(gaps) - 1):
        if gaps[idx] > gaps[idx+1]:
            return primes[:2]
    
    return seq
# EVOLVE-BLOCK-END
