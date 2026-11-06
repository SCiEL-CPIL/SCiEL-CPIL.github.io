"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    import time
    
    deadline = time.time() + 85.0
    
    n = 10000
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    N = len(primes)
    
    if N < 2:
        return primes
    
    # dp[i] = Pareto frontier of (last_gap, length, prev) sorted by gap
    dp = [[] for _ in range(N)]
    dp[0] = [(0, 1, -1)]
    
    best_len = 1
    best_end = 0
    # Track best predecessor for reconstruction
    best_prev = [-1] * N
    best_gap_used = [0] * N
    
    for i in range(1, N):
        frontier = []
        for j in range(i):
            gap = primes[i] - primes[j]
            dj = dp[j]
            if not dj:
                continue
            lo, hi = 0, len(dj)
            while lo < hi:
                mid = (lo + hi) // 2
                if dj[mid][0] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            pos = lo
            if pos > 0:
                entry = dj[pos - 1]
                new_len = entry[1] + 1
                frontier.append((gap, new_len, j))
        
        if frontier:
            frontier.sort(key=lambda x: (x[0], -x[1]))
            pf = []
            for g, l, p in frontier:
                if pf and pf[-1][0] == g:
                    if l > pf[-1][1]:
                        pf[-1] = (g, l, p)
                else:
                    if pf and pf[-1][1] >= l:
                        continue
                    pf.append((g, l, p))
            if not pf or pf[0][0] > 0:
                pf.insert(0, (0, 1, -1))
            elif pf[0][0] == 0 and pf[0][1] < 1:
                pf[0] = (0, 1, -1)
            dp[i] = pf
            # Find max length entry and track its prev
            max_l = 0
            max_entry = None
            for e in pf:
                if e[1] > max_l:
                    max_l = e[1]
                    max_entry = e
            if max_l > best_len:
                best_len = max_l
                best_end = i
                best_prev[i] = max_entry[2]
                best_gap_used[i] = max_entry[0]
        else:
            dp[i] = [(0, 1, -1)]
        
        if time.time() > deadline:
            break
    
    # Reconstruct
    result = []
    idx = best_end
    for _ in range(best_len):
        result.append(primes[idx])
        idx = best_prev[idx]
        if idx == -1:
            break
    result.reverse()
    
    return result
# EVOLVE-BLOCK-END
