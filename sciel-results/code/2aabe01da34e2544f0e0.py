import time
import bisect
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 50.0  # Safety margin

    N = 10000
    # Sieve of Eratosthenes
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)
    primes_arr = np.array(primes, dtype=np.int32)

    # dp[i] is a list of (gap, length, prev_idx, prev_entry_idx) tuples, sorted by gap.
    # Lengths are strictly increasing.
    dp = [[] for _ in range(n)]
    
    # Pre-extract gaps as separate lists for faster binary search
    dp_gaps = [None] * n  # dp_gaps[i] = list of gaps in dp[i]
    dp_lens = [None] * n  # dp_lens[i] = list of lengths in dp[i]

    def reduce_frontier(cands_gaps, cands_lens, cands_pi, cands_pei):
        """
        cands_gaps: list of gaps
        cands_lens: list of lengths
        cands_pi: list of prev_idx
        cands_pei: list of prev_entry_idx
        Returns (frontier, gaps_list, lens_list)
        """
        if not cands_gaps:
            return [], [], []
        
        # For each gap, keep max length (and corresponding pi, pei)
        # Use dict for O(1) lookup
        best = {}
        for idx in range(len(cands_gaps)):
            g = cands_gaps[idx]
            l = cands_lens[idx]
            pi = cands_pi[idx]
            pei = cands_pei[idx]
            if g not in best or l > best[g][0]:
                best[g] = (l, pi, pei)
        
        # Sort by gap
        sorted_items = sorted(best.items())
        
        frontier = []
        gaps_list = []
        lens_list = []
        max_len_so_far = 0
        for g, (l, pi, pei) in sorted_items:
            if l > max_len_so_far:
                frontier.append((g, l, pi, pei))
                gaps_list.append(g)
                lens_list.append(l)
                max_len_so_far = l
        return frontier, gaps_list, lens_list

    for i in range(n):
        if time.time() - t0 > DEADLINE:
            break
            
        pi_val = primes[i]
        cands_gaps = []
        cands_lens = []
        cands_pi = []
        cands_pei = []
        
        for j in range(i):
            gap = pi_val - primes[j]
            
            # Option 1: Start new sequence
            cands_gaps.append(gap)
            cands_lens.append(2)
            cands_pi.append(j)
            cands_pei.append(-1)
            
            # Option 2: Extend existing sequence from j
            if dp[j]:
                jgaps = dp_gaps[j]
                idx = bisect.bisect_right(jgaps, gap) - 1
                if idx >= 0:
                    prev_len = dp_lens[j][idx]
                    cands_gaps.append(gap)
                    cands_lens.append(prev_len + 1)
                    cands_pi.append(j)
                    cands_pei.append(idx)
        
        frontier, gaps_list, lens_list = reduce_frontier(
            cands_gaps, cands_lens, cands_pi, cands_pei
        )
        dp[i] = frontier
        dp_gaps[i] = gaps_list
        dp_lens[i] = lens_list

    # Find the best sequence
    best_len = 1
    best_i = 0
    best_k = -1
    
    for i in range(n):
        for k, (g, l, pi, pei) in enumerate(dp[i]):
            if l > best_len:
                best_len = l
                best_i = i
                best_k = k

    # Reconstruct
    if best_len <= 1:
        return [primes[0]]

    result = []
    cur_i = best_i
    cur_k = best_k
    
    while True:
        result.append(primes[cur_i])
        if cur_k == -1:
            break
        g, l, pi, pei = dp[cur_i][cur_k]
        cur_i = pi
        cur_k = pei
    
    result.reverse()
    return result
