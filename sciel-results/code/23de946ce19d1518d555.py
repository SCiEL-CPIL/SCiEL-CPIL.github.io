# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Generate primes <= 10000
    def sieve(n):
        is_prime = np.ones(n + 1, dtype=bool)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        return np.where(is_prime)[0]
    
    primes = sieve(10000)
    n = len(primes)
    
    if n < 2:
        return list(primes)
    
    # dp[i] will store a dict or we use a 2D array
    # dp[i][g] = length of longest valid subsequence ending at prime i with last gap = g
    # But gaps can be up to 10000, and n ~ 1229
    # Use: for each i, we store best[i] = array of size (max_gap+1) where best[i][g] = max length ending at i with last gap exactly g
    # Then for transition: for pair (j, i) with gap g, dp_val = 1 + max(best[j][0:g+1])
    
    # To speed up, for each j we maintain a prefix-max array
    # prefix_best[j][g] = max(best[j][0], best[j][1], ..., best[j][g])
    
    max_gap = primes[-1] - primes[0]
    
    # best[i][g] = longest subseq ending at prime i with last gap exactly g
    # We'll store as a list of arrays
    best = [None] * n
    prefix_best = [None] * n
    
    # Initialize: each prime alone has length 1, gap = 0 (no previous)
    # Actually, a single element has no gap. Let's say length 1 with "gap" = 0
    for i in range(n):
        arr = np.zeros(max_gap + 1, dtype=np.int32)
        arr[0] = 1  # single element, no gap
        best[i] = arr
        prefix_best[i] = np.maximum.accumulate(arr)
    
    # Transition: for i from 1 to n-1, for j from 0 to i-1:
    # gap = primes[i] - primes[j]
    # dp_val = 1 + prefix_best[j][gap]  (max length ending at j with last gap <= gap)
    # update best[i][gap] = max(best[i][gap], dp_val)
    
    # But this is O(n^2 * max_gap) if we recompute prefix each time... 
    # Actually prefix_best[j] is precomputed, so each transition is O(1) lookup
    # Total: O(n^2) transitions, each O(1) with precomputed prefix
    
    # Wait, but after updating best[i], we need to recompute prefix_best[i] before i is used
    # Since we process i in order, and for each i we update best[i] from all j < i,
    # we can compute prefix_best[i] after all updates to i are done
    
    # Actually no - we update best[i][gap] potentially multiple times (from different j's),
    # so we need all updates done before computing prefix_best[i]
    
    # Let's restructure: for each i, collect all candidate updates, then finalize
    
    # Better: process i from 0 to n-1. For each i, first compute all transitions from j < i,
    # then finalize prefix_best[i].
    
    # But we already initialized best[i] with arr[0]=1. Now for each j < i:
    # gap = primes[i] - primes[j]
    # candidate = 1 + prefix_best[j][gap]
    # best[i][gap] = max(best[i][gap], candidate)
    
    # After all j processed for a given i, compute prefix_best[i] = max.accumulate(best[i])
    
    # But we need prefix_best[j] for j < i to already be computed. Yes, since we go in order.
    
    # Let's redo:
    best = [None] * n
    prefix_best = [None] * n
    
    # Prime 0: just itself
    arr0 = np.zeros(max_gap + 1, dtype=np.int32)
    arr0[0] = 1
    best[0] = arr0
    prefix_best[0] = np.maximum.accumulate(arr0)
    
    for i in range(1, n):
        # Initialize best[i] with the "single element" option
        arr_i = np.zeros(max_gap + 1, dtype=np.int32)
        arr_i[0] = 1  # starting a new sequence at i
        
        for j in range(i):
            gap = primes[i] - primes[j]
            if gap <= max_gap:
                candidate = 1 + prefix_best[j][gap]
                if candidate > arr_i[gap]:
                    arr_i[gap] = candidate
        
        best[i] = arr_i
        prefix_best[i] = np.maximum.accumulate(arr_i)
        
        if time.time() > deadline:
            break
    
    # Find the overall maximum length and reconstruct
    max_len = 0
    best_end_i = 0
    best_end_gap = 0
    for i in range(n):
        if prefix_best[i] is not None:
            # max length ending at i is prefix_best[i][-1] (max over all gaps)
            ml = int(prefix_best[i][-1])
            if ml > max_len:
                max_len = ml
                best_end_i = i
                # find the gap that achieves this
                # prefix_best[i][-1] is the max, find which gap index gives it
                # Actually we need to find the gap g where best[i][g] == max_len
                g_idx = np.argmax(best[i])
                best_end_gap = int(g_idx)
    
    # Reconstruct the sequence
    # We need parent pointers. Let me add those.
    
    # Actually, let me redo with parent tracking.
    
    # Redo with parents:
    # parent[i][g] = the index j that was the predecessor when best[i][g] was set
    # parent_gap[i][g] = the gap that was used at j (i.e., primes[j] - primes[parent[i][g]])
    
    # This is getting complex. Let me just do a simpler approach given the deadline.
    
    # Simpler: just return the best sequence found by greedy, or do a limited DP
    # Given the seed scores 1.0, let's try to do a proper DP with parent tracking
    
    # Let me restart with parent tracking properly
    
    best = [None] * n
    prefix_best = [None] * n
    parent_idx = [None] * n  # parent_idx[i][g] = j
    parent_gap = [None] * n  # parent_gap[i][g] = the gap at j (primes[j] - primes[k])
    
    arr0 = np.zeros(max_gap + 1, dtype=np.int32)
    arr0[0] = 1
    best[0] = arr0
    prefix_best[0] = np.maximum.accumulate(arr0)
    parent_idx[0] = np.full(max_gap + 1, -1, dtype=np.int32)
    parent_gap[0] = np.full(max_gap + 1, 0, dtype=np.int32)
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        arr_i = np.zeros(max_gap + 1, dtype=np.int32)
        arr_i[0] = 1
        par_i = np.full(max_gap + 1, -1, dtype=np.int32)
        par_gap_i = np.zeros(max_gap + 1, dtype=np.int32)
        
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            if gap > max_gap:
                continue
            candidate = 1 + int(prefix_best[j][gap])
            if candidate > arr_i[gap]:
                arr_i[gap] = candidate
                par_i[gap] = j
                # The gap at j that achieved prefix_best[j][gap]:
                # We need to find which g' <= gap gives the max in best[j]
                # prefix_best[j][gap] = max(best[j][0..gap])
                # We need to track which gap achieves this max
                # This is tricky. Let's store the argmax as well.
        
        best[i] = arr_i
        prefix_best[i] = np.maximum.accumulate(arr_i)
        parent_idx[i] = par_i
        parent_gap[i] = par_gap_i
    
    # This parent tracking is getting complicated. Let me use a different strategy:
    # Just find the max length, then do a backward search to reconstruct.
    
    # Actually, for the purpose of this task, let me just try a cleaner DP.
    
    # New approach: dp[i] = list of (gap, length) pairs, but that's not efficient.
    
    # Let me just go with the greedy seed approach but try to improve it slightly,
    # or implement a proper DP with careful parent tracking.
    
    # Given time constraints, let me implement a clean O(n^2) DP with parent pointers.
    
    # dp[i] = (max_length, last_gap, prev_index) for the best sequence ending at i
    # But that's not sufficient because we need to track the gap constraint.
    
    # OK let me just do the full DP properly.
    
    # State: best[i][g] = max length of valid subsequence ending at prime i with last gap = g
    # Transition: best[i][g] = max(best[i][g], 1 + max{best[j][g'] : g' <= g, g = p[i]-p[j]})
    
    # For reconstruction, store: par[i][g] = j, and par_g[i][g] = g' (the gap at j)
    
    # To find g' efficiently, I'll store for each j and each gap threshold, the argmax gap.
    
    # Let me precompute argmax_prefix[j][t] = the gap g' in [0,t] that maximizes best[j][g']
    
    n = len(primes)
    max_gap = int(primes[-1] - primes[0])
    
    best = np.zeros((n, max_gap + 1), dtype=np.int32)
    argmax_prefix = np.zeros((n, max_gap + 1), dtype=np.int32)
    
    # Initialize
    for i in range(n):
        best[i, 0] = 1
        argmax_prefix[i, 0] = 0
    
    # Compute argmax_prefix for i=0
    for t in range(1, max_gap + 1):
        if best[0, t] >= best[0, argmax_prefix[0, t-1]]:
            argmax_prefix[0, t] = t
        else:
            argmax_prefix[0, t] = argmax_prefix[0, t-1]
    
    # Parent tracking
    par_i = np.full((n, max_gap + 1), -1, dtype=np.int32)
    par_g = np.zeros((n, max_gap + 1), dtype=np.int32)
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        pi = int(primes[i])
        for j in range(i):
            g = pi - int(primes[j])
            if g > max_gap:
                continue
            # best[j, g'] for g' <= g: the max is at g' = argmax_prefix[j, g]
            g_star = int(argmax_prefix[j, g])
            candidate = 1 + int(best[j, g_star])
            if candidate > best[i, g]:
                best[i, g] = candidate
                par_i[i, g] = j
                par_g[i, g] = g_star
        
        # Update argmax_prefix for i
        for t in range(1, max_gap + 1):
            if best[i, t] >= best[i, argmax_prefix[i, t-1]]:
                argmax_prefix[i, t] = t
            else:
                argmax_prefix[i, t] = argmax_prefix[i, t-1]
    
    # Find the overall best
    max_len = 0
    best_i = 0
    best_g = 0
    for i in range(n):
        # max over all gaps for prime i
        ml = int(best[i, :].max()) if max_gap + 1 > 0 else 0
        if ml > max_len:
            max_len = ml
            best_i = i
            best_g = int(np.argmax(best[i, :]))
    
    # Reconstruct
    seq = []
    i = best_i
    g = best_g
    while i >= 0 and len(seq) < n + 1:
        seq.append(int(primes[i]))
        if par_i[i, g] == -1:
            break
        i, g = int(par_i[i, g]), int(par_g[i, g])
    
    seq.reverse()
    
    if not seq:
        seq = [2]
    
    return seq
# EVOLVE-BLOCK-END
