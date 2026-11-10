"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import bisect
import numpy as np

def construct():
    start = time.time()
    deadline = start + 50
    
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
    
    # DP approach:
    # dp[j] = list of (last_gap, length, prev_idx) Pareto-optimal states ending at prime j
    # A state (g, l, p) means: subsequence ending at primes[j], last gap is g, length l, previous prime index is p
    # Pareto-optimal: no other state has both smaller/equal last_gap AND larger/equal length
    
    # For efficiency, we process primes in order and maintain Pareto frontiers
    
    # states[j] = sorted list of (gap, length) pairs, where for each gap we keep the max length
    # We want to query: for prime j and a given gap threshold G, what's the max length of a 
    # subsequence ending at some i < j with last_gap <= G?
    
    # Let's use a different formulation:
    # For each prime j, we want to find the longest subsequence ending at j.
    # A subsequence ending at j with previous element at i has:
    #   gap = primes[j] - primes[i]
    #   length = 1 + (length of best subsequence ending at i with last_gap <= gap)
    
    # So we need: for each i, and for any threshold T, the max length of subsequence ending at i with last_gap <= T.
    
    # Let's store for each i: a sorted list of (last_gap, max_length) pairs.
    # Then querying max length with last_gap <= T is a binary search.
    
    # states[i] = list of (gap, length) sorted by gap, with length being the max length achievable with that last_gap
    # Actually, we want: for a given threshold T, max over all states with gap <= T of length.
    # So we can store: for each i, a list of (gap, length) where length is non-increasing as gap increases.
    # Then we binary search for the rightmost gap <= T and take the length at that position.
    
    # Let me restructure:
    # frontier[i] = list of (gap, length) pairs, sorted by gap ascending.
    # For each consecutive pair, length is non-increasing (otherwise the earlier one dominates).
    # Query: max length with gap <= T -> binary search for rightmost gap <= T, return that length.
    
    # But actually, we want the MAX length among all states with gap <= T.
    # If we sort by gap ascending and keep only Pareto-optimal states (length non-increasing),
    # then the max length with gap <= T is the length at the position of the largest gap <= T.
    
    # Let's implement this.
    
    # frontier[i] = list of (gap, length) sorted by gap, length non-increasing
    # To build frontier[j]:
    #   For each i < j:
    #     gap_ij = primes[j] - primes[i]
    #     best_len_i = query(frontier[i], gap_ij)  # max length with last_gap <= gap_ij
    #     candidate = (gap_ij, best_len_i + 1)
    #   Also, the single-element sequence: (0, 1) -- but we need at least 2 elements for gaps
    #   Actually, a sequence of length 1 has no gap. Let's handle length >= 2.
    #   For length 2: gap = primes[j] - primes[i], length = 2, for any i < j.
    
    # Let me reconsider. A subsequence of length k has k-1 gaps.
    # For a subsequence ending at j with previous element at i:
    #   If the subsequence has length 2: just [primes[i], primes[j]], gap = primes[j]-primes[i]
    #   If longer: the subsequence ending at i must have last_gap <= primes[j]-primes[i]
    
    # So: dp[j][i] = max length of subsequence ending at j with previous element at i
    #   = 2 if no valid extension from i
    #   = 1 + (max length of subsequence ending at i with last_gap <= primes[j]-primes[i])
    
    # The "max length of subsequence ending at i with last_gap <= T" is what we query from frontier[i].
    
    # Let's define:
    #   best_len[i][T] = max length of subsequence ending at prime i with last gap <= T
    #   This is a step function in T.
    
    # For each i, we maintain a list of breakpoints: [(gap_1, len_1), (gap_2, len_2), ...]
    # where gap_1 < gap_2 < ... and len_1 >= len_2 >= ...
    # best_len[i][T] = len_k where gap_k is the largest gap <= T (or 1 if no gap <= T, meaning only length-1 sequences)
    
    # Actually, for a subsequence of length 1 (just prime i), there's no last gap. We can think of its "last gap" as 0 or -infinity.
    # So best_len[i][T] >= 1 for any T >= 0.
    
    # Let's initialize: each prime i has a base state: length 1, last_gap = 0 (or -1 to indicate no gap).
    # When extending from i to j: new_gap = primes[j] - primes[i], new_length = best_len[i][new_gap] + 1
    # But best_len[i][new_gap] includes the length-1 base state, so new_length >= 2.
    
    # OK let me just implement this directly.
    
    # For each i, frontier[i] is a list of (gap, length) sorted by gap, with length non-increasing.
    # The base state is (0, 1) -- but gap 0 means "no previous element" or "last gap is 0".
    # Actually, let's use gap = -1 for the base state (length 1, no gap).
    
    # Hmm, this is getting complicated. Let me use a cleaner formulation.
    
    # Let f(i, T) = max length of a nondecreasing-gap subsequence ending at prime i, with last gap <= T.
    # Base: f(i, T) >= 1 for all T (the subsequence [primes[i]]).
    # Recurrence: f(i, T) = max(1, max over j < i of (f(j, primes[i]-primes[j]) + 1) where primes[i]-primes[j] <= T)
    #            = max(1, max over j < i with primes[i]-primes[j] <= T of (f(j, primes[i]-primes[j]) + 1))
    
    # For a fixed i, as T varies, f(i, T) is a step function.
    # The breakpoints occur at values of primes[i] - primes[j] for j < i.
    
    # Let's compute f(i, T) for all relevant T values.
    
    # For prime i, the relevant T values are: primes[i] - primes[j] for j = 0, ..., i-1.
    # Let's sort these: g_1 < g_2 < ... < g_{i} where g_k = primes[i] - primes[i-k] (roughly).
    
    # Actually, let's just compute for each pair (j, i) with j < i:
    #   gap = primes[i] - primes[j]
    #   f(i, gap) >= f(j, gap) + 1
    
    # And we want the max over all i of f(i, infinity).
    
    # Let's use the frontier representation:
    # For each i, we maintain a dict or sorted list mapping gap -> max_length.
    # After processing all j < i, we take the Pareto frontier.
    
    # Let's implement with lists.
    
    # frontier[i] = sorted list of (gap, length) where length is the max length for that exact gap,
    # and we've removed dominated entries (if gap1 < gap2 and len1 >= len2, remove (gap2, len2)).
    
    # To query f(i, T): binary search for the largest gap <= T in frontier[i], return that length.
    # If no gap <= T, return 1 (base case).
    
    # Let's code this up.
    
    # To save memory and time, we'll process primes one by one.
    
    # For each prime index i (from 0 to n-1):
    #   Start with frontier[i] = []
    #   For each j < i:
    #     gap = primes[i] - primes[j]
    #     # Query frontier[j] for max length with last_gap <= gap
    #     best_prev = 1  # base: just [primes[j]]
    #     if frontier[j]:
    #         # Binary search in frontier[j] for rightmost gap <= gap
    #         # frontier[j] is sorted by gap ascending
    #         # We want the rightmost entry with gap_val <= gap
    #         # Since lengths are non-increasing, the rightmost such entry has the smallest length
    #         # Wait, no. We want the MAX length with gap <= T.
    #         # If lengths are non-increasing with gap, then the max length with gap <= T is at the SMALLEST gap.
    #         # That's not right either.
    #         
    #         # Let me reconsider. The Pareto frontier has entries (g1, l1), (g2, l2), ... with g1 < g2 < ... and l1 >= l2 >= ...
    #         # For a query T, we want max l_k such that g_k <= T.
    #         # Since l_k is non-increasing, the max l_k with g_k <= T is l_k where g_k is the LARGEST gap <= T.
    #         # No wait, that gives the SMALLEST l_k.
    #         
    #         # I think I have the ordering wrong. Let me reconsider.
    #         # Entry (g, l) means: there exists a subsequence ending at i with last gap exactly g and length l.
    #         # Pareto-optimal: no entry (g', l') has g' <= g AND l' >= l (with at least one strict).
    #         # So if we sort by g ascending, l must be strictly decreasing (otherwise earlier entries dominate).
    #         # For query T: max l over entries with g <= T.
    #         # Since l is strictly decreasing with g, the max l with g <= T is at the SMALLEST g (i.e., g=first entry if first g <= T).
    #         # That's not helpful for binary search.
    #         
    #         # Alternative: sort by g ascending, and for each g, store the MAX l over all entries with gap <= g.
    #         # I.e., store a prefix-max array. Then query T is just binary search for rightmost g <= T, and return the prefix-max value.
    
    # Let me restructure:
    # For each i, after computing all (gap, length) pairs, I'll:
    #   1. Sort by gap
    #   2. For each gap, keep the max length
    #   3. Compute prefix max of lengths
    #   4. Remove dominated entries (where prefix max doesn't increase)
    #   5. Store as (gaps_array, prefix_max_lengths_array)
    # Query: binary search for rightmost gap <= T, return prefix_max_lengths at that position.
    
    # This is O(i log i) per prime i, total O(n^2 log n).
    # n = 1229, so n^2 log n ≈ 1.5M * 11 ≈ 16M operations. Should be fine.
    
    # Let's implement.
    
    # gaps_list[i] = sorted array of unique gaps
    # pmax_list[i] = prefix max of lengths at each gap
    
    # Actually, let's just use Python lists for simplicity, and optimize if needed.
    
    # frontier[i] = list of (gap, prefix_max_length) sorted by gap
    # To build: collect all (gap, length) pairs, sort, compute prefix max, keep only where prefix max increases.
    
    # Let's code:
    
    # For i = 0: frontier[0] = [(0, 1)]  -- base state, gap 0, length 1
    # Actually, for i=0, there's no j < 0, so only the base state.
    
    # For i > 0:
    #   candidates = [(0, 1)]  # base state
    #   For j = 0 to i-1:
    #     gap = primes[i] - primes[j]
    #     # Query frontier[j] for max length with last_gap <= gap
    #     prev_len = 1  # default: just [primes[j]]
    #     if frontier[j]:
    #         # Binary search in frontier[j] for rightmost gap <= gap
    #         # frontier[j] is list of (gap_val, pmax_len) sorted by gap_val
    #         # Use bisect on the gap values
    #         # Extract gap values for binary search
    #         # For efficiency, let's store gaps and pmax separately
    #         pass
    #     candidates.append((gap, prev_len + 1))
    #   # Now build frontier[i] from candidates
    #   # Sort by gap, for each gap keep max length, compute prefix max, filter dominated
    
    # This is getting complex. Let me just implement it directly with some optimization.
    
    # Let me use a different approach: for each i, store a dict gap -> max_length,
    # then convert to sorted list with prefix max.
    
    # Actually, let me just use the dense DP but with numpy vectorization for the inner loop.
    
    # Dense DP: dp[j][i] = max length of subsequence ending at prime i with previous prime at j.
    # But this is O(n^2) space and O(n^3) time.
    
    # Let me try a smarter approach.
    
    # For each prime i, I want to compute:
    #   f(i) = max over j < i of (f_ext(j, primes[i]-primes[j]) + 1)
    #   where f_ext(j, T) = max length of subsequence ending at j with last_gap <= T
    
    # And I want to track the full function f_ext(i, T) for all T.
    
    # Let me just implement the frontier approach with careful coding.
    
    # Store for each i: 
    #   fgaps[i] = sorted list of gap values
    #   fpmax[i] = corresponding prefix max lengths
    
    # Query: given i and T, find rightmost index in fgaps[i] where fgaps[i][idx] <= T, return fpmax[i][idx].
    
    # Let's implement:
    
    fgaps = [None] * n
    fpmax = [None] * n
    fprev = [None] * n  # to track which j gave the best result, for reconstruction
    
    # Base case: i = 0
    fgaps[0] = [0]
    fpmax[0] = [1]
    fprev[0] = []
    
    best_len = 1
    best_i = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        # Collect candidates: (gap, length, prev_j)
        candidates = [(0, 1, -1)]  # base state
        
        pi = primes[i]
        
        for j in range(i):
            gap = pi - primes[j]
            # Query fgaps[j], fpmax[j] for max length with last_gap <= gap
            prev_len = 1
            prev_idx_in_frontier = -1
            if fgaps[j]:
                # Binary search for rightmost gap <= gap in fgaps[j]
                # fgaps[j] is sorted ascending
                idx = bisect.bisect_right(fgaps[j], gap) - 1
                if idx >= 0:
                    prev_len = fpmax[j][idx]
                    prev_idx_in_frontier = idx
            
            candidates.append((gap, prev_len + 1, j))
        
        # Now build frontier[i] from candidates
        # Sort by gap
        candidates.sort(key=lambda x: x[0])
        
        # For each gap, keep the max (length, prev_j)
        # Then compute prefix max of lengths
        # Then filter dominated entries
        
        unique_gaps = []
        max_len_at_gap = []
        prev_j_at_gap = []
        
        k = 0
        while k < len(candidates):
            g = candidates[k][0]
            max_l = candidates[k][1]
            pj = candidates[k][2]
            k += 1
            while k < len(candidates) and candidates[k][0] == g:
                if candidates[k][1] > max_l:
                    max_l = candidates[k][1]
                    pj = candidates[k][2]
                k += 1
            unique_gaps.append(g)
            max_len_at_gap.append(max_l)
            prev_j_at_gap.append(pj)
        
        # Compute prefix max
        n_cand = len(unique_gaps)
        pmax = [0] * n_cand
        pmax_prev = [0] * n_cand
        cur_max = 0
        cur_pj = -1
        for k in range(n_cand):
            if max_len_at_gap[k] > cur_max:
                cur_max = max_len_at_gap[k]
                cur_pj = prev_j_at_gap[k]
            pmax[k] = cur_max
            pmax_prev[k] = cur_pj
        
        # Filter dominated: keep only entries where pmax increases
        fg = []
        fp = []
        fpj = []
        last_pmax = -1
        for k in range(n_cand):
            if pmax[k] > last_pmax:
                fg.append(unique_gaps[k])
                fp.append(pmax[k])
                fpj.append(pmax_prev[k])
                last_pmax = pmax[k]
        
        fgaps[i] = fg
        fpmax[i] = fp
        fprev[i] = fpj
        
        if fp[-1] > best_len:
            best_len = fp[-1]
            best_i = i
    
    if best_len <= 1:
        return primes[:2]
    
    # Reconstruct the sequence
    # We need to trace back from best_i.
    # For prime i, the best state has length fpmax[i][-1] and previous prime index fprev[i][-1].
    # But we need to be more careful: the best state at i might not be the last entry in the frontier.
    
    # Actually, the max length for prime i is fpmax[i][-1] (the last prefix max).
    # The previous prime index is fprev[i][-1].
    # But we also need to know what gap was used, to continue the trace.
    
    # Hmm, this reconstruction is tricky with the prefix max approach.
    # Let me store additional info: for each i, the gap that achieves the max length.
    
    # Actually, let me just do a simple backtracking:
    # seq = [primes[best_i]]
    # i = best_i
    # while len(seq) < best_len:
    #   j = fprev[i][-1]  # previous prime index
    #   if j < 0:
    #       break
    #   seq.append(primes[j])
    #   i = j
    # seq.reverse()
    
    # But this might not give the correct sequence because the "best" state at i might have a specific gap,
    # and the previous state at j must have last_gap <= that gap.
    
    # Let me think about this differently. Let me store for each i:
    #   best_gap[i] = the gap that achieves the max length for prime i
    #   best_prev[i] = the previous prime index
    #   best_len[i] = the max length
    
    # Then reconstruction is straightforward.
    
    # Let me redo the computation to track this.
    
    # Actually, let me just recompute with explicit tracking.
    
    # For each i, after building the frontier, the max length is fpmax[i][-1].
    # The gap that achieves this is the gap at the position where the prefix max first reaches this value.
    # And the previous prime is fprev[i] at that position.
    
    # Let me store:
    #   max_gap[i] = the gap value where the overall max length is first achieved
    #   max_prev[i] = the previous prime index at that gap
    
    # Let me redo:
    
    fgaps = [None] * n
    fpmax = [None] * n
    fprev = [None] * n
    max_gap = [0] * n
    max_prev = [-1] * n
    max_len_arr = [1] * n
    
    fgaps[0] = [0]
    fpmax[0] = [1]
    fprev[0] = [-1]
    max_gap[0] = 0
    max_prev[0] = -1
    max_len_arr[0] = 1
    
    best_len = 1
    best_i = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        candidates = [(0, 1, -1)]
        
        pi = primes[i]
        
        for j in range(i):
            gap = pi - primes[j]
            prev_len = 1
            if fgaps[j]:
                idx = bisect.bisect_right(fgaps[j], gap) - 1
                if idx >= 0:
                    prev_len = fpmax[j][idx]
            
            candidates.append((gap, prev_len + 1, j))
        
        candidates.sort(key=lambda x: x[0])
        
        unique_gaps = []
        max_len_at_gap = []
        prev_j_at_gap = []
        
        k = 0
        while k < len(candidates):
            g = candidates[k][0]
            max_l = candidates[k][1]
            pj = candidates[k][2]
            k += 1
            while k < len(candidates) and candidates[k][0] == g:
                if candidates[k][1] > max_l:
                    max_l = candidates[k][1]
                    pj = candidates[k][2]
                k += 1
            unique_gaps.append(g)
            max_len_at_gap.append(max_l)
            prev_j_at_gap.append(pj)
        
        n_cand = len(unique_gaps)
        pmax = [0] * n_cand
        pmax_prev = [0] * n_cand
        cur_max = 0
        cur_pj = -1
        for k in range(n_cand):
            if max_len_at_gap[k] > cur_max:
                cur_max = max_len_at_gap[k]
                cur_pj = prev_j_at_gap[k]
            pmax[k] = cur_max
            pmax_prev[k] = cur_pj
        
        fg = []
        fp = []
        fpj = []
        last_pmax = -1
        for k in range(n_cand):
            if pmax[k] > last_pmax:
                fg.append(unique_gaps[k])
                fp.append(pmax[k])
                fpj.append(pmax_prev[k])
                last_pmax = pmax[k]
        
        fgaps[i] = fg
        fpmax[i] = fp
        fprev[i] = fpj
        
        # The max length for prime i is fp[-1]
        if fp[-1] > max_len_arr[i]:
            max_len_arr[i] = fp[-1]
            # Find the gap and prev that achieve this
            # The max is achieved at the last entry of the filtered frontier
            max_gap[i] = fg[-1]
            max_prev[i] = fpj[-1]
        
        if max_len_arr[i] > best_len:
            best_len = max_len_arr[i]
            best_i = i
    
    if best_len <= 1:
        return primes[:2]
    
    # Reconstruct
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
