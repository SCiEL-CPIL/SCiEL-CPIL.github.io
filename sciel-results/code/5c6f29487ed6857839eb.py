"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    # Step 1: Generate all primes up to 10000 using Sieve of Eratosthenes
    limit = 10000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    n = len(primes)

    # Step 2: Dynamic programming for longest subsequence with non-decreasing gaps
    # dp[i] = (length, last_gap, prev_index) for best sequence ending at primes[i]
    # We want to maximize length, and for ties, we prefer smaller last_gap (more flexible)
    
    # For each prime index i, we look at all j < i where primes[i] - primes[j] >= last_gap of sequence ending at j
    # To make this efficient, we can use the fact that we want non-decreasing gaps.
    
    # Let's define:
    # best[i] = maximum length of a valid subsequence ending at primes[i]
    # last_gap[i] = the gap of the last step in the best subsequence ending at primes[i]
    # prev[i] = index of previous prime in the best subsequence ending at primes[i]
    
    # But we need to be careful: for a given i, among all valid j's, we want the one that gives
    # the longest sequence, and if tied, the one with the smallest last gap.
    
    # Actually, let's think more carefully. If we have a subsequence ending at j with last gap g,
    # and we extend it to i, the new last gap is primes[i] - primes[j]. This must be >= g.
    # The new length is length[j] + 1.
    
    # For each i, we want to find the j < i that maximizes length[j] subject to primes[i] - primes[j] >= last_gap[j].
    # Among those with maximum length, we prefer the one with smallest last_gap[j] (so that future extensions are easier).
    
    # This is a 2D optimization which is tricky. Let's use a different approach.
    
    # Alternative: For each possible starting point and use a greedy/DP hybrid.
    
    # Let's use: dp[i] = list of (length, last_gap) pairs representing Pareto-optimal sequences ending at i.
    # A pair (l1, g1) dominates (l2, g2) if l1 >= l2 and g1 <= g2.
    
    # But this could blow up. Let's try a simpler approach first:
    
    # For each i, maintain:
    #   best_len[i] = max length of valid subsequence ending at i
    #   best_gap[i] = the last gap of the best (longest) subsequence ending at i; if tie, min gap
    #   prev[i] = previous index
    
    # For i from 0 to n-1:
    #   best_len[i] = 1, best_gap[i] = 0, prev[i] = -1
    #   For j from 0 to i-1:
    #     gap = primes[i] - primes[j]
    #     if gap >= best_gap[j]:
    #       candidate_len = best_len[j] + 1
    #       if candidate_len > best_len[i] or (candidate_len == best_len[i] and gap < best_gap[i]):
    #         best_len[i] = candidate_len
    #         best_gap[i] = gap
    #         prev[i] = j
    
    # This is O(n^2) which for n=1229 is about 1.5 million operations. Very fast.
    
    # But wait: is this correct? The issue is that best_gap[j] is the last gap of the BEST (longest) subsequence ending at j.
    # But there might be a shorter subsequence ending at j with a smaller last gap that allows a longer extension.
    
    # Example: subsequence A ending at j has length 5, last gap 100. Subsequence B ending at j has length 4, last gap 10.
    # If primes[i] - primes[j] = 50, we can extend B (50 >= 10) to get length 5, but not A (50 < 100).
    # So the best extension might come from a non-longest subsequence ending at j.
    
    # So we need to keep track of multiple states per j. Let's use the Pareto frontier approach.
    
    # For each i, maintain a list of (length, last_gap) pairs that are Pareto-optimal.
    # A pair (l, g) is kept if there's no other pair (l', g') with l' >= l and g' <= g and (l', g') != (l, g).
    
    # For each i:
    #   Start with [(1, 0)]
    #   For each j < i:
    #     gap = primes[i] - primes[j]
    #     For each (l, g) in states[j]:
    #       if gap >= g:
    #         candidate = (l+1, gap)
    #         Add to states[i] if not dominated
    #   Keep only Pareto-optimal states in states[i]
    
    # The number of Pareto-optimal states per i should be small in practice.
    
    states = [[] for _ in range(n)]
    prev_state = [None] * n  # For backtracking: (j, index_in_states_j)
    
    for i in range(n):
        # Start with the single-element subsequence
        current_states = [(1, 0)]
        
        if time.time() > deadline:
            break
        
        for j in range(i):
            gap = primes[i] - primes[j]
            for (l, g) in states[j]:
                if gap >= g:
                    candidate = (l + 1, gap)
                    # Check if candidate is dominated by any existing state in current_states
                    dominated = False
                    for (cl, cg) in current_states:
                        if cl >= candidate[0] and cg <= candidate[1]:
                            dominated = True
                            break
                    if not dominated:
                        # Remove any states dominated by candidate
                        new_states = []
                        for (cl, cg) in current_states:
                            if not (candidate[0] >= cl and candidate[1] <= cg):
                                new_states.append((cl, cg))
                        new_states.append(candidate)
                        current_states = new_states
        
        states[i] = current_states
    
    # Find the best state overall
    best_len = 0
    best_i = -1
    best_state_idx = -1
    for i in range(n):
        for idx, (l, g) in enumerate(states[i]):
            if l > best_len:
                best_len = l
                best_i = i
                best_state_idx = idx
    
    if best_i == -1:
        return [2]
    
    # Backtrack to reconstruct the sequence
    # We need to track which state led to each state. Let's redo with backtracking info.
    
    # Actually, let's redo the DP with backtracking information.
    
    # For each i and each state index in states[i], store (prev_i, prev_state_idx)
    # states[i][idx] = (length, last_gap, prev_i, prev_state_idx)
    
    states2 = [[] for _ in range(n)]
    
    for i in range(n):
        # Start with the single-element subsequence: (1, 0, -1, -1)
        current_states = [(1, 0, -1, -1)]
        
        if time.time() > deadline:
            break
        
        for j in range(i):
            gap = primes[i] - primes[j]
            for (l, g, pj, psi) in states2[j]:
                if gap >= g:
                    candidate = (l + 1, gap, j, psi)
                    # Check if candidate is dominated by any existing state in current_states
                    # Dominance: (l1, g1) dominates (l2, g2) if l1 >= l2 and g1 <= g2
                    dominated = False
                    for (cl, cg, cpj, cpsi) in current_states:
                        if cl >= candidate[0] and cg <= candidate[1]:
                            dominated = True
                            break
                    if not dominated:
                        # Remove any states dominated by candidate
                        new_states = []
                        for (cl, cg, cpj, cpsi) in current_states:
                            if not (candidate[0] >= cl and candidate[1] <= cg):
                                new_states.append((cl, cg, cpj, cpsi))
                        new_states.append(candidate)
                        current_states = new_states
        
        states2[i] = current_states
    
    # Find the best state overall
    best_len = 0
    best_i = -1
    best_state_idx = -1
    for i in range(n):
        for idx, (l, g, pj, psi) in enumerate(states2[i]):
            if l > best_len:
                best_len = l
                best_i = i
                best_state_idx = idx
    
    if best_i == -1:
        return [2]
    
    # Backtrack
    result = []
    cur_i = best_i
    cur_idx = best_state_idx
    while cur_i != -1:
        result.append(primes[cur_i])
        _, _, pj, psi = states2[cur_i][cur_idx]
        cur_i = pj
        cur_idx = psi
    
    result.reverse()
    return result
# EVOLVE-BLOCK-END
