"""Exact DP approach for longest prime subsequence with non-decreasing gaps."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # 1. Generate primes up to 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n_primes = len(primes)
    
    # 2. Dynamic Programming
    # We want to find the longest sequence p_0, p_1, ..., p_k such that
    # p_0 < p_1 < ... < p_k
    # and (p_{i+1} - p_i) >= (p_i - p_{i-1}) for all i >= 1.
    
    # Let dp[i][j] be the length of the longest valid sequence ending with the pair (primes[i], primes[j])
    # where i < j.
    # The gap of the last step is primes[j] - primes[i].
    # To extend this sequence to a new prime primes[k] (where k > j), we need:
    # (primes[k] - primes[j]) >= (primes[j] - primes[i])
    
    # Base case: Any pair (primes[i], primes[j]) with i < j is a valid sequence of length 2.
    # We can extend any pair (i, j) to k if primes[k] - primes[j] >= primes[j] - primes[i].
    
    # Optimization:
    # Instead of a full 3D DP (which is O(N^3) and might be slow in Python for N=1229 -> 1.8 billion ops),
    # we can iterate through the "last gap" constraint more efficiently or use a 2D DP with careful iteration.
    
    # Let's analyze the complexity. N=1229.
    # O(N^3) is ~ 1.8 * 10^9. This is too slow for pure Python within 50s.
    # We need a smarter DP or a different formulation.
    
    # Alternative DP State:
    # Let best_end[i] be the maximum length of a valid sequence ending at primes[i].
    # This is not sufficient because the validity of the next step depends on the *previous* gap, not just the last element.
    
    # We need to know the last gap.
    # Let dp[j][g] = max length of sequence ending at primes[j] with last gap exactly g?
    # Gaps can be up to 10000. This is a 2D array of size N x MaxGap.
    # MaxGap is roughly 10000. N is 1229. 1229 * 10000 = 12.29 million states.
    # Transition: To compute dp[j][g], we look for all i < j such that primes[j] - primes[i] = g.
    # If such an i exists, the previous gap g_prev must be <= g.
    # So dp[j][g] = 1 + max( dp[i][g_prev] for all g_prev <= g where i is the unique prime such that primes[j]-primes[i]=g )
    # Wait, i is determined by j and g. primes[i] = primes[j] - g.
    # If primes[j] - g is a prime, say primes[i], then we can transition from any sequence ending at primes[i] with last gap <= g.
    
    # So:
    # Let max_len_ending_at_i_with_gap_le_X be a data structure we can query.
    # Actually, for each prime index i, we can maintain a list of (gap, max_length) pairs, or a Fenwick tree / Segment tree for max query over gaps.
    # But gaps are not dense integers 1..10000 in a simple way? They are differences of primes.
    # However, the gap value g is an integer.
    
    # Let's refine the DP:
    # dp[i] is a dictionary or array where dp[i][g] = max length of sequence ending at primes[i] with last gap g.
    # To update dp[j][g] where g = primes[j] - primes[i]:
    # We need the max value of dp[i][g_prev] for all g_prev <= g.
    
    # Since we iterate j from 0 to N-1, and for each j we consider all i < j,
    # we can maintain for each i a data structure that allows "max query over range [0, g]".
    # Since max gap is ~10000, we can use a simple array of size 10001 for each i?
    # That would be 1229 * 10001 integers. ~12M integers. Memory is fine.
    # Time to update: For each j, we iterate i from 0 to j-1.
    # For each i, we calculate g = primes[j] - primes[i].
    # We query the max length in dp[i] for gaps <= g.
    # Then we update dp[j][g] = max(dp[j][g], queried_max + 1).
    
    # To make the query fast, we can pre-compute prefix maxima for each i's gap array?
    # But dp[i] is being built as we go? No, we build dp[j] using dp[i] where i < j.
    # So when we are at step j, all dp[i] for i < j are fully computed.
    # Therefore, for each i, we can maintain a "prefix_max" array or a sorted list of (gap, max_len) that allows O(log N) or O(1) amortized queries.
    
    # Simplest approach:
    # For each i, maintain an array `max_len_for_gap_le` of size MaxGap+1.
    # `max_len_for_gap_le[i][g]` = max length of sequence ending at primes[i] with last gap <= g.
    # When we add a new transition to dp[i] with gap g_new and length L_new:
    # We update `max_len_for_gap_le[i][g_new]` = max(current, L_new).
    # But we need the prefix max. We can update the suffix? No, prefix.
    # If we update `dp[i][g_new]`, then `max_len_for_gap_le[i][g]` for all g >= g_new might change.
    # This update is O(MaxGap) per transition.
    # Number of transitions is O(N^2). Total time O(N^2 * MaxGap) = 1229^2 * 10000 ~ 1.5 * 10^10. Too slow.
    
    # Better: Use a Fenwick Tree (Binary Indexed Tree) for each i to support point updates and prefix max queries in O(log MaxGap).
    # Or, since we only query prefix max [0, g], and updates are point updates, Fenwick tree is perfect.
    # MaxGap is 10000. Log2(10000) is ~14.
    # Total operations: N^2 * Log(MaxGap) = 1.5M * 14 = 21M operations. This is very fast in Python.
    
    import numpy as np
    
    # We need a Fenwick Tree for Max.
    # Standard Fenwick for max is a bit tricky because max is not invertible, but for prefix max queries and point updates (where update value is always increasing or we just take max), it works if we initialize with -inf or 0.
    # Actually, standard Fenwick for max supports:
    # update(i, val): tree[i] = max(tree[i], val) for i, i+(i&-i), ...
    # query(i): max of tree[i], tree[i-(i&-i)], ...
    # This works for prefix max if we only do "max" updates.
    
    class FenwickMax:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (size + 1) # 1-indexed
        
        def update(self, idx, val):
            # idx is 1-based in tree, but we use 0-based gap + 1
            i = idx + 1
            while i <= self.size:
                if val > self.tree[i]:
                    self.tree[i] = val
                else:
                    # Optimization: if val <= tree[i], we might still need to check children?
                    # No, in a max Fenwick tree, if the current node is already >= val, 
                    # and the update is "max", the ancestors might already be >= val?
                    # Not necessarily. But if we just set tree[i] = max(tree[i], val), it's safe.
                    pass
                i += i & -i
        
        def query(self, idx):
            # Prefix max up to idx (0-based)
            # Returns max of range [0, idx]
            res = 0
            i = idx + 1
            while i > 0:
                if self.tree[i] > res:
                    res = self.tree[i]
                i -= i & -i
            return res

    # Max possible gap is limit (10000).
    max_gap = limit
    # We need a Fenwick tree for each prime index i.
    # 1229 Fenwick trees of size 10001.
    # Memory: 1229 * 10001 * 8 bytes (int) ~ 100 MB. This is acceptable.
    
    # To save memory and time, we can use a list of lists or numpy arrays.
    # Let's use a list of FenwickMax objects.
    
    bit_trees = [FenwickMax(max_gap) for _ in range(n_primes)]
    
    # dp_len[i] will store the max length ending at i, but we need to track the specific gap for the final answer reconstruction.
    # Actually, we just need the max length overall.
    # But to reconstruct the path, we need to store the predecessor.
    # Let's store:
    # best_len[i] = max length of any sequence ending at i.
    # prev_idx[i] = the index i_prev such that the sequence ending at i with length best_len[i] came from i_prev.
    # prev_gap[i] = the gap used in that transition.
    
    # However, the "best" length ending at i might come from a specific gap.
    # We need to track, for each i, the best length achievable and the parameters that achieved it.
    
    # Let's refine:
    # When we compute a candidate length L for ending at j with gap g (coming from i):
    # If L > best_len[j], we update best_len[j] = L, prev_idx[j] = i, prev_gap[j] = g.
    # If L == best_len[j], we don't care (any valid path is fine).
    
    best_len = [1] * n_primes # Base case: single prime is length 1? 
    # Wait, the problem asks for sequences. A single prime has no gaps. 
    # "successive gaps nondecreasing" is vacuously true for length 1 or 2.
    # But we want to maximize length.
    # Let's initialize best_len[i] = 1.
    # For any pair (i, j) with i < j, length is at least 2.
    
    prev_idx = [-1] * n_primes
    prev_gap = [0] * n_primes
    
    # Initialize Fenwick trees with the base case?
    # A sequence of length 1 ending at i has "last gap" 0? Or undefined?
    # If we consider the transition from "start" to i, the gap is 0?
    # If we start with i, and then go to j, the gap is primes[j]-primes[i].
    # The condition is gap_j >= gap_prev.
    # If there is no gap_prev, the condition is always true.
    # So we can treat the "initial state" as having gap 0 and length 1.
    # So for each i, we update bit_trees[i] with gap 0 and value 1.
    
    for i in range(n_primes):
        bit_trees[i].update(0, 1) # Gap 0, length 1
    
    # Now iterate j from 1 to N-1
    for j in range(1, n_primes):
        pj = primes[j]
        # Iterate i from 0 to j-1
        for i in range(j):
            pi = primes[i]
            g = pj - pi
            # Query max length ending at i with last gap <= g
            # bit_trees[i] stores lengths for sequences ending at i.
            # The query returns the max length L_prev such that the last gap of that sequence <= g.
            L_prev = bit_trees[i].query(g)
            
            # The new sequence length ending at j with last gap g is L_prev + 1
            L_new = L_prev + 1
            
            # Update bit_trees[j] for gap g
            # We only need to update if L_new is better than what's currently in the tree for gap g?
            # The Fenwick tree update handles the max logic.
            bit_trees[j].update(g, L_new)
            
            # Update global best for j
            if L_new > best_len[j]:
                best_len[j] = L_new
                prev_idx[j] = i
                prev_gap[j] = g
    
    # Find the global maximum length
    max_length = max(best_len)
    end_idx = best_len.index(max_length)
    
    # Reconstruct the sequence
    seq = []
    cur = end_idx
    while cur != -1:
        seq.append(primes[cur])
        cur = prev_idx[cur]
    
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
