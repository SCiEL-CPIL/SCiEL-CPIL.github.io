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
    
    # For each pair (j, k) with j < k, we want the longest valid subsequence ending at k
    # with previous element at j. The constraint is that the gap before j must be <= gap_jk.
    # We use a Pareto frontier approach: for each j, maintain a list of (gap, length, prev_index)
    # where gap is the last gap in the subsequence ending at j, and we keep only non-dominated states.
    
    # Pareto[j] = list of (last_gap, subseq_length, prev_index) for subsequences ending at j
    # We maintain this as a sorted list by last_gap, keeping only states where no other state
    # has both <= gap and >= length.
    
    # Initialize: for each j, the trivial subsequence [primes[0], ..., primes[j]] has length j+1
    # but we need to track the actual last gap.
    
    # Better approach: dp[j] is a dict or list of Pareto-optimal (last_gap, length, prev) tuples
    # For each k > j, we can extend any state in dp[j] where last_gap <= primes[k] - primes[j]
    
    # To make this efficient, we'll use the fact that we process j in order, and for each j,
    # we can query dp[j] for the best length given a gap constraint.
    
    # Let's use a different formulation: 
    # Let f[j][g] = longest subsequence ending at j with last gap exactly g? No, too many gaps.
    
    # Alternative: For each j, maintain a list of (gap, length) pairs that are Pareto optimal.
    # When processing k, we need max length among states in dp[j] with gap <= primes[k]-primes[j].
    
    # We can precompute for each j a sorted list of (gap, length) and use binary search.
    
    # But the number of Pareto states per j could be large. Let's try a simpler approach first:
    # Use the original DP but with careful Pareto tracking.
    
    # Actually, let's use the insight from inspiration 2: maintain for each j a list of
    # (last_gap, best_length) that are Pareto optimal, sorted by last_gap.
    
    # Pareto[j] will be a list of (gap, length, prev_idx) tuples, sorted by gap, where
    # no tuple is dominated by another (i.e., for any two tuples, if gap1 <= gap2 then length1 >= length2).
    
    # Initialize Pareto[0] = [] (no previous element)
    # For j > 0, Pareto[j] starts with the trivial extension from some i < j.
    
    # This is getting complex. Let me try a different approach: use the original DP but
    # ensure we're computing the correct Pareto frontier.
    
    # For each (j, k), dp[j][k] should be the maximum length of a valid subsequence ending at k
    # with second-to-last element at j. This requires that there exists i < j such that
    # primes[j] - primes[i] <= primes[k] - primes[j] and the subsequence ...i, j, k is valid.
    
    # The key insight: for fixed j and k, we want max over i < j with primes[i] >= 2*primes[j] - primes[k]
    # of (dp[i][j] + 1), or 2 if no such i exists.
    
    # We can precompute for each j a data structure that allows querying:
    # given threshold T, find max dp[i][j] for i < j with primes[i] >= T.
    
    # Since primes[i] is increasing in i, we can use a suffix maximum array.
    
    # Let's rebuild the DP more carefully.
    
    dp = np.zeros((n, n), dtype=np.int32)
    prev = np.full((n, n), -1, dtype=np.int32)
    
    best_len = 2
    best_j, best_k = 0, 1
    
    # Convert primes to numpy array for faster bisect? No, bisect works on lists.
    # But we can precompute the primes list for bisect.
    
    for j in range(1, n):
        if time.time() > deadline:
            break
        
        # Build suffix max of dp[i][j] for i = 0..j-1
        # suff_val[i] = max(dp[i][j], dp[i+1][j], ..., dp[j-1][j])
        # suff_arg[i] = argmax
        # We can do this with a simple loop, but let's see if we can vectorize or speed up.
        # Since j is small relative to n initially, and grows, the total work is O(n^2).
        # The inner loop for k is O(n), and the suffix max is O(j). Total O(n^2).
        
        suff_val = np.zeros(j + 1, dtype=np.int32)
        suff_arg = np.full(j + 1, -1, dtype=np.int32)
        
        # dp[:, j] is the column j. We need suffix max of dp[0:j, j].
        col_j = dp[:j, j]
        
        # Compute suffix max and argmax
        # We need: for each position i, the max of dp[0:i, j] (i.e., dp[0][j], dp[1][j], ..., dp[i-1][j])
        # Wait, no. We need: for a given threshold index lo, find max dp[i][j] for i >= lo and i < j.
        # So suff_val[i] = max(dp[i][j], dp[i+1][j], ..., dp[j-1][j])
        
        # Let me recompute: we want for each i from 0 to j-1, the max of dp[i:j, j]
        # i.e., max of dp[i][j], dp[i+1][j], ..., dp[j-1][j]
        
        # Actually, looking at the code: lo is the first index where primes[lo] >= threshold.
        # We want max dp[i][j] for i in [lo, j-1].
        # So suff_val[i] should be max(dp[i][j], dp[i+1][j], ..., dp[j-1][j])
        
        # That's what the original code does. Let me keep it but fix the logic.
        
        current_max = 0
        current_arg = -1
        for i in range(j - 1, -1, -1):
            if col_j[i] > current_max:
                current_max = col_j[i]
                current_arg = i
            suff_val[i] = current_max
            suff_arg[i] = current_arg
        
        # Now for each k > j
        for k in range(j + 1, n):
            gap_jk = primes[k] - primes[j]
            threshold = primes[j] - gap_jk
            
            # Find the first index lo such that primes[lo] >= threshold
            # bisect_left returns the insertion point.
            lo = bisect.bisect_left(primes, threshold)
            
            if lo >= j:
                # No valid i exists, so the best is just the pair (j, k) with length 2
                dp[j, k] = 2
                prev[j, k] = -1
            else:
                # We want max dp[i][j] for i in [lo, j-1]
                bv = suff_val[lo]
                ba = suff_arg[lo]
                if bv > 0:
                    dp[j, k] = bv + 1
                    prev[j, k] = ba
                else:
                    dp[j, k] = 2
                    prev[j, k] = -1
            
            if dp[j, k] > best_len:
                best_len = dp[j, k]
                best_j, best_k = j, k
    
    if best_len <= 2:
        return primes[:2]
    
    # Reconstruct the sequence
    seq = [primes[best_k], primes[best_j]]
    j, k = best_j, best_k
    while len(seq) < best_len:
        i = prev[j, k]
        if i < 0:
            break
        seq.append(primes[i])
        k = j
        j = i
    
    seq.reverse()
    
    # Validate the result
    if len(seq) < 2:
        return primes[:2]
    
    # Check that all elements are prime and <= 10000
    for p in seq:
        if p > 10000 or not is_prime[p]:
            return primes[:2]
    
    # Check that the sequence is increasing
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i+1]:
            return primes[:2]
    
    # Check that gaps are nondecreasing
    gaps = [seq[i+1] - seq[i] for i in range(len(seq) - 1)]
    for i in range(len(gaps) - 1):
        if gaps[i] > gaps[i+1]:
            return primes[:2]
    
    return seq
# EVOLVE-BLOCK-END
