"""Rewritten program using DP on a DAG of prime subsequences with nondecreasing gaps."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Generate all primes <= 10000
    N = 10000
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    n = len(primes)
    
    # We want the longest subsequence of primes p_1 < p_2 < ... < p_k such that
    # gaps g_i = p_{i+1} - p_i are nondecreasing: g_1 <= g_2 <= ... <= g_{k-1}
    
    # DP approach:
    # Let dp[i][j] = length of longest valid subsequence ending with primes[i], primes[j]
    # where i < j, and the last gap is primes[j] - primes[i].
    # Actually, let's define it more carefully.
    # 
    # Let f(j, g) be the max length of a valid subsequence ending at prime index j with last gap g.
    # But g can be large. Instead, use:
    # dp[i][j] = max length of valid subsequence ending at primes[i], primes[j] (i < j)
    # The transition: dp[i][j] = 1 + max{ dp[h][i] : h < i and primes[i] - primes[h] <= primes[j] - primes[i] }
    # Base case: dp[i][j] = 2 for all i < j (subsequence of just primes[i], primes[j])
    
    # This is O(n^3) naively. We need to optimize.
    # 
    # Alternative: For each pair (i, j) with i < j, we want to find the best h < i such that
    # primes[i] - primes[h] <= primes[j] - primes[i], i.e., primes[h] >= primes[i] - (primes[j] - primes[i]) = 2*primes[i] - primes[j].
    # Since primes are sorted, we can binary search for the smallest h such that primes[h] >= 2*primes[i] - primes[j].
    # Then we need max dp[h][i] for h in [h_min, i-1].
    # 
    # We can precompute for each i, an array best[i][h] = max dp[h][i] for h <= some index.
    # Actually, for fixed i, as we iterate j > i, the threshold 2*primes[i] - primes[j] decreases as j increases.
    # So h_min decreases (or stays) as j increases. This means the set of valid h's grows.
    # 
    # Let's restructure: for each i (from 0 to n-1), we maintain a data structure that allows us to query
    # max dp[h][i] for h in [0, h_max] where h_max is determined by the gap constraint.
    # 
    # Actually, let's think differently. Process primes in order. For each prime j, we want to compute
    # the best subsequence ending at j. But the state needs to include the previous prime and the last gap.
    # 
    # Let's use: dp[j] = list/array where dp[j][gap_index]... no, gaps are not bounded nicely.
    # 
    # Let's stick with dp[i][j] but optimize with prefix maxima.
    # For each i, after computing all dp[h][i] for h < i, we can build a prefix max array:
    # prefix_max_i[k] = max{ dp[h][i] : h <= k }
    # Then for a given j > i, we find h_min = smallest h such that primes[h] >= 2*primes[i] - primes[j].
    # Then dp[i][j] = 1 + prefix_max_i[h_min - 1] if h_min > 0, else 2 (base case).
    # Wait, if h_min = 0, then all h from 0 to i-1 are valid, so dp[i][j] = 1 + max(prefix_max_i[i-1], 1) ... 
    # Actually base case is dp[i][j] >= 2. And if there's a valid h, dp[i][j] = 1 + max dp[h][i].
    # If no valid h, dp[i][j] = 2.
    
    # This is O(n^2 log n) due to binary search, which for n ~ 1229 (primes <= 10000) is about 1229^2 * 11 ≈ 16M operations. Should be fine.
    
    # Let's implement this.
    
    # dp[i][j] for i < j. We'll store it as a 2D array of shape (n, n), initialized to 0.
    # Only entries with i < j are meaningful.
    
    dp = np.zeros((n, n), dtype=np.int32)
    
    # For each i from 1 to n-1 (i is the second-to-last prime index)
    # We need prefix_max for column i: prefix_max_i[k] = max{ dp[h][i] : h <= k }
    # But dp[h][i] is only defined for h < i.
    
    # Let's iterate i from 1 to n-1. For each i, we first need to have computed all dp[h][i] for h < i.
    # Then we build prefix_max for column i.
    # Then for each j > i, we compute dp[i][j].
    
    # Actually, we should iterate in a way that when we compute dp[i][j], all dp[h][i] for h < i are already computed.
    # Since h < i < j, if we iterate i from 0 to n-1, and for each i, iterate j from i+1 to n-1,
    # then when we're at (i, j), all dp[h][i] for h < i have been computed in previous iterations of i.
    
    # But we need prefix_max for column i. Let's build it on the fly.
    
    # For each i, after all dp[h][i] for h < i are set, compute:
    # prefix_max_i = np.maximum.accumulate(dp[:i, i])
    # Then for j in range(i+1, n):
    #   threshold = 2 * primes[i] - primes[j]
    #   h_min = np.searchsorted(primes, threshold, side='left')
    #   if h_min < i:
    #       best_prev = prefix_max_i[h_min - 1] if h_min > 0 else 0
    #       # Wait, prefix_max_i[k] = max{ dp[h][i] : h <= k }
    #       # We want h in [h_min, i-1], so we want max{ dp[h][i] : h_min <= h <= i-1 }
    #       # That's not directly prefix_max. We need a range maximum query.
    #       # Alternatively, we can use: max{ dp[h][i] : h >= h_min and h < i }
    #       # Let's define suffix_max_i[k] = max{ dp[h][i] : h >= k and h < i }
    #       # Then dp[i][j] = 1 + suffix_max_i[h_min] if h_min < i, else 2
    
    # Let's use suffix_max.
    
    for i in range(1, n):
        # Build suffix_max for column i: suffix_max_i[k] = max{ dp[h][i] : h >= k and h < i }
        # For k from i-1 down to 0
        col = dp[:i, i]
        # suffix_max[k] = max(col[k:])
        # We can compute this by iterating from the end
        suffix_max = np.zeros(i, dtype=np.int32)
        if i > 0:
            suffix_max[i-1] = col[i-1]
            for k in range(i-2, -1, -1):
                suffix_max[k] = max(col[k], suffix_max[k+1])
        
        for j in range(i+1, n):
            threshold = 2 * primes[i] - primes[j]
            h_min = np.searchsorted(primes, threshold, side='left')
            if h_min < i:
                best_prev = suffix_max[h_min]
                if best_prev > 0:
                    dp[i][j] = best_prev + 1
                else:
                    dp[i][j] = 2
            else:
                dp[i][j] = 2
    
    # The answer is the maximum value in dp, or 2 if n < 2, or 1 if n < 1.
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]
    
    max_len = int(np.max(dp))
    if max_len < 2:
        max_len = 2
    
    # Now we need to reconstruct the actual subsequence.
    # Find the (i, j) that achieves max_len.
    max_idx = np.unravel_index(np.argmax(dp), dp.shape)
    i_star, j_star = int(max_idx[0]), int(max_idx[1])
    
    # Reconstruct backwards
    seq = [int(primes[j_star]), int(primes[i_star])]
    i, j = i_star, j_star
    while len(seq) < max_len:
        # Find h < i such that dp[h][i] + 1 == dp[i][j] and primes[i] - primes[h] <= primes[j] - primes[i]
        gap = primes[j] - primes[i]
        found = False
        for h in range(i-1, -1, -1):
            if dp[h][i] + 1 == dp[i][j] and (primes[i] - primes[h]) <= gap:
                seq.append(int(primes[h]))
                i, j = h, i
                found = True
                break
        if not found:
            # Should not happen
            break
    
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
