import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit

    # 1. Generate primes <= 10000
    limit = 10000
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.array([i for i, flag in enumerate(is_prime) if flag])
    N = len(primes)
    
    if N < 2:
        return primes.tolist()

    # 2. Longest path in DAG where edge (i, j) exists if gap[i-1] <= gap[j-1] 
    #    Wait, the condition is on the gaps between consecutive elements in the subsequence.
    #    Let the subsequence be p_{i_1}, p_{i_2}, ..., p_{i_k}.
    #    Gaps are g_m = p_{i_{m+1}} - p_{i_m}.
    #    Condition: g_1 <= g_2 <= ... <= g_{k-1}.
    #    
    #    This is not a simple DAG on indices because the "state" includes the last gap.
    #    However, we can use DP where dp[j][last_gap_idx] is too expensive.
    #    
    #    Alternative perspective:
    #    Let dp[j] be the maximum length of a valid subsequence ending at primes[j].
    #    But we also need to know the last gap to ensure the next gap is >= last gap.
    #    So the state is (last_prime_index, last_gap).
    #    Since last_gap = primes[j] - primes[i] for some i < j, the state is effectively (i, j).
    #    Let L[i][j] = max length of a valid subsequence ending with ... primes[i], primes[j].
    #    Then L[i][j] = 1 + max{ L[h][i] : h < i and (primes[i]-primes[h]) <= (primes[j]-primes[i]) }
    #    Base case: L[i][j] = 2 for all i < j.
    #    
    #    We want max L[i][j] over all i < j.
    #    
    #    N is ~1229. N^2 is ~1.5M.
    #    The transition for L[i][j] requires finding max L[h][i] for h in a range.
    #    For fixed i, as j increases, the required gap (primes[j]-primes[i]) increases.
    #    So the set of valid h's (where primes[i]-primes[h] <= primes[j]-primes[i]) grows.
    #    Specifically, we need primes[h] >= 2*primes[i] - primes[j].
    #    Let limit_val = 2*primes[i] - primes[j].
    #    We need h such that primes[h] >= limit_val.
    #    Since primes are sorted, this corresponds to h >= lower_bound(limit_val).
    #    Also h < i.
    #    So we need max L[h][i] for h in [lower_bound(limit_val), i-1].
    #    
    #    We can precompute for each i an array of L[h][i] for h < i.
    #    Then for each j > i, we need the max over a suffix of this array (from some index to i-1).
    #    We can precompute suffix maximums for each i.
    #    
    #    Algorithm:
    #    1. Initialize L as a 2D array of size N x N with 2s (or 0s, but min length 2 for pairs).
    #       Actually, L[i][j] is only defined for i < j.
    #    2. For each i from 0 to N-1:
    #         Build suffix_max array for column i: 
    #         suffix_max_i[k] = max(L[h][i] for h in [k, i-1])
    #         But we also need to track the argmax to reconstruct the path.
    #         So store suffix_idx_i[k] = argmax.
    #    3. For each j > i:
    #         limit_val = 2*primes[i] - primes[j]
    #         Find smallest h such that primes[h] >= limit_val. Let this be h_min.
    #         If h_min < i:
    #             best_h = suffix_idx_i[h_min]
    #             L[i][j] = L[best_h][i] + 1
    #             prev[i][j] = best_h
    #         Else:
    #             L[i][j] = 2 (base case, just i, j)
    #             prev[i][j] = -1
    #    
    #    Wait, the suffix_max should be computed BEFORE we fill in L[i][j]?
    #    No, L[h][i] depends on pairs (h, i) where h < i.
    #    So when processing i, all L[h][i] for h < i are already computed.
    #    So we can build the suffix max for column i at the start of iteration i.
    #    Then we can fill in L[i][j] for all j > i.
    #    
    #    Complexity: O(N^2) time, O(N^2) space.
    #    N=1229, N^2 = 1.5M. This is very fast.

    # Use lists for flexibility, or NumPy for speed?
    # NumPy might be tricky with variable length suffix max and argmax.
    # Let's stick to Python lists for the DP table, but use bisect for the lower bound.
    # Or use NumPy for the primes array and bisect on it.
    
    import bisect
    
    # L[i][j] = max length ending at i, j (i < j)
    # We only need L[i][j] for i < j.
    # To save memory, we could store only the upper triangle, but N^2 is fine.
    # L is a list of lists.
    L = [[0] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]
    
    # Initialize base cases: L[i][j] = 2 for all i < j
    # We don't need to explicitly set all, but let's be careful.
    # Actually, the recurrence handles it. If no valid h, L[i][j] = 2.
    
    best_len = 2
    best_i, best_j = 0, 1
    
    # primes is a numpy array, bisect works on it? 
    # bisect.bisect_left works on lists. Let's convert to list for bisect, or use np.searchsorted.
    # np.searchsorted is faster.
    
    for i in range(N - 1):
        # Build suffix max for column i (indices h < i)
        # We need max L[h][i] for h in [k, i-1]
        # Let's create arrays of length i
        # suffix_val[k] = max(L[h][i] for h in [k, i-1])
        # suffix_idx[k] = argmax
        
        if i > 0:
            suffix_val = [0] * i
            suffix_idx = [0] * i
            
            suffix_val[i-1] = L[i-1][i]
            suffix_idx[i-1] = i-1
            
            for h in range(i-2, -1, -1):
                if L[h][i] >= suffix_val[h+1]:
                    suffix_val[h] = L[h][i]
                    suffix_idx[h] = h
                else:
                    suffix_val[h] = suffix_val[h+1]
                    suffix_idx[h] = suffix_idx[h+1]
        else:
            suffix_val = []
            suffix_idx = []
            
        for j in range(i + 1, N):
            # Check time
            if time.time() > deadline:
                break
                
            limit_val = 2 * primes[i] - primes[j]
            
            # Find smallest h such that primes[h] >= limit_val
            # h must be < i
            h_min = np.searchsorted(primes, limit_val, side='left')
            
            if h_min < i:
                # We need max L[h][i] for h in [h_min, i-1]
                best_h = suffix_idx[h_min]
                new_len = L[best_h][i] + 1
                L[i][j] = new_len
                prev[i][j] = best_h
                
                if new_len > best_len:
                    best_len = new_len
                    best_i = i
                    best_j = j
            else:
                # No valid h, so the subsequence is just [primes[i], primes[j]]
                L[i][j] = 2
                prev[i][j] = -1
                
            if time.time() > deadline:
                break
                
    # Backtrack
    res = []
    i, j = best_i, best_j
    while j != -1:
        res.append(primes[j])
        i, j = j, prev[i][j]
        
    res.reverse()
    return res.tolist()
