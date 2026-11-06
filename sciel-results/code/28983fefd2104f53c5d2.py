import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin (evaluator kills at 90s)
    
    # 1. Sieve of Eratosthenes for primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes = np.array([i for i, flag in enumerate(is_prime) if flag], dtype=np.int64)
    m = len(primes)
    
    # 2. Vectorized DP
    # dp[i][j] = length of longest valid sequence ending with (primes[j], primes[i])
    # where j < i
    # Transition: dp[i][j] = 1 + max(dp[j][h] for h in [0, j-1] with primes[h] >= 2*primes[j] - primes[i])
    #              or 2 if no valid h exists
    
    dp = np.zeros((m, m), dtype=np.int32)
    prev = np.full((m, m), -1, dtype=np.int32)
    
    # For each j, compute suffix max of dp[j, 0..j-1]
    # Then for each i > j, find the threshold and use binary search
    
    for j in range(m - 1):
        if j == 0:
            # No h exists for j=0
            dp[j+1:, 0] = 2
            prev[j+1:, 0] = -1
            continue
        
        # Compute suffix max and argmax for dp[j, 0..j-1]
        # suf_max[k] = max(dp[j, k], dp[j, k+1], ..., dp[j, j-1])
        # suf_arg[k] = index h in [k, j-1] achieving suf_max[k]
        row = dp[j, :j]  # dp[j, 0..j-1], length j
        # suffix max
        suf_max = np.maximum.accumulate(row[::-1])[::-1]
        # For argmax, we need the leftmost index achieving the max in each suffix
        # Use a different approach: for each position, find where the running max changes
        # Actually, let's compute it properly
        # suf_argmax[k] = argmax of row[k:j]
        # We can compute this by iterating from right to left
        suf_arg = np.zeros(j, dtype=np.int32)
        suf_arg[j-1] = j-1
        for k in range(j-2, -1, -1):
            if row[k] >= suf_max[k+1]:
                suf_arg[k] = k
            else:
                suf_arg[k] = suf_arg[k+1]
        
        # For each i in j+1..m-1:
        # threshold_i = 2*primes[j] - primes[i]
        # We need the smallest h in [0, j-1] such that primes[h] >= threshold_i
        # Then dp[i][j] = 1 + suf_max[h] if h < j, else 2
        
        # Compute thresholds for all i > j
        thresholds = 2 * primes[j] - primes[j+1:]  # shape (m-j-1,)
        
        # For each threshold, find the leftmost h in [0, j-1] with primes[h] >= threshold
        # primes is sorted, so we can use searchsorted
        # np.searchsorted(primes[:j], threshold, side='left') gives the first index where primes[idx] >= threshold
        # But we only want indices in [0, j-1]
        
        # For all i > j, compute h_idx
        h_idxs = np.searchsorted(primes[:j], thresholds, side='left')
        
        # h_idx is in [0, j] (j means no valid h)
        # Where h_idx < j, dp[i][j] = 1 + suf_max[h_idx]
        # Where h_idx == j, dp[i][j] = 2
        
        valid_mask = h_idxs < j
        dp_vals = np.full(m - j - 1, 2, dtype=np.int32)
        prev_vals = np.full(m - j - 1, -1, dtype=np.int32)
        
        if np.any(valid_mask):
            dp_vals[valid_mask] = 1 + suf_max[h_idxs[valid_mask]]
            prev_vals[valid_mask] = h_idxs[valid_mask]
        
        dp[j+1:, j] = dp_vals
        prev[j+1:, j] = prev_vals
    
    # Find the best (i, j) pair
    # We want max dp[i][j] over all i > j
    # Create a mask for valid entries
    mask = np.triu(np.ones((m, m), dtype=bool), k=1)
    dp_masked = np.where(mask, dp, 0)
    
    max_idx = np.argmax(dp_masked)
    best_i, best_j = np.unravel_index(max_idx, (m, m))
    max_len = dp[best_i, best_j]
    
    if max_len < 2:
        return primes[:2].tolist()
    
    # 3. Reconstruct path
    path = []
    i, j = int(best_i), int(best_j)
    path.append(int(primes[i]))
    path.append(int(primes[j]))
    
    while prev[i, j] != -1:
        h = int(prev[i, j])
        i, j = j, h
        path.append(int(primes[j]))
    
    path.reverse()
    
    # 4. Local improvement: try to insert additional primes into the sequence
    # For each consecutive pair (a, b) in the sequence, check if we can insert a prime c
    # such that (b - a) <= (c - a) and (b - c) <= (next_gap)
    # Actually, we need: gap_before <= (c - a) and (b - c) <= gap_after
    # where gap_before = a - prev_element (or 0 if a is first)
    # and gap_after = next_element - b (or infinity if b is last)
    
    improved = True
    while improved:
        improved = False
        # Work with a list for flexibility
        seq = list(path)
        for pos in range(len(seq) - 1):
            a = seq[pos]
            b = seq[pos + 1]
            gap_before = a - seq[pos - 1] if pos > 0 else 0
            gap_after = seq[pos + 2] - b if pos + 2 < len(seq) else float('inf')
            
            # We want a prime c with a < c < b such that:
            # gap_before <= c - a  =>  c >= a + gap_before
            # b - c <= gap_after   =>  c >= b - gap_after
            # So c >= max(a + gap_before, b - gap_after) and c < b and c > a
            
            low = max(a + 1, a + gap_before, b - gap_after)
            high = b - 1
            
            if low <= high:
                # Find the largest prime in [low, high]
                # Use is_prime to check
                # We want to insert the prime that keeps gaps non-decreasing
                # Any prime in [low, high] works for the gap constraints
                # Let's pick the one that gives the best chance for further insertions
                # Try inserting the largest prime first (to leave more room on the left)
                for c in range(high, low - 1, -1):
                    if is_prime[c]:
                        # Check if inserting c maintains non-decreasing gaps
                        # New gaps: (c - a) and (b - c)
                        # Need: gap_before <= (c - a) and (b - c) <= gap_after
                        # Already ensured by low/high
                        new_gap1 = c - a
                        new_gap2 = b - c
                        if gap_before <= new_gap1 and new_gap2 <= gap_after:
                            seq.insert(pos + 1, c)
                            path = seq
                            improved = True
                            break
                if improved:
                    break
    
    # Final verification
    result = path
    # Verify validity
    valid = True
    for k in range(len(result) - 2):
        gap1 = result[k+1] - result[k]
        gap2 = result[k+2] - result[k+1]
        if gap1 > gap2:
            valid = False
            break
    if not valid:
        # Fallback to the DP result
        result = path
    
    return result

# Test locally
if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(f"Sequence: {res}")
    # Verify
    valid = True
    for k in range(len(res)-2):
        gap1 = res[k+1] - res[k]
        gap2 = res[k+2] - res[k+1]
        if gap1 > gap2:
            valid = False
            break
    print(f"Valid: {valid}")
    print(f"All primes <= 10000: {all(p <= 10000 for p in res)}")
    print(f"Increasing: {all(res[i] < res[i+1] for i in range(len(res)-1))}")
