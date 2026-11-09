import time
import numpy as np

def construct():
    """
    Finds the longest subsequence of primes <= 10000 with non-decreasing gaps.
    Uses Dynamic Programming with vectorized NumPy operations for the inner loop.
    """
    deadline = time.time() + 85.0
    LIMIT = 10000
    
    # 1. Generate primes using Sieve of Eratosthenes
    sieve = np.ones(LIMIT + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.array([i for i, is_p in enumerate(sieve) if is_p], dtype=np.int32)
    N = len(primes)
    
    # Maximum possible gap between two primes <= 10000 is 10000 - 2 = 9998
    # But for simplicity and safety, we can use LIMIT as the size of gap dimension.
    # Actually, max gap is smaller. Let's compute max_gap to save memory?
    # Max gap between consecutive primes is small, but we are picking arbitrary subsequences.
    # The gap in our DP state is the difference between consecutive elements in the subsequence.
    # The maximum possible gap is 10000 - 2 = 9998.
    # However, the number of distinct gaps is not too large.
    # But the DP state dp[i][g] depends on g up to LIMIT.
    # N is ~1229. LIMIT is 10000.
    # dp size: 1229 * 10001 * 4 bytes (int32) ~ 49 MB. This is acceptable.
    
    G = LIMIT + 1
    
    # dp[i][g]: max length of valid sequence ending at primes[i] with last gap exactly g.
    # We use g=0 for single element sequences (no gap).
    dp = np.zeros((N, G), dtype=np.int32)
    dp[:, 0] = 1
    
    # pred[i][g]: index j of previous prime in optimal sequence ending at i with gap g.
    pred = np.full((N, G), -1, dtype=np.int32)
    
    # pref[i][g]: max(dp[i][0], ..., dp[i][g])
    # This is the max length of a valid sequence ending at primes[i] with last gap <= g.
    pref = np.zeros((N, G), dtype=np.int32)
    
    best_len = 1
    best_i = 0
    best_g = 0
    
    # 2. Fill DP table
    for i in range(1, N):
        if time.time() > deadline:
            break
            
        pi = primes[i]
        
        # For each previous prime j < i, the gap is gap_ij = pi - primes[j].
        # We want to update dp[i][gap_ij] = max(dp[i][gap_ij], pref[j][gap_ij] + 1).
        # Note: pref[j][gap_ij] is the best length ending at j with gap <= gap_ij.
        
        # Vectorized approach:
        # gaps = pi - primes[:i]  (array of shape (i,))
        # We need to update dp[i, gaps] with values from pref[:i, gaps] + 1.
        
        gaps = pi - primes[:i]  # shape (i,)
        
        # We need to access pref[j, gaps[j]] for each j.
        # This is a gather operation.
        # indices_j = np.arange(i)
        # indices_g = gaps
        # vals = pref[indices_j, indices_g] + 1
        
        indices_j = np.arange(i, dtype=np.int32)
        vals = pref[indices_j, gaps] + 1
        
        # Now we have a list of proposed values for dp[i, gaps].
        # We want to update dp[i, gaps] = max(dp[i, gaps], vals).
        # But there might be duplicate gaps?
        # primes are distinct, so gaps are distinct for fixed i?
        # No, pi - p_j are distinct because p_j are distinct.
        # So gaps are unique.
        
        # However, we also need to update pred.
        # If vals[k] > dp[i, gaps[k]], then pred[i, gaps[k]] = indices_j[k].
        
        current_dp_row = dp[i, gaps]
        mask = vals > current_dp_row
        if np.any(mask):
            dp[i, gaps[mask]] = vals[mask]
            pred[i, gaps[mask]] = indices_j[mask]
        
        # Compute prefix max for row i
        # pref[i] = np.maximum.accumulate(dp[i])
        # This is O(G) per i, total O(N*G) ~ 12M operations. Fast.
        pref[i] = np.maximum.accumulate(dp[i])
        
        # Update global best
        max_len_i = np.max(dp[i])
        if max_len_i > best_len:
            best_len = max_len_i
            best_g = np.argmax(dp[i])
            best_i = i
            
    # 3. Reconstruct sequence
    seq = []
    if best_len > 0:
        i = best_i
        g = best_g
        while i >= 0:
            seq.append(int(primes[i]))
            if i == 0:
                break
            j = pred[i][g]
            if j == -1:
                break
            # Find the specific gap h used in predecessor state j
            # such that dp[j][h] == pref[j][g] and h <= g.
            # We know that the transition used gap g from j to i.
            # The state at j must have had a gap h such that h <= g.
            # And dp[j][h] contributed to pref[j][g].
            # Actually, pref[j][g] = max_{h<=g} dp[j][h].
            # We need to find h such that dp[j][h] == pref[j][g].
            # There might be multiple such h. Any is valid.
            # Let's find the largest h <= g such that dp[j][h] == pref[j][g].
            # Or the smallest?
            # It doesn't matter for validity, as long as the chain is valid.
            # Let's find any h.
            
            target_val = pref[j][g]
            # Search for h in [0, g] such that dp[j, h] == target_val
            # We can iterate backwards from g to 0 to find the first match.
            # This is O(g) in worst case, but g is usually small.
            # Or we can use np.where on dp[j, :g+1].
            
            # Efficient way:
            # h_candidates = np.where(dp[j, :g+1] == target_val)[0]
            # if len(h_candidates) > 0:
            #     h = h_candidates[-1] # Pick largest
            # else:
            #     h = 0 # Fallback
            
            # Since g can be up to 10000, np.where is O(g).
            # In the worst case, reconstruction is O(L^2) where L is length.
            # L is small (maybe < 20), so this is fine.
            
            h_candidates = np.where(dp[j, :g+1] == target_val)[0]
            if len(h_candidates) > 0:
                h = h_candidates[-1]
            else:
                h = 0
                
            i = j
            g = h
        seq.reverse()
        
    return seq
