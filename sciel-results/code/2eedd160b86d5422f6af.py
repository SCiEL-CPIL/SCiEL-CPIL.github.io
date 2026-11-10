import numpy as np

def construct():
    N = 10_000_000
    
    # Compute sigma(n) for all n from 1 to N using sieve
    sigma = np.ones(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    # We search for pairs (a, b) such that |2a - 3b| = k
    # Error = k / (2b)
    # We want to minimize this.
    # Current best is initialized to infinity.
    best_err = float('inf')
    best_a = -1
    best_b = -1
    
    # Check k=0, 1, 2, ...
    # For each k, we check a = (3b + k) / 2 and a = (3b - k) / 2
    # Note: a must be integer, so 3b +/- k must be even.
    
    # We can vectorize for each k.
    # Max k to check: if k / (2 * 1) > best_err, no smaller b will help, 
    # but larger b might. 
    # Actually, min error for a given k is k / (2 * N).
    # If k / (2 * N) >= best_err, we can stop.
    
    max_k = int(2 * N * best_err) if best_err < float('inf') else 2 * N
    # But best_err is inf initially. 
    # Let's just iterate k up to a reasonable limit, e.g., 1000.
    # Or dynamically update max_k.
    
    # First, find the best for k=1 and k=0 (current logic essentially)
    # But we can just loop k from 1 to some limit.
    
    # Let's iterate k from 1 to 1000. 
    # For each k, we can check all b in parallel.
    
    for k in range(1, 1001):
        # Check a = (3b + k) / 2
        # 3b + k must be even.
        # b parity depends on k.
        # If k is even, b can be any.
        # If k is odd, b must be odd.
        
        # Range of b:
        # a <= N => 3b + k <= 2N => b <= (2N - k) // 3
        # a >= 1 => 3b + k >= 2 => b >= (2 - k) // 3 (which is negative for k>=1, so b>=1)
        
        max_b = (2 * N - k) // 3
        if max_b < 1:
            continue
            
        # Generate b values
        if k % 2 == 0:
            b_vals = np.arange(1, max_b + 1, dtype=np.int64)
        else:
            b_vals = np.arange(1, max_b + 1, 2, dtype=np.int64) # Odd b
            
        if len(b_vals) == 0:
            continue
            
        a_vals = (3 * b_vals + k) // 2
        
        # Check bounds for a
        mask = (a_vals >= 1) & (a_vals <= N)
        if not np.any(mask):
            continue
            
        b_valid = b_vals[mask]
        a_valid = a_vals[mask]
        
        # Check sigma equality
        valid = sigma[a_valid] == sigma[b_valid]
        
        if np.any(valid):
            # Find b that minimizes error k / (2b) => maximize b
            b_max = np.max(b_valid[valid])
            # There might be multiple a for the same b? No, a is determined by b and k.
            # So just take the max b.
            # Wait, we need the specific a corresponding to that b.
            # Since a is linear in b, we can find the index.
            idx_max = np.where(b_valid == b_max)[0][0]
            a_candidate = int(a_valid[idx_max])
            b_candidate = int(b_max)
            
            err = k / (2.0 * b_candidate)
            if err < best_err:
                best_err = err
                best_a = a_candidate
                best_b = b_candidate
                # Update max_k limit?
                # If best_err is very small, we might not need to check large k.
                # But k loop is small (1000).
                
        # Check a = (3b - k) / 2
        # 3b - k must be even.
        # a >= 1 => 3b - k >= 2 => b >= (k + 2) // 3
        # a <= N => 3b - k <= 2N => b <= (2N + k) // 3
        
        min_b = (k + 2) // 3
        max_b2 = (2 * N + k) // 3
        
        if min_b > max_b2:
            continue
            
        if k % 2 == 0:
            b_vals2 = np.arange(min_b, max_b2 + 1, dtype=np.int64)
        else:
            # Adjust start to be odd
            if min_b % 2 == 0:
                min_b += 1
            b_vals2 = np.arange(min_b, max_b2 + 1, 2, dtype=np.int64)
            
        if len(b_vals2) == 0:
            continue
            
        a_vals2 = (3 * b_vals2 - k) // 2
        
        # Check bounds for a (already handled by min_b, but a<=N is max_b2)
        # a >= 1 is handled by min_b.
        # Just in case, re-verify.
        mask2 = (a_vals2 >= 1) & (a_vals2 <= N)
        if not np.any(mask2):
            continue
            
        b_valid2 = b_vals2[mask2]
        a_valid2 = a_vals2[mask2]
        
        valid2 = sigma[a_valid2] == sigma[b_valid2]
        
        if np.any(valid2):
            b_max2 = np.max(b_valid2[valid2])
            idx_max2 = np.where(b_valid2 == b_max2)[0][0]
            a_candidate = int(a_valid2[idx_max2])
            b_candidate = int(b_max2)
            
            err = k / (2.0 * b_candidate)
            if err < best_err:
                best_err = err
                best_a = a_candidate
                best_b = b_candidate
                
    if best_a == -1:
        return [11, 6]
        
    # Check if error is 0 (should not happen for k>=1)
    if best_err == 0.0:
        return [11, 6]
        
    return [best_a, best_b]
