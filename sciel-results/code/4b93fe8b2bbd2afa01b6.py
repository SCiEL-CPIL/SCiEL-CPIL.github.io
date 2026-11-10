import numpy as np

def construct():
    N = 10_000_000
    
    # Compute sigma(n) for all n from 1 to N using sieve
    # Corrected initialization
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    # Initial best error based on current program performance
    # Score ~ 4e6 -> Error ~ 2.5e-7
    # We want to find errors smaller than this.
    # If we find none, we return the best found (which might be worse than current? 
    # No, we should track the best found in this run.
    # If this run finds nothing better, we should probably return the current best known 
    # or re-run the old logic? 
    # The directive says "design a new program... from scratch".
    # If the new program fails to find a pair, it's invalid.
    # We must ensure we find at least one valid pair.
    # The old program found one. 
    # We can seed E_best with a large value, but limit K_max to avoid O(N^2).
    # Let's set E_best to a value that ensures we check at least some pairs.
    # If we set E_best = 1.0, K_max = 2b. That's too many.
    # Let's iterate b from N down to 1.
    # For each b, we check k up to some limit.
    # To guarantee finding *something*, we can check k=1, 2, 3 for all b?
    # That's 3 * 10^7 checks. Feasible.
    # But if the best solution has k=100, we miss it.
    # However, error for k=100, b=10^7 is 100/2e7 = 5e-6.
    # Error for k=1, b=10^6 is 1/2e6 = 5e-7.
    # So small k is generally better if b is large.
    # The current best is ~2.5e-7.
    # So we need k/b < 5e-7.
    # If we restrict k to <= 10, we cover b > 2e6.
    # If the best solution has k=100, it must have b > 2e8, which is impossible.
    # So the best solution MUST have small k.
    # Specifically, k < 5e-7 * b.
    # For b=10^7, k < 5.
    # For b=10^6, k < 0.5.
    # So checking k up to 10 for all b is safe and fast.
    
    E_best = float('inf')
    best_a = 1
    best_b = 1
    
    # We will check k from 1 to K_LIMIT
    # K_LIMIT should be large enough to catch the best solution.
    # Based on analysis, K_LIMIT = 20 is safe.
    K_LIMIT = 20
    
    # Iterate b from N down to 1
    # Optimization: If 1/(2*b) > E_best, then even k=1 is too large.
    # So we can break when b < 1/(2*E_best).
    
    for b in range(N, 0, -1):
        # Check if k=1 can improve
        if E_best < float('inf'):
            min_err_for_b = 1.0 / (2.0 * b)
            if min_err_for_b >= E_best:
                break
                
        # Determine parity of b
        # 3*b + k must be even.
        # If b is even, 3b is even, so k must be even.
        # If b is odd, 3b is odd, so k must be odd.
        
        # Start k based on parity
        if b % 2 == 0:
            k_start = 2
        else:
            k_start = 1
            
        # Iterate k
        for k in range(k_start, K_LIMIT + 1, 2):
            a = (3 * b + k) // 2
            if a > N or a < 1:
                continue
            if sigma[a] == sigma[b]:
                err = k / (2.0 * b)
                if err < E_best:
                    E_best = err
                    best_a = a
                    best_b = b
                    # Since we are iterating k increasing, this is the best for this b.
                    # We can break the k loop.
                    break
                    
    # If no pair found (should not happen), fallback
    if E_best == float('inf'):
        # Fallback to a known valid pair, e.g. from old logic or small search
        # Let's do a small brute force for fallback
        for b in range(1, 1000):
            for k in range(1, 10, 2 if b%2 else 2):
                a = (3*b+k)//2
                if 1 <= a <= N:
                    if sigma[a] == sigma[b]:
                        E_best = k / (2.0 * b)
                        best_a = a
                        best_b = b
                        break
            if E_best != float('inf'):
                break
                
    return [int(best_a), int(best_b)]
