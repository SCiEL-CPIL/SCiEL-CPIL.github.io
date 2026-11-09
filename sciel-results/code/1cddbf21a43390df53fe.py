import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    N = 10000000
    
    # Compute sigma(n) for all n <= N using NumPy
    # sigma = np.ones(N + 1, dtype=np.int64)
    # for i in range(1, N + 1):
    #     sigma[i::i] += i
    
    # This loop in Python might be slow if not optimized, but NumPy vectorization helps.
    # However, 10^7 iterations of the outer loop in Python is slow (~10s-1s).
    # Let's try it.
    
    sigma = np.ones(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline - 10:
        # Fallback if sigma computation took too long (unlikely, but safe)
        return [11, 6]

    best_error_num = None
    best_error_den = None
    best_a = None
    best_b = None
    
    # Search for best pair
    # Iterate b from N down to 1
    # For each b, try small k
    # Error = |k| / (2b)
    
    # We can limit k search. 
    # If we find a solution with k=1 at b=B, error is 1/(2B).
    # We can stop searching for smaller b if 1/(2b) >= current_best_error.
    # i.e., b <= 1/(2*current_best_error)
    
    # Let's iterate b from N down to 1
    # To speed up, we can use a list of k values to try: [1, -1, 2, -2, ...]
    # But we should check k=1, -1 first for every b?
    # Yes, because for a fixed b, smaller |k| gives smaller error.
    
    # Optimization: 
    # If we have a current best error E_best = num_best / den_best.
    # For a given b, if 1/(2b) >= E_best, then even k=1 gives error >= E_best.
    # So we can stop checking this b and all subsequent smaller b's?
    # No, because for smaller b, the error for k=1 is larger.
    # So if 1/(2b) >= E_best, then for this b and all smaller b, the minimum possible error (with k=1) is >= E_best.
    # Thus, we can break the loop entirely!
    
    # Wait, is it possible that a smaller b with a larger k gives a smaller error?
    # Error = |k| / (2b).
    # If b decreases, denominator decreases, so error increases (for fixed k).
    # If k increases, numerator increases, so error increases.
    # So the global minimum must be found at large b and small k.
    # Therefore, if we find a solution with error E, we only need to consider pairs (a,b) such that |k|/(2b) < E.
    # Since b is decreasing in our loop, once 1/(2b) >= E, no further pairs can improve the score.
    
    # So:
    # 1. Initialize best_error to infinity.
    # 2. Loop b from N down to 1.
    # 3. If 1/(2b) >= best_error: break.
    # 4. For k in [1, -1, 2, -2, ...]:
    #    a = (3b + k) / 2. Check validity.
    #    If sigma[a] == sigma[b]:
    #       err = abs(k) / (2b)
    #       If err < best_error: update best.
    #       Break inner loop? No, we might find smaller k? No, we iterate k in increasing |k|.
    #       So first match for this b is best for this b.
    #       But we still need to check smaller b? No, because for smaller b, min error is larger.
    #       Wait. If we update best_error, the threshold 1/(2b) for the *next* b (which is smaller) will be even larger relative to the new best_error?
    #       Let's check.
    #       Current b. Next b' = b-1.
    #       Threshold for b': 1/(2(b-1)).
    #       If 1/(2b) >= best_error, we break.
    #       If we find a better error at b, best_error decreases.
    #       Then 1/(2(b-1)) is definitely >= new best_error?
    #       Not necessarily. 1/(2b) might have been < old best_error, so we didn't break.
    #       We found a match, updated best_error.
    #       Now for b-1, we check if 1/(2(b-1)) >= new best_error.
    #       If yes, break.
    
    # This logic holds.
    
    k_values = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10]
    # Extend k_values if needed, but small k is most likely to yield best result with large b.
    
    for b in range(N, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Pruning check
        if best_error_num is not None:
            # Check if 1/(2b) >= best_error_num / best_error_den
            # 1 * best_error_den >= best_error_num * 2b
            if best_error_den >= best_error_num * 2 * b:
                break
                
        sb = sigma[b]
        
        # Try k values
        for k in k_values:
            # a = (3b + k) / 2
            # 3b + k must be even
            if (3 * b + k) % 2 != 0:
                continue
            a = (3 * b + k) // 2
            if a < 1 or a > N:
                continue
            if sigma[a] == sb:
                # Found a match
                # Error is |k| / (2b)
                # Since k != 0 (we started from 1), error is non-zero.
                err_num = abs(k)
                err_den = 2 * b
                
                # Check if this is better
                if best_error_num is None or (err_num * best_error_den < best_error_num * err_den):
                    best_error_num = err_num
                    best_error_den = err_den
                    best_a = a
                    best_b = b
                    # We found a match for this b. Since we iterate k by |k|, this is the best for this b.
                    # Can we break the b loop?
                    # No, because a smaller b might have a much smaller k? No, k is bounded by our list.
                    # But wait, if we found k=1 at b, error is 1/(2b).
                    # For any b' < b, min error is 1/(2b') > 1/(2b).
                    # So we can break the b loop immediately!
                    # Is this true?
                    # Yes, because error = |k|/(2b). Min |k| is 1.
                    # So min error for any b' is 1/(2b').
                    # Since b' < b, 1/(2b') > 1/(2b).
                    # So no smaller b can yield a better score than the one we just found (assuming k=1).
                    # What if we found k=2 at b? Error 2/(2b) = 1/b.
                    # Then min error for b' is 1/(2b').
                    # If 1/(2b') < 1/b, i.e., b' > b/2, it could be better.
                    # So we cannot always break.
                    # But if we found k=1, we can break.
                    if abs(k) == 1:
                        break # Break b loop
                    else:
                        # We found a match with k > 1.
                        # We should continue checking smaller b, but the pruning check will handle it.
                        pass
                break # Break k loop for this b, as we found the smallest |k| for this b

    if best_a is not None:
        return [int(best_a), int(best_b)]
        
    return [11, 6]
