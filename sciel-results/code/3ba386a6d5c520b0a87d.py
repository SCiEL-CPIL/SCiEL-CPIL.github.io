import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n <= N using a sieve
    # This is the most expensive part.
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # We can break if time is running out, but we need sigma for the search.
    # If we run out of time during sieve, we can't search effectively.
    # However, 10^7 is usually manageable in < 10s with NumPy.
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 1_000_000 == 0:
            if time.time() > deadline - 10:
                # Fallback to a known pair if sieve takes too long
                # 11, 6 is from seed: sigma(11)=12, sigma(6)=12.
                # |11/6 - 1.5| = |1.833 - 1.5| = 0.333
                # This is a valid pair, though not optimal.
                return [11, 6]
                
    if time.time() > deadline - 5:
        # Not enough time for search, return known valid pair
        return [11, 6]
        
    # 2. Direct Arithmetic Search
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5) # ~0.333
    
    # We iterate k = |2a - 3b|
    # We want small k and large b.
    # Error = k / (2b)
    
    # We'll try k from 1 up to a limit.
    # If we find a solution for k=1, error is ~ 1/(2*10^7) ~ 5e-8.
    # This is extremely small.
    
    max_k_to_try = 1000 # We likely find a solution for k=1 or 2 quickly.
    
    for k in range(1, max_k_to_try + 1):
        if time.time() > deadline - 2:
            break
            
        # Pruning: if minimum possible error for this k is worse than best, break
        # Min error for k is k / (2 * N)
        if k / (2.0 * N) >= best_err:
            break
            
        # a = (3b + k) / 2
        # a <= N => 3b + k <= 2N => b <= (2N - k) / 3
        # b >= 1
        # Parity: 3b + k must be even.
        # If k is even, b is even.
        # If k is odd, b is odd.
        
        max_b = (2 * N - k) // 3
        if max_b < 1:
            continue
            
        # Adjust start_b to correct parity
        # We want 3*start_b + k to be even.
        start_b = max_b
        if (3 * start_b + k) % 2 != 0:
            start_b -= 1
            
        # Iterate b downwards with step 2 to maintain parity
        # We only need to check a few large b's for each k?
        # Actually, if we find ANY valid pair for a given k, the error is k/(2b).
        # Larger b means smaller error. So we should start from largest b.
        # If we find one, it's the best for this k.
        # Should we continue to next k?
        # Yes, a smaller k with smaller b might be worse than larger k with much larger b?
        # No, error = k / 2b.
        # If we find a solution for k=1 with b ~ N, err ~ 1/N.
        # If we find a solution for k=2 with b ~ N, err ~ 2/N.
        # So k=1 is better.
        # Generally, smaller k is better if b is comparable.
        # Since b is bounded by N, and we pick max b, smaller k usually wins.
        # However, it's possible k=1 has no solution with large b, but k=2 does.
        # So we should check all b's for k=1 first?
        # Iterating all b's for k=1 could be slow if no solution exists near max b.
        # But we can limit the depth of search for each k if time is tight.
        
        # Let's iterate b downwards.
        # How deep?
        # If we don't find a solution in the first 10000 values of b for a given k,
        # it might be rare. But sigma collisions are not that rare.
        # Let's just iterate until time is up or we find one.
        
        found = False
        for b_val in range(start_b, 0, -2):
            if time.time() > deadline - 1:
                break
                
            a_val = (3 * b_val + k) // 2
            if a_val > N or a_val < 1:
                continue
                
            if sigma[a_val] == sigma[b_val]:
                err = k / (2.0 * b_val)
                if err < best_err:
                    best_err = err
                    best_a = a_val
                    best_b = b_val
                    # Since we are iterating k ascending, and b descending,
                    # this is a very good candidate.
                    # We can break out of b loop.
                    # Should we break out of k loop?
                    # If k=1, err ~ 1/N.
                    # If k=2, min err ~ 2/N.
                    # So if we found k=1, any k>=2 will have err >= 2/N > 1/N.
                    # So we can break k loop if we found a solution for k=1.
                    # More generally, if best_err <= (k+1)/(2N), we can stop?
                    # No, best_err might be larger if b is small.
                    # But we are picking max b.
                    # So for a given k, we found the best possible error for that k.
                    # If k=1, best_err ~ 1/(2*max_b).
                    # Next k=2, min possible error is 2/(2*N) = 1/N.
                    # If 1/(2*max_b) < 1/N, then k=1 solution is better.
                    # Since max_b ~ 2N/3, 1/(2*max_b) ~ 3/(4N) ~ 0.75/N.
                    # 0.75/N < 1/N. So k=1 solution is better.
                    # So if we find a solution for k=1, we can stop entirely?
                    # Wait, what if b for k=1 is small?
                    # We iterate b from max downwards. We found the FIRST one.
                    # So b is large.
                    # So yes, if we find a solution for k=1, it's likely the global minimum.
                    # We can break the k loop.
                    found = True
                    break
            # We can also add a check: if we've scanned too many b's without success, 
            # maybe this k has no large-b solution. But sigma collisions are frequent.
            
        if found and k == 1:
            # If we found a solution for k=1, it's extremely unlikely that k>1 will be better
            # given the max b constraint.
            break
            
    return [int(best_a), int(best_b)]
