import time
from math import isqrt

def construct():
    start_time = time.time()
    timeout = 110  # seconds, leave margin for 120s limit
    
    # Strategy:
    # 1. Generate powerful numbers up to M.
    # 2. Store them in a set for O(1) lookup.
    # 3. Test a set of candidate N values to find the one with the most pairs.
    #    We hypothesize that N values in the range [M, 2M] are good candidates.
    #    Specifically, we test N = M * r for various rational r, or just specific values.
    #    Given the density rho(x) ~ x^(-1/3), the convolution sum might peak somewhere in the middle.
    
    M = 10**9  # Limit for powerful numbers
    
    # Generate powerful numbers up to M
    powerful_set = set()
    # v^3 <= M => v <= M^(1/3)
    max_v = int(M ** (1/3)) + 1
    
    for v in range(1, max_v + 1):
        v_cubed = v ** 3
        if v_cubed > M:
            break
        max_u = isqrt(M // v_cubed)
        for u in range(1, max_u + 1):
            p = u * u * v_cubed
            powerful_set.add(p)
            
    # Check time
    if time.time() - start_time > timeout * 0.5:
        # If generation took too long, reduce M? 
        # But M=1e9 generation is fast (~1s).
        pass
        
    # Candidate N values to test
    # We want to find N with max pairs a+b=N, a,b in powerful_set, a<=b.
    # Let's test N values that are multiples of some base, or specific fractions of M.
    # The range of sums is [2, 2M].
    # Let's test N = M, 1.5M, 2M, and some nearby values.
    # Also, maybe smaller N? 
    # Density is higher for small x, but the number of available partners is smaller.
    # Let's sample N from [0.1M, 2M].
    
    candidates = []
    # Add specific "round" numbers
    candidates.append(M)
    candidates.append(int(1.5 * M))
    candidates.append(2 * M)
    
    # Add some sampled values
    # We can't test all, so we test a logarithmic or linear spread.
    # Let's test 20 values spread between 0.5M and 2M.
    for i in range(20):
        frac = 0.5 + i * (1.5 / 19.0) # 0.5 to 2.0
        candidates.append(int(frac * M))
        
    # Also test some smaller values just in case
    candidates.append(int(0.1 * M))
    candidates.append(int(0.5 * M))
    
    # Remove duplicates and sort
    candidates = list(set(candidates))
    
    best_N = 0
    best_pairs = []
    best_count = 0
    
    powerful_list = sorted(powerful_set) # Sorted list for a<=b constraint? 
    # Actually, for counting, we just need the set. 
    # For constructing pairs, we need to iterate.
    # To count efficiently: 
    # For each candidate N:
    #   count = 0
    #   for a in powerful_list:
    #       if a > N // 2: break
    #       if (N - a) in powerful_set:
    #           count += 1
    # This is O(K log K) for sorting + O(Candidates * K) for counting.
    # K ~ 650,000. Candidates ~ 25. 
    # 25 * 650,000 = 16,250,000 iterations. 
    # This might take ~5-10 seconds in Python. Acceptable.
    
    # Optimization: 
    # We can break early if a > N//2.
    # Also, we can pre-compute the list.
    
    for N in candidates:
        if time.time() - start_time > timeout - 10:
            break
            
        count = 0
        # Iterate through powerful numbers a such that a <= N/2
        # We need a <= N-a => 2a <= N => a <= N//2
        limit_a = N // 2
        
        # Binary search for the end index in powerful_list where value <= limit_a
        # Or just loop and break. 
        # Since powerful_list is sorted, we can use bisect.
        import bisect
        end_idx = bisect.bisect_right(powerful_list, limit_a)
        
        # Count pairs
        # To speed up, we can just check membership.
        # We don't need to store the pairs yet, just count.
        # If this count is a new max, we can store the N and re-construct pairs later.
        # Or we can store the pairs directly if we expect few? 
        # No, we don't know if it's max until we finish counting.
        # So we count first.
        
        # Optimization: 
        # If count is already > best_count, we can keep going.
        # But we can't stop early unless we know we can't beat best_count? 
        # No, we must finish the count to know the total.
        
        # However, if best_count is high, and we are low, we might not win.
        # But we can't prune easily without knowing the distribution.
        
        # Let's just count.
        for i in range(end_idx):
            a = powerful_list[i]
            b = N - a
            if b in powerful_set:
                count += 1
                
        if count > best_count:
            best_count = count
            best_N = N
            # We don't store pairs yet to save time/memory, we'll regenerate for best_N at the end.
            # But wait, if we have multiple N with same max count, we pick the first.
            # That's fine.
            
    # Now, construct pairs for best_N
    pairs = []
    if best_N > 0:
        limit_a = best_N // 2
        end_idx = bisect.bisect_right(powerful_list, limit_a)
        for i in range(end_idx):
            a = powerful_list[i]
            b = best_N - a
            if b in powerful_set:
                pairs.append([a, b])
                
    return {'N': best_N, 'pairs': pairs}
