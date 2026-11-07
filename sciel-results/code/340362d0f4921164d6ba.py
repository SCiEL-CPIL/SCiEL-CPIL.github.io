"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import bisect

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    N = 10000000
    
    # 1. Sieve for sigma: sigma[n] = sum of divisors of n
    # Using NumPy for vectorized updates
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Optimization: Only iterate d up to N
    for d in range(1, N + 1):
        sigma[d::d] += d
        
        # Check deadline periodically
        if d % 500000 == 0:
            if time.time() - t0 > DEADLINE - 10: # Leave 10s for sorting/search
                return [11, 6]
                
    if time.time() - t0 > DEADLINE - 10:
        return [11, 6]
        
    # 2. Sort indices by sigma value, then by index n
    # This allows us to process groups of equal sigma in sorted order of n
    # argsort is stable, so sorting by sigma then n is equivalent to sorting by (sigma, n)
    # We create an array of indices 1..N
    indices = np.arange(1, N + 1)
    
    # Sort indices based on sigma[indices]
    # We need to sort by (sigma[n], n). 
    # np.lexsort does this. Keys are (last key, ..., first key).
    # So keys = (indices, sigma[indices]) means primary key is sigma, secondary is index.
    # Wait, lexsort sorts by the last key first.
    # keys = (indices, sigma[indices]) -> sorts by sigma[indices], then by indices.
    sorted_indices = np.lexsort((indices, sigma[indices]))
    
    if time.time() - t0 > DEADLINE - 5:
        return [11, 6]
        
    # 3. Identify groups of equal sigma
    sorted_sigmas = sigma[sorted_indices]
    
    # Find where sigma value changes
    # diff > 0 indicates a change (since sorted, it can only increase or stay same)
    change_mask = np.diff(sorted_sigmas) > 0
    # Start indices of each group
    group_starts = np.concatenate(([0], np.where(change_mask)[0] + 1))
    # End indices of each group
    group_ends = np.concatenate((np.where(change_mask)[0] + 1, [len(sorted_indices)]))
    
    # 4. Iterate through groups
    best_num = 4  # |2*11 - 3*6| = 4
    best_den = 12 # 2*6
    best_a = 11
    best_b = 6
    
    # Precompute 1.5 as float for comparison, but we use integers for exactness
    # We want to minimize |2a - 3b| / (2b)
    
    for start, end in zip(group_starts, group_ends):
        if time.time() - t0 > DEADLINE - 2:
            break
            
        group_size = end - start
        if group_size < 2:
            continue
            
        # Extract the sorted list of n for this group
        # sorted_indices[start:end] are the indices in the original array (which are the values n)
        # Since we sorted by (sigma, n), these n values are already sorted in increasing order.
        L = sorted_indices[start:end]
        
        # Two-pointer search for best pair (a, b) in L minimizing |2a - 3b|
        # Let b = L[i]. We want a close to 1.5 * b.
        # Since L is sorted, the index j for a is monotonic with i.
        
        j = 0
        # We iterate b over L
        # Note: a and b must be distinct? 
        # If a == b, then |a/b - 1.5| = |1 - 1.5| = 0.5.
        # If a != b, we can get much closer.
        # The problem says "Minimize |a/b - 3/2|". It does not explicitly forbid a=b, 
        # but sigma(a)=sigma(b) is trivially true if a=b.
        # However, usually "ratio" implies distinct or the trivial solution is not the goal.
        # The fallback [11, 6] has a != b.
        # Let's check if a=b is allowed. "Approximate a ratio...". 
        # If a=b, ratio is 1. Error 0.5.
        # If we find a pair with error < 0.5, it's better.
        # The prompt says "positive Python integers... sigma(a)=sigma(b)".
        # It does NOT say a != b.
        # BUT, the fallback is [11,6]. And the score is based on the raw value.
        # If a=b is allowed, the best score would be for any a where sigma(a) is shared? No, if a=b, sigma(a)=sigma(a) is always true.
        # So if a=b is allowed, we can pick ANY a. 
        # Then we just minimize |a/a - 1.5| = 0.5.
        # This would be a constant score for any valid a.
        # However, the task implies finding a *non-trivial* pair or the "approximation" suggests we are looking for a specific structure.
        # Let's look at the seed score: 4077490.0000.
        # This is a very large number.
        # Wait, the score is `seed/value`.
        # If the value is small, the score is large.
        # The current score is ~4e6.
        # If the value was 0.5, the score would be seed/0.5 = 2*seed.
        # What is the seed? 
        # The prompt says "1.0 equals the seed program".
        # The current program has score 4077490.
        # This implies the "seed" in the formula `seed/value` is not 1.0 but a specific number, OR the score is defined differently.
        # "The search score is seed/value, so 1.0 equals the seed program".
        # This implies the seed program's value is `seed`.
        # So the seed program's raw error is `seed`.
        # If the seed program returns [11, 6], error is |11/6 - 1.5| = |1.833 - 1.5| = 0.333.
        # So seed = 0.333? No, the score is 4077490.
        # Maybe the score is `1 / value`? No, "seed/value".
        # Let's assume the "seed" is a large constant defined by the evaluator, and the "value" is the raw error.
        # Or maybe the "value" is the integer numerator/denominator representation?
        # "score is the binary64 serialization of that rational".
        # This part is confusing. "raw value is float(|a/b - 3/2|)".
        # "score is seed/value".
        # If value is ~0.333, and score is ~4e6, then seed is ~1.3e6.
        
        # Regardless, we want to MINIMIZE the raw value |a/b - 1.5|.
        
        # Back to the two-pointer search.
        # We assume a and b should be distinct to get a better approximation than 0.5?
        # Actually, if we can get close to 1.5, the error is much smaller than 0.5.
        # So we definitely want distinct a, b if possible.
        
        # Two pointer logic:
        # For each b = L[i], we want a in L such that a is close to 1.5 * b.
        # Let target = 1.5 * b.
        # We maintain j such that L[j] is the smallest element >= target? 
        # Or we can just check neighbors.
        
        # Since L is sorted, and b increases, target increases. So j is non-decreasing.
        
        # Initialize j to 0.
        # For i in range(group_size):
        #    b = L[i]
        #    target = 1.5 * b
        #    # Advance j while L[j] < target - some_margin? 
        #    # Actually, just advance j while L[j] < target.
        #    # Then L[j] is the first >= target. L[j-1] is the last < target.
        #    # Check L[j] and L[j-1] as candidates for a.
        
        # One complication: a must be in the SAME group.
        # L contains all elements in the group.
        
        # Let's implement this.
        
        j = 0
        for i in range(group_size):
            b = int(L[i])
            # target = 1.5 * b
            # To avoid float precision issues, we compare 2*a with 3*b
            # We want 2*a close to 3*b.
            
            # Advance j
            # We want L[j] such that 2*L[j] >= 3*b
            # while j < group_size and 2 * int(L[j]) < 3 * b:
            #     j += 1
            
            # This inner while loop makes it O(K^2) in worst case if j doesn't reset?
            # No, j only increases. Total operations for one group is O(K).
            
            # However, python loop overhead for `while` might be high.
            # We can use bisect on the numpy array slice?
            # bisect.bisect_left(L, target) is O(log K).
            # O(K log K) vs O(K).
            # For small K, bisect is fine. For large K, two-pointer is better.
            # Given N=10^7, max group size is likely small (< 1000).
            # Bisect is implemented in C and very fast.
            # Let's stick with bisect for simplicity and speed of implementation, 
            # unless the group size is huge.
            
            # Actually, let's use the two-pointer approach with numpy vectorization?
            # No, the logic is sequential.
            
            # Let's try bisect first, it's simpler and likely fast enough for K < 10000.
            target = 1.5 * b
            idx = bisect.bisect_left(L, target)
            
            candidates = []
            if idx < group_size:
                candidates.append(L[idx])
            if idx > 0:
                candidates.append(L[idx-1])
                
            for a in candidates:
                a_int = int(a)
                if a_int == b:
                    continue
                    
                # Calculate error
                num = abs(2 * a_int - 3 * b)
                if num == 0:
                    # Exact 3/2. Invalid per rules? "raw value of exactly 0 cannot be scored"
                    # So we must skip or handle. The rule says "counts as invalid".
                    # So we must avoid num == 0.
                    continue
                    
                den = 2 * b
                
                # Compare num/den with best_num/best_den
                # num * best_den < best_num * den
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a = a_int
                    best_b = b
                    
    # Final verification
    a = int(best_a)
    b = int(best_b)
    
    if a <= 0 or b <= 0 or a > N or b > N:
        return [11, 6]
        
    if sigma[a] != sigma[b]:
        return [11, 6]
        
    if 2 * a == 3 * b:
        return [11, 6]
        
    return [a, b]

# EVOLVE-BLOCK-END
