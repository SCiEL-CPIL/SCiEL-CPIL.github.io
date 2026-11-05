import time
import numpy as np
from math import isqrt
from collections import defaultdict

def _powerful_numbers_up_to(M):
    """Generate all powerful numbers up to M using a sparse approach."""
    if M < 1:
        return set()
    
    powerful = set()
    # Powerful numbers are of the form u^2 * v^3 where v is square-free.
    # To generate them up to M:
    # v^3 <= M => v <= M^(1/3)
    # u^2 <= M / v^3 => u <= sqrt(M / v^3)
    
    max_v = int(M ** (1/3)) + 2
    # Optimization: Only iterate v that are square-free? 
    # Actually, if v is not square-free, say v = k^2 * w, then u^2 * k^6 * w^3 = (u*k^3)^2 * w^3.
    # This generates duplicates. Using all v is simpler but slower.
    # Given M up to 1e18, M^(1/3) is 1e6. Iterating 1e6 is fast.
    
    for v in range(1, max_v + 1):
        v_cubed = v ** 3
        if v_cubed > M:
            break
        
        # Check if v is square-free to avoid duplicates?
        # If v is not square-free, the generated numbers are still powerful, just duplicated.
        # We use a set, so duplicates are handled.
        # However, skipping non-square-free v reduces the number of iterations slightly?
        # No, the inner loop size depends on v.
        # Let's just iterate all v. The set handles uniqueness.
        
        max_u = isqrt(M // v_cubed)
        # Generate u^2 * v^3 for u in 1..max_u
        # This is an arithmetic progression of squares scaled.
        # For large max_u, this loop is heavy.
        # But sum of max_u over all v is roughly sum sqrt(M/v^3) ~ M^0.5 * sum v^-1.5 ~ O(M^0.5).
        # For M=1e18, this is ~1e9 operations? That might be too slow in Python.
        
        # Wait, the previous code did this for M=5e7 and it was fine.
        # For M=1e18, this will take minutes/hours.
        
        # We need a faster generation or a different strategy.
        # The prompt says "generate all powerful numbers up to 10^18 ... ~10^7 count".
        # If the count is 10^7, generating them should be fast if we can iterate them directly.
        
        # Is there a way to iterate powerful numbers in order without checking every u?
        # Not easily.
        
        # Let's reconsider the time budget. 120s.
        # Generating 10^7 numbers in Python:
        # If we do 10^7 additions/inserts, it's ~1s-2s.
        # The bottleneck is the generation logic.
        # Sum_{v=1}^{1e6} sqrt(1e18 / v^3) = 1e9 * Sum_{v=1}^{1e6} v^-1.5.
        # Sum v^-1.5 converges to ~2.6. So ~2.6 * 1e9 = 2.6e9 iterations.
        # This is too slow for Python (approx 2600 seconds).
        
        # We MUST optimize the generation or limit M.
        # The previous program used M=5e7.
        # 5e7^(1/3) ~ 370.
        # Sum_{v=1}^{370} sqrt(5e7 / v^3) ~ sqrt(5e7) * 2.6 ~ 7000 * 2.6 ~ 18000 iterations.
        # That's why it was fast.
        
        # If we want to go to M=1e12:
        # M^(1/3) = 1e4.
        # Sum ~ sqrt(1e12) * 2.6 = 1e6 * 2.6 = 2.6e6 iterations.
        # This is feasible in Python (< 1s).
        
        # If we want M=1e15:
        # M^(1/3) = 1e5.
        # Sum ~ sqrt(1e15) * 2.6 = 3.16e7 * 2.6 ~ 8e7 iterations.
        # This is ~80-100 seconds in Python. Borderline.
        
        # So we can probably afford M up to 1e15 or maybe 2e15.
        # Let's try M = 2 * 10^15.
        
        # Actually, let's stick to the bucketing strategy but with a realistic M.
        # If we generate up to M, we can search for N up to 2M.
        
        # Let's implement the generation with a time check.
        
        # To speed up generation, we can use numpy for the inner loop if max_u is large enough?
        # u^2 * v^3.
        # u = np.arange(1, max_u+1)
        # nums = (u**2) * v_cubed
        # powerful.update(nums.tolist())
        # This vectorizes the inner loop.
        
        if max_u > 100:
            u = np.arange(1, max_u + 1, dtype=np.int64)
            nums = (u * u) * v_cubed
            # Filter out > M just in case of overflow or rounding
            nums = nums[nums <= M]
            powerful.update(nums.tolist())
        else:
            for u in range(1, max_u + 1):
                val = u * u * v_cubed
                if val <= M:
                    powerful.add(val)
                    
    return powerful

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Strategy:
    # 1. Generate powerful numbers up to M.
    # 2. Sort them.
    # 3. Use bucketed FFT to find dense sum regions.
    
    # Try a sequence of M values, starting large and decreasing if time runs out.
    # M = 10^15 is likely too slow for generation in pure Python even with numpy inner loop?
    # Let's test M = 10^12 first (fast), then try 10^14.
    
    target_Ms = [
        10**14,
        10**13,
        10**12,
        10**11
    ]
    
    for M in target_Ms:
        if time.time() > deadline - 30:
            break
            
        try:
            # Generate powerful numbers
            t_gen_start = time.time()
            powerful_set = _powerful_numbers_up_to(M)
            t_gen_end = time.time()
            
            if time.time() > deadline - 20:
                continue
                
            powerful_list = sorted(powerful_set)
            K = len(powerful_list)
            
            if K < 2:
                continue
            
            # Bucketing Strategy
            # We want to find N with many representations a+b=N.
            # Split powerful_list into buckets based on magnitude.
            # Bucket i contains numbers in [2^i, 2^(i+1)).
            # We only need to convolve buckets that can sum to N.
            # To find the global max, we can convolve all pairs of buckets (i, j) 
            # where the sum of ranges is within [MinN, MaxN].
            # Actually, simpler:
            # Just do a full FFT on the sparse array?
            # No, M is 1e14, array size 1e14 is impossible.
            
            # So we MUST use bucketing.
            
            # Let's define buckets by powers of 2.
            # Bucket 0: [1, 2)
            # Bucket 1: [2, 4)
            # ...
            # Bucket k: [2^k, 2^(k+1))
            
            # For a given N, a is in bucket i, b is in bucket j.
            # a ~ 2^i, b ~ 2^j.
            # N ~ 2^i + 2^j.
            
            # We can iterate over pairs of buckets (i, j) with i <= j.
            # For each pair, extract the numbers in those buckets.
            # Do an FFT convolution of these two sparse lists.
            # The result is the count of pairs (a,b) with a in bucket i, b in bucket j, summing to N.
            # We sum these counts for all (i,j) to get the total count for N.
            
            # However, doing this for ALL pairs of buckets is expensive if there are many buckets.
            # Number of buckets for M=1e14 is ~46.
            # Pairs ~ 46^2 / 2 ~ 1000.
            # Each FFT size depends on the range of sums.
            # Bucket i range: [2^i, 2^(i+1)).
            # Bucket j range: [2^j, 2^(j+1)).
            # Sum range: [2^i + 2^j, 2^(i+1) + 2^(j+1)).
            # Width of sum range is approx 2^(max(i,j)+1).
            # For i=j=k, width is ~ 2^(k+2).
            # FFT size ~ 2^(k+2).
            # Total cost ~ sum_{i,j} 2^(max(i,j)+2) * log(2^(max(i,j)+2)).
            # This is dominated by the largest buckets.
            # Largest bucket k ~ 46. Size ~ 2^48. Too big.
            
            # We need to be smarter.
            # We only care about N where the count is high.
            # High count N usually comes from overlapping "dense" regions.
            # Powerful numbers are sparse.
            # Maybe we should just check the sums of the largest powerful numbers?
            # No, small numbers are denser.
            
            # Alternative:
            # Use the "Random Sampling" inspiration?
            # Or "Two-pointer" on sorted list?
            
            # Let's try a different structural change:
            # Instead of FFT, use a **Hash Map of Sums** for a subset?
            # No, K is large.
            
            # Let's refine the Bucket FFT.
            # We can limit the FFT size by only considering a "window" of N.
            # But we don't know where the window is.
            
            # What if we only convolve buckets that are "close" in magnitude?
            # If i and j differ significantly, the sum is dominated by the larger one.
            # The number of representations for a fixed N from (i,j) with i << j is small.
            # The maximum number of representations likely comes from i ~ j.
            
            # Let's try:
            # 1. Generate powerful numbers up to M.
            # 2. Sort them.
            # 3. For each bucket k, take the numbers in bucket k.
            # 4. Convolve bucket k with itself (and maybe k-1, k+1).
            # 5. Find the N with max count in these local convolutions.
            # 6. Verify.
            
            # This reduces the number of FFTs significantly.
            
            buckets = defaultdict(list)
            for p in powerful_list:
                # Find bucket index
                # p is in [2^k, 2^(k+1))
                # k = floor(log2(p))
                k = p.bit_length() - 1
                buckets[k].append(p)
            
            max_k = max(buckets.keys()) if buckets else 0
            
            # We will process bucket pairs (i, j) where |i-j| <= 2.
            # And we will limit the FFT size to a reasonable max, say 2^20.
            # If the bucket range is too large, we split it further?
            # Or just skip very large buckets for FFT and use a different method for them?
            # Actually, for large buckets, the numbers are sparse.
            # We can just use a set intersection approach for large buckets?
            
            # Let's try the local convolution approach.
            
            local_best_count = 0
            local_best_N = 0
            local_best_pairs = []
            
            # We need to accumulate counts for N across all bucket pairs.
            # Since N can be large, we can't use a global array.
            # We can use a dictionary: N -> count.
            # But N can be up to 2M.
            
            # Let's just find the top candidates from each bucket pair convolution,
            # verify them, and keep the global best.
            
            for i in range(max_k + 1):
                if time.time() > deadline - 10:
                    break
                
                list_i = buckets.get(i, [])
                if not list_i:
                    continue
                    
                for j in range(i, min(i + 3, max_k + 1)):
                    list_j = buckets.get(j, [])
                    if not list_j:
                        continue
                    
                    # Perform FFT convolution of list_i and list_j
                    # To avoid huge FFTs, we can downsample or limit size?
                    # No, we need exact counts.
                    
                    # If the range is too large, skip?
                    # Max val in i: ~2^(i+1)
                    # Max val in j: ~2^(j+1)
                    # Max sum: ~2^(j+2)
                    # Min sum: ~2^i + 2^j
                    
                    # If max sum > 2^24, FFT might be slow/memory heavy.
                    # Let's cap the FFT size.
                    max_sum = (1 << (j + 2))
                    if max_sum > 2**24:
                        # Use a sparse method for this pair?
                        # Or just skip?
                        # Skipping might miss the best N if it's in a large bucket.
                        # But large buckets have fewer numbers, so counts are likely lower.
                        # Let's try a simple loop for large buckets if list sizes are small.
                        if len(list_i) * len(list_j) < 10**6:
                            # Direct counting
                            set_j = set(list_j)
                            # We want to count pairs (a,b) with a in list_i, b in list_j, a+b=N.
                            # This is O(K_i * K_j) to generate all sums.
                            # Too slow if K is large.
                            # But we assumed product < 1e6.
                            # So we can generate all sums and count frequencies.
                            from collections import Counter
                            sums = Counter()
                            for a in list_i:
                                for b in list_j:
                                    sums[a+b] += 1
                            if sums:
                                max_n, max_c = sums.most_common(1)[0]
                                if max_c > local_best_count:
                                    # Verify and update
                                    # We need to find the actual pairs for max_n
                                    # This requires another pass.
                                    # For now, just track count.
                                    # We'll do a final verification step.
                                    local_best_count = max_c
                                    local_best_N = max_n
                        continue
                    
                    # FFT for smaller ranges
                    # Determine size
                    size = 1
                    while size < max_sum + 1:
                        size *= 2
                    if size > 2**24:
                        continue
                        
                    A = np.zeros(size, dtype=np.float64)
                    B = np.zeros(size, dtype=np.float64)
                    
                    for p in list_i:
                        A[p] = 1.0
                    for p in list_j:
                        B[p] = 1.0
                        
                    F = np.fft.rfft(A)
                    G = np.fft.rfft(B)
                    H = F * G
                    counts = np.fft.irfft(H, n=size)
                    
                    # Find max in valid range
                    min_sum = (1 << i) + (1 << j)
                    # max_sum is approx 2^(j+2), but let's use size-1
                    # Actually, max possible sum is max(list_i) + max(list_j)
                    real_max_sum = max(list_i) + max(list_j)
                    
                    segment = counts[min_sum:real_max_sum+1]
                    if segment.size == 0:
                        continue
                        
                    max_val = np.max(segment)
                    if max_val > local_best_count:
                        # Find the index
                        idx = np.argmax(segment) + min_sum
                        local_best_count = int(max_val)
                        local_best_N = int(idx)
                        
            # After scanning all bucket pairs, we have a candidate N (local_best_N)
            # We must verify it exactly and retrieve pairs.
            
            if local_best_N > 0 and local_best_count > best_count:
                N = local_best_N
                half_N = N // 2
                pairs = []
                # Use the global powerful_set for verification
                # Note: powerful_set contains numbers up to M.
                # If N > M, b = N-a might be > M.
                # But we generated powerful numbers up to M.
                # If a <= M and b <= M, then a+b <= 2M.
                # Our search was limited to sums of numbers in [1, M].
                # So N <= 2M.
                # If b > M, it won't be in powerful_set.
                # So we only find pairs where both a,b <= M.
                # This is a valid subset of all pairs.
                
                for a in powerful_list:
                    if a > half_N:
                        break
                    b = N - a
                    if b in powerful_set:
                        pairs.append([a, b])
                        
                if len(pairs) > best_count:
                    best_count = len(pairs)
                    best_N = N
                    best_pairs = pairs
                    
        except MemoryError:
            continue
        except Exception:
            continue
            
    # Fallback
    if best_count <= 1:
        small_candidates = [5, 9, 12, 16, 18, 20, 25, 27, 32, 36, 45, 48, 50, 64, 81, 98, 100]
        for N in small_candidates:
            if time.time() > deadline - 2:
                break
            M_small = N
            p_set = _powerful_numbers_up_to(M_small)
            half = N // 2
            pairs = []
            for a in sorted(p_set):
                if a > half:
                    break
                b = N - a
                if b in p_set:
                    pairs.append([a, b])
            if len(pairs) > best_count:
                best_count = len(pairs)
                best_N = N
                best_pairs = pairs
                
    return {'N': best_N, 'pairs': best_pairs}
