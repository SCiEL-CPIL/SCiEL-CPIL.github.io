import time
import numpy as np
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 10s safety margin
    deadline = start_time + 140.0
    
    def generate_powerful(limit):
        """
        Generate all powerful numbers <= limit.
        A number is powerful if it is of the form u^2 * v^3.
        """
        pw = set()
        # v^3 <= limit => v <= limit^(1/3)
        max_v = int(limit ** (1/3)) + 2
        
        for v in range(1, max_v + 1):
            v_cubed = v ** 3
            if v_cubed > limit:
                break
            # u^2 <= limit / v^3 => u <= sqrt(limit / v^3)
            max_u = isqrt(limit // v_cubed)
            for u in range(1, max_u + 1):
                p = u * u * v_cubed
                if p <= limit:
                    pw.add(p)
        return sorted(pw)

    # Determine M based on available time
    # Generating powerful numbers up to M=10^8 takes ~1-2s.
    # FFT bucketing is the main cost.
    # Let's aim for M = 50,000,000 initially. 
    # If time permits, we can try higher, but 50M is a good balance.
    # K (number of powerful numbers) for M=50M is approx ~5000-6000.
    
    M = 50_000_000
    
    if time.time() > deadline - 30:
        M = 20_000_000

    powerful_list = generate_powerful(M)
    K = len(powerful_list)
    
    if K == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    powerful_set = set(powerful_list)
    
    # Multi-Resolution FFT Strategy
    # We divide the sorted powerful_list into buckets.
    # For each pair of buckets (i, j), we compute the convolution of their indicator arrays.
    # The sum of counts for a specific N is accumulated across all bucket pairs.
    
    # Number of buckets
    # If K is small, we can use more buckets.
    # Let's target a bucket size such that the FFT size is manageable.
    # Max value in bucket i is roughly M * (i+1) / num_buckets.
    # If we use num_buckets = 10, max value in last bucket is M.
    # Convolution of last two buckets requires array size ~ 2*M.
    # To reduce peak memory, we can split the *range* of sums?
    # No, we need all sums.
    # However, if we bucket by *index*, the values are spread out.
    # Actually, the most efficient way to reduce memory is to bucket by *value range*.
    # But powerful numbers are sparse.
    
    # Let's stick to index-based bucketing for simplicity, but be careful with the last bucket.
    # Actually, a better approach for memory is to just do the full FFT if M is small enough,
    # or use a "sparse" convolution technique if M is large.
    
    # Given M=50M, a full FFT of size 100M (2*50M) requires 100M * 8 bytes = 800MB.
    # This might be too high for some environments.
    # Let's use bucketing to keep individual FFTs smaller.
    
    num_buckets = 20
    bucket_size = max(1, K // num_buckets)
    buckets = []
    for i in range(0, K, bucket_size):
        buckets.append(powerful_list[i:i+bucket_size])
    
    # global_counts will store the count of representations for each N
    # Since N can be up to 2*M, we can't use a dense array of size 2*M (100M) if memory is tight.
    # But 100M integers is 400MB (int32) or 800MB (int64).
    # Using a dictionary is safer for memory but slower.
    # Let's use a dictionary for global_counts.
    
    global_counts = defaultdict(int)
    
    # We need to be careful with time.
    # Number of bucket pairs is num_buckets * (num_buckets + 1) / 2 ~ 210.
    # Each FFT is on arrays of size (max_val_i + max_val_j + 1).
    # For the largest buckets, this is ~ M.
    # FFT of size 50M takes ~1s. 210 * 1s = 210s. Too slow!
    
    # Optimization:
    # We only need to find the MAX count.
    # We don't need to accumulate ALL counts if we can identify hotspots.
    # However, the "Multi-Resolution" inspiration suggests this is the way.
    # How to make it faster?
    # 1. Reduce M.
    # 2. Reduce num_buckets (fewer FFTs, but larger FFTs).
    # 3. Use a smarter search.
    
    # Let's reconsider the Random Sampling approach from the current program.
    # It scored 4.66. The parent (FFT) scored 4.8.
    # The FFT approach is deterministic and likely finds the true max for the given M.
    # The bottleneck of FFT is time/memory for large M.
    
    # What if we use a HYBRID?
    # 1. Use FFT for a moderate M (e.g., 20M) to find a strong candidate N.
    # 2. Then, verify and potentially extend the search for that specific N using the full set of powerful numbers up to a larger M?
    # No, N is fixed by the sum of two numbers. If we increase M, we might find NEW pairs for the SAME N.
    # So, if we find N* with count C using M=20M, we can check if there are more pairs for N* using powerful numbers up to M=100M.
    # This is a "Local Refinement" strategy.
    
    # Let's implement:
    # 1. Generate powerful numbers up to M_small = 20,000,000.
    # 2. Perform Full FFT on M_small to find N_best and its count C_small.
    # 3. Generate powerful numbers up to M_large = 100,000,000.
    # 4. Check N_best against the full set up to M_large to get C_large.
    # 5. Also, check a few other top candidates from the small FFT.
    
    # This avoids the cost of FFT on M_large.
    
    M_small = 20_000_000
    M_large = 80_000_000 # Can go higher if time permits
    
    # Step 1: Small FFT
    pw_small = generate_powerful(M_small)
    K_small = len(pw_small)
    
    # Create dense array for FFT
    # Size must be at least 2 * M_small
    # Next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * M_small:
        fft_size *= 2
        
    arr_small = np.zeros(fft_size, dtype=np.float32)
    for p in pw_small:
        arr_small[p] = 1.0
        
    # FFT
    f_small = np.fft.rfft(arr_small)
    f_small_sq = f_small * f_small
    conv_small = np.fft.irfft(f_small_sq, n=fft_size)
    
    # The count for sum N is conv_small[N]
    # Note: conv_small includes (a,a) pairs. 
    # Number of pairs (a,b) with a<=b and a+b=N is:
    # If N is even, (conv[N] + 1) // 2 ? No.
    # conv[N] = sum_{a+b=N} arr[a]arr[b].
    # This counts ordered pairs (a,b).
    # If a != b, (a,b) and (b,a) are both counted.
    # If a == b, (a,a) is counted once.
    # We want unordered distinct pairs a <= b.
    # Let C_ordered = conv[N].
    # Let D = number of pairs with a == b (i.e., 2a = N).
    # C_ordered = 2 * (pairs with a != b) + D
    # Pairs with a <= b = (pairs with a != b) + D = (C_ordered - D) / 2 + D = (C_ordered + D) / 2.
    # D is 1 if N/2 is a powerful number, 0 otherwise.
    
    # Find top candidates from small FFT
    # We only care about N <= 2 * M_small
    # Get top 20 indices
    top_indices = np.argsort(conv_small[:2*M_small])[::-1][:20]
    
    candidates = []
    for idx in top_indices:
        N_cand = int(idx)
        if N_cand < 2:
            continue
        # Calculate exact count for this N using pw_small
        # This is to get the exact integer count and verify
        c_exact = 0
        for a in pw_small:
            if a > N_cand // 2:
                break
            b = N_cand - a
            if b in set(pw_small): # Use set for O(1)
                c_exact += 1
        candidates.append((N_cand, c_exact))
        
    # Sort candidates by count
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Step 2: Large Verification
    # Generate powerful numbers up to M_large
    pw_large = generate_powerful(M_large)
    pw_large_set = set(pw_large)
    
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    # Verify top candidates from small search against the large set
    # Also, we should check if there are NEW N values in the range (2*M_small, 2*M_large] 
    # that might have high counts?
    # Unlikely to have *more* representations than the top ones in the small range,
    # because the density of powerful numbers is higher in the small range (relative to N)?
    # Actually, the number of representations r(N) is roughly proportional to the number of divisors or similar arithmetic functions.
    # It's safer to assume the max is in the small range, but let's verify a few random large N?
    # No, let's just verify the top 10 candidates from the small FFT.
    
    for N_cand, _ in candidates[:10]:
        if time.time() > deadline - 10:
            break
            
        # Count pairs in pw_large
        exact_count = 0
        pairs = []
        # Iterate through a in pw_large such that a <= N_cand // 2
        # pw_large is sorted
        for a in pw_large:
            if a > N_cand // 2:
                break
            b = N_cand - a
            if b in pw_large_set:
                exact_count += 1
                pairs.append([a, b])
                
        if exact_count > best_count:
            best_count = exact_count
            best_N = N_cand
            best_pairs = pairs
            
    # What if the best N is NOT in the top 10 of the small FFT?
    # This is a risk.
    # To mitigate, we can do a random sampling check on the large set for N values 
    # that are sums of two large powerful numbers.
    
    # Let's do a quick random sample from the large set to see if we find a better N.
    # Sample size S=1000.
    if time.time() < deadline - 20:
        import random
        random.seed(42)
        S = min(1000, K)
        indices = random.sample(range(K), S)
        sample = [pw_large[i] for i in indices]
        
        counts_large = defaultdict(int)
        for i in range(len(sample)):
            a = sample[i]
            for j in range(i, len(sample)):
                b = sample[j]
                N_val = a + b
                counts_large[N_val] += 1
                
        # Get top 5 from this sample
        top_large = sorted(counts_large.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for N_cand, c in top_large:
            # Verify
            exact_count = 0
            pairs = []
            for a in pw_large:
                if a > N_cand // 2:
                    break
                b = N_cand - a
                if b in pw_large_set:
                    exact_count += 1
                    pairs.append([a, b])
                    
            if exact_count > best_count:
                best_count = exact_count
                best_N = N_cand
                best_pairs = pairs

    return {'N': best_N, 'pairs': best_pairs}
