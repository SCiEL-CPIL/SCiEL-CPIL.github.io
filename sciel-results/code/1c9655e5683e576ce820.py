import time
import math
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for safety
    
    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        # x = u^2 * v^3 where v is square-free
        max_v = int(round(limit ** (1/3))) + 2
        for v in range(1, max_v + 1):
            if not is_square_free(v):
                continue
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    # Generate powerful numbers up to a limit
    # 10^10 allows N up to 2*10^10 comfortably, and enough density
    # Let's try 2 * 10^10 to be safe for N up to 10^11
    L = 20000000000  # 2 * 10^10
    
    P = generate_powerful(L)
    P_array = np.array(P, dtype=np.int64)
    L_p = len(P)
    
    if time.time() > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Generate candidate N values
    # 1. Top sums of small powerful numbers
    K = min(L_p, 3000)
    small_P = P[:K]
    
    sums = []
    # To save memory and time, we can just use a Counter
    sum_counts = Counter()
    for i in range(K):
        ai = small_P[i]
        # Optimization: only add to counter if sum is within reasonable bounds?
        # We want N to be large enough to have many reps, but small enough to verify fast.
        # Sums of small P will be small.
        # Let's just collect all.
        for j in range(i, K):
            s = ai + small_P[j]
            sum_counts[s] += 1
            
    # Get top candidates
    top_candidates = [n for n, _ in sum_counts.most_common(2000)]
    
    # 2. Add some random candidates in larger ranges
    # The density of powerful numbers is ~ sqrt(N).
    # N around 10^10 has ~ 10^5 powerful numbers below it.
    # Random sampling might hit a "lucky" N.
    import random
    random.seed(42)
    for _ in range(500):
        # Log-uniform distribution in [10^8, 10^11]
        log_low = math.log10(10**8)
        log_high = math.log10(10**11)
        log_n = random.uniform(log_low, log_high)
        n = int(10 ** log_n)
        top_candidates.append(n)
        
    # Remove duplicates
    top_candidates = list(set(top_candidates))
    
    best_N = 0
    best_pairs = []
    best_count = 0
    
    # Sort candidates by frequency from sum_counts (if available) or just iterate
    # We want to check the most promising ones first
    # Create a map for quick lookup of count for sorting
    # But top_candidates includes randoms which have count 0 in sum_counts
    # Let's just iterate. We have ~2500 candidates.
    
    # Optimization for verification:
    # For a given N, we need a in P such that a <= N/2 and N-a in P.
    # Using numpy:
    # 1. Get indices of P <= N//2
    # 2. Compute B = N - P_subset
    # 3. Check membership of B in P_array
    
    for N in top_candidates:
        if time.time() > deadline - 2:
            break
            
        if N < 2:
            continue
            
        # Find a <= N//2
        # P_array is sorted
        idx_end = np.searchsorted(P_array, N // 2)
        if idx_end == 0:
            continue
            
        P_subset = P_array[:idx_end]
        
        # Compute complements
        B = N - P_subset
        
        # Check which B are in P_array
        # np.isin is efficient
        mask = np.isin(B, P_array)
        
        count = np.sum(mask)
        
        if count > best_count:
            best_count = count
            best_N = N
            # Extract pairs
            a_vals = P_subset[mask]
            b_vals = B[mask]
            # Ensure a <= b
            # Since a <= N//2, b = N-a >= N//2 >= a. So a <= b is guaranteed.
            # However, we need to return them as list of lists
            # Convert numpy arrays to python lists
            # Only if count is reasonable to avoid memory issues?
            # Max count is likely < 10000
            best_pairs = [[int(a), int(b)] for a, b in zip(a_vals, b_vals)]
            
            # If we found a very high count, we might want to stop early or continue?
            # Continue to use remaining time for potentially better N.
            
    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': best_pairs}
