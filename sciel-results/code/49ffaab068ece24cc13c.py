import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin (150s kill, 20s buffer)
    
    # Initialize with known seed
    best_M = 333312
    best_ks = [336, 372, 434]
    
    def update_best(M, ks):
        nonlocal best_M, best_ks
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks

    def search_up_to(K):
        """
        Computes S(k) = k * sigma(k) for k in [1, K] using a fast sieve.
        Updates best_M and best_ks if a better collision is found.
        """
        if time.time() > deadline:
            return
        
        # 1. Sieve for sigma using numpy for speed
        # sigma[i] will hold the sum of divisors of i
        sigma = np.zeros(K + 1, dtype=np.int64)
        
        # Optimization: The inner loop is the bottleneck.
        # We iterate i from 1 to K, adding i to multiples of i.
        # This is O(K log K).
        # Using numpy slicing: sigma[i::i] += i
        # This is much faster than a Python loop.
        
        for i in range(1, K + 1):
            if time.time() > deadline:
                return
            sigma[i::i] += i
        
        # 2. Compute M = k * sigma(k) for all k
        # We need to be careful with overflow. 
        # k <= K, sigma(k) can be up to ~K * log(log K) or so? No, sigma(n) < 2.5n for n>1 usually?
        # Actually sigma(n) can be large. But K is at most a few million.
        # Max K ~ 2*10^7. sigma(k) ~ k * 2. So M ~ 2*k^2 ~ 8*10^14, well within int64.
        # Even if K=10^8, M ~ 10^16, still within int64 (9.2*10^18).
        
        ks = np.arange(1, K + 1, dtype=np.int64)
        Ms = ks * sigma
        
        # Filter out M > 10^18
        valid_mask = Ms <= 10**18
        Ms = Ms[valid_mask]
        ks = ks[valid_mask]
        
        # 3. Group by M
        # We want to find M values that appear most frequently.
        # Since Ms is sorted? No, Ms is not necessarily sorted.
        # But we can sort Ms and then count consecutive equal values.
        # Or use np.unique with return_counts.
        
        # np.unique is O(N log N) which is fine for N ~ 10^7.
        unique_Ms, counts, inverse = np.unique(Ms, return_counts=True, return_inverse=True)
        
        # Find the maximum count
        max_count = np.max(counts)
        
        if max_count > len(best_ks):
            # Find all M values that have this max_count
            # We need to return the list of k's for one such M.
            # The task says: "The score is the number of listed solutions".
            # It implies we pick ONE M and list ALL its solutions k.
            # Wait, the contract says: "construct() -> {'M': M, 'k': [k_1, ...]} ... score is the number of listed solutions".
            # So we pick the M with the most k's.
            
            # Get the indices of the unique_Ms that have max_count
            best_M_indices = np.where(counts == max_count)[0]
            
            # Pick the first one (or any)
            # We need to retrieve the k's corresponding to this M.
            # inverse tells us which unique_M index each element in Ms belongs to.
            
            # Let's pick the M with the smallest value? Or just the first one found?
            # It doesn't matter for the score, only the count.
            # But we need to extract the k's.
            
            target_M_idx = best_M_indices[0]
            target_M = unique_Ms[target_M_idx]
            
            # Find all k's where inverse == target_M_idx
            # This is O(N).
            # mask = (inverse == target_M_idx)
            # found_ks = ks[mask]
            
            # More efficient:
            # We can use np.where(inverse == target_M_idx)
            found_indices = np.where(inverse == target_M_idx)[0]
            found_ks = ks[found_indices]
            
            # Convert to Python list of ints
            # Check constraint: k_i <= isqrt(M)
            # Since M = k*sigma(k) and sigma(k) >= k+1 for k>1, M >= k(k+1) > k^2.
            # So k < sqrt(M) is always true for k > 1.
            # For k=1, M=1, sqrt(M)=1, 1<=1 holds.
            # So all k found are valid.
            
            k_list = [int(k) for k in found_ks]
            
            # Verify distinctness (they are distinct by construction of k range)
            # Verify types
            update_best(int(target_M), k_list)

    # Strategy:
    # We want to maximize K.
    # Time complexity is dominated by the sieve loop.
    # Let's benchmark with a small K first to estimate speed.
    
    # Benchmark with K=100,000
    K_bench = 100000
    t_start = time.time()
    search_up_to(K_bench)
    t_bench = time.time() - t_start
    
    # The time is roughly linear in K * log(K) due to the harmonic series in the sieve.
    # T(K) ~ C * K * log(K)
    # T(K_bench) = C * K_bench * log(K_bench)
    # We want to find K such that T(K) < deadline - elapsed
    # C * K * log(K) < remaining_time
    # (K * log(K)) / (K_bench * log(K_bench)) < remaining_time / T_bench
    # Let ratio = remaining_time / T_bench
    # K * log(K) < ratio * K_bench * log(K_bench)
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining <= 0:
        return {'M': best_M, 'k': best_ks}
        
    # Estimate max K
    # Let's just try a few powers of 2 or 10 and pick the largest that fits.
    # Or solve for K.
    # log(K) * K = A
    # K = A / log(K)
    # Iterate: K_0 = A, K_{n+1} = A / log(K_n)
    
    A = (remaining / t_bench) * K_bench * math.log(K_bench)
    
    K_est = A
    for _ in range(10):
        K_est = A / math.log(K_est + 1)
        
    # Cap K_est to something reasonable, e.g., 50,000,000
    # Memory for sigma array of size 50M int64 is 400MB.
    # Memory for Ms, ks, unique, etc. adds more.
    # Let's cap at 20,000,000 to be safe on memory (160MB for sigma, plus overhead).
    # Actually, np.unique on 20M elements might be slow or memory heavy.
    # Let's cap at 10,000,000 first.
    
    max_K_allowed = 10000000
    K_target = int(min(K_est, max_K_allowed))
    
    # Ensure K_target is at least K_bench (we already ran that)
    if K_target <= K_bench:
        return {'M': best_M, 'k': best_ks}
        
    # Run the search for K_target
    search_up_to(K_target)
    
    # If we have time left, can we go bigger?
    # The estimation might be off.
    # Let's check time again.
    elapsed = time.time() - start_time
    if time.time() < deadline - 5.0:
         # Try to go bigger?
         # The relationship is K log K.
         # If we used T, we can try T * 1.5?
         # But memory might be the limit.
         # Let's stick to one big run to avoid complexity and memory spikes.
         pass

    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
