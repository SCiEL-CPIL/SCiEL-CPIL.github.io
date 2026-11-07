import numpy as np
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_M = 333312
    best_ks = [336, 372, 434]
    
    def update_best(M, ks):
        nonlocal best_M, best_ks
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks

    def run_search(K):
        if time.time() > deadline - 5:
            return
        
        try:
            # 1. Sieve for sigma
            sigma = np.zeros(K + 1, dtype=np.int64)
            for i in range(1, K + 1):
                sigma[i::i] += i
            
            if time.time() > deadline - 10:
                return

            # 2. Compute M = k * sigma(k)
            ks = np.arange(1, K + 1, dtype=np.int64)
            vals = ks * sigma[1:]
            
            # Filter M <= 10^18
            mask = vals <= 10**18
            if not np.all(mask):
                vals = vals[mask]
                ks = ks[mask]
            
            if len(vals) == 0:
                return

            # 3. Sort and group
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_ks = ks[order]
            
            if time.time() > deadline - 5:
                return

            # Find boundaries of groups
            diffs = np.diff(sorted_vals)
            breaks = np.where(diffs != 0)[0] + 1
            starts = np.concatenate(([0], breaks))
            ends = np.concatenate((breaks, [len(sorted_vals)]))
            counts = ends - starts
            
            # Find groups with count > 1
            valid_idx = np.where(counts > 1)[0]
            
            for i in valid_idx:
                M = int(sorted_vals[starts[i]])
                k_list = sorted_ks[starts[i]:ends[i]].tolist()
                sqrt_M = math.isqrt(M)
                valid_ks = [k for k in k_list if k <= sqrt_M]
                if len(valid_ks) > len(best_ks):
                    update_best(M, valid_ks)
                    
        except Exception:
            pass

    # Try increasing K
    # Start with a large K that fits in memory
    # 10^8 takes ~800MB. 2*10^8 takes ~1.6GB.
    # Let's try 10^8 first.
    run_search(100000000)
    
    # If we have time, try larger?
    # Unlikely to be much faster, but maybe 2*10^8?
    # The sieve time scales linearly with K.
    # Sorting time scales with K log K.
    # If 10^8 took 20s, 2*10^8 might take 40s + 45s = 85s.
    # Total 105s. Fits in 145s.
    if time.time() < deadline - 60:
        run_search(200000000)
        
    return {'M': int(best_M), 'k': best_ks}
