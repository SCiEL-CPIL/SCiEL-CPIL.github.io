"""Search for M with many distinct k where k*sigma(k)=M, k<=isqrt(M)."""
import math
import time
import sys
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: Use NumPy for fast sigma sieve and vectorized grouping.
    # Increase K to 50,000,000 if memory allows.
    
    K = 50000000
    
    try:
        # 1. Compute Sigma using NumPy
        sigma = np.zeros(K + 1, dtype=np.int64)
        for d in range(1, K + 1):
            sigma[d::d] += d
        
        # 2. Compute M values
        k_vals = np.arange(1, K + 1, dtype=np.int64)
        M_vals = k_vals * sigma[1:K+1]
        
        # Filter M <= 10^18
        mask = M_vals <= 10**18
        M_filtered = M_vals[mask]
        k_filtered = k_vals[mask]
        
        if len(M_filtered) == 0:
            pass
        else:
            # 3. Find most frequent M
            unique_M, counts = np.unique(M_filtered, return_counts=True)
            max_count = np.max(counts)
            
            if max_count > best_count:
                best_count = int(max_count)
                target_M = unique_M[np.argmax(counts)]
                best_M = int(target_M)
                
                # Get all k that map to best_M
                # Use boolean mask on M_filtered
                target_ks = k_filtered[M_filtered == target_M]
                best_k = [int(x) for x in target_ks]
                
    except MemoryError:
        # Fallback to smaller K if memory fails
        K = 10000000
        try:
            sigma = np.zeros(K + 1, dtype=np.int64)
            for d in range(1, K + 1):
                sigma[d::d] += d
            
            k_vals = np.arange(1, K + 1, dtype=np.int64)
            M_vals = k_vals * sigma[1:K+1]
            
            mask = M_vals <= 10**18
            M_filtered = M_vals[mask]
            k_filtered = k_vals[mask]
            
            if len(M_filtered) > 0:
                unique_M, counts = np.unique(M_filtered, return_counts=True)
                max_count = np.max(counts)
                
                if max_count > best_count:
                    best_count = int(max_count)
                    target_M = unique_M[np.argmax(counts)]
                    best_M = int(target_M)
                    target_ks = k_filtered[M_filtered == target_M]
                    best_k = [int(x) for x in target_ks]
                    
        except MemoryError:
            # Final fallback to pure Python small K
            K = 100000
            sigma = [0] * (K + 1)
            for d in range(1, K + 1):
                for m in range(d, K + 1, d):
                    sigma[m] += d
            
            groups = defaultdict(list)
            for k in range(1, K + 1):
                M = k * sigma[k]
                if M <= 10**18:
                    groups[M].append(k)
            
            for M, ks in groups.items():
                cnt = len(ks)
                if cnt > best_count:
                    best_count = cnt
                    best_M = M
                    best_k = ks

    # Final safety: ensure types are plain Python int and list
    best_k = [int(x) for x in best_k]
    best_M = int(best_M)
    
    return {'M': best_M, 'k': best_k}
