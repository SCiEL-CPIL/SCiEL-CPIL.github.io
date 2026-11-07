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
    # We can afford a larger K because grouping is vectorized.
    K = 20000000
    
    try:
        # 1. Compute Sigma using NumPy
        sigma = np.zeros(K + 1, dtype=np.int64)
        for d in range(1, K + 1):
            sigma[d::d] += d
            
        if time.time() > deadline - 30:
            return {'M': best_M, 'k': best_k}

        # 2. Compute M values
        # k is 1..K, so indices 0..K-1 in Ms correspond to k=1..K
        # sigma[1] is sigma(1)
        ks_range = np.arange(1, K + 1, dtype=np.int64)
        Ms = ks_range * sigma[1:]
        
        if time.time() > deadline - 20:
            return {'M': best_M, 'k': best_k}

        # 3. Group by M using np.unique
        # This finds unique M values and their counts
        unique_Ms, inverse, counts = np.unique(Ms, return_inverse=True, return_counts=True)
        
        # Find the index of the maximum count
        max_count_idx = np.argmax(counts)
        max_count = counts[max_count_idx]
        
        if max_count > best_count:
            best_M = int(unique_Ms[max_count_idx])
            # Find all k such that Ms[k-1] == best_M
            # np.where returns indices in Ms
            k_indices = np.where(Ms == best_M)[0]
            best_k = [int(x + 1) for x in k_indices] # k is 1-indexed
            best_count = max_count

    except MemoryError:
        # Fallback to smaller K
        K = 10000000
        try:
            sigma = np.zeros(K + 1, dtype=np.int64)
            for d in range(1, K + 1):
                sigma[d::d] += d
            
            if time.time() > deadline - 20:
                return {'M': best_M, 'k': best_k}

            ks_range = np.arange(1, K + 1, dtype=np.int64)
            Ms = ks_range * sigma[1:]
            
            unique_Ms, inverse, counts = np.unique(Ms, return_inverse=True, return_counts=True)
            max_count_idx = np.argmax(counts)
            max_count = counts[max_count_idx]
            
            if max_count > best_count:
                best_M = int(unique_Ms[max_count_idx])
                k_indices = np.where(Ms == best_M)[0]
                best_k = [int(x + 1) for x in k_indices]
                best_count = max_count
        except MemoryError:
            # Final fallback to pure Python small K (slower but safe)
            K = 2000000
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
