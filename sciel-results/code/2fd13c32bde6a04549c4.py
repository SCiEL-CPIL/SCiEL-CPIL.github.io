"""Search for M with many distinct k where k*sigma(k)=M, k<=isqrt(M)."""
import math
import time
import sys
from collections import defaultdict
import numpy as np

def _sigma_exact(n):
    """Compute sigma(n) exactly using trial division."""
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            power_sum = 1
            pk = 1
            while temp % p == 0:
                temp //= p
                pk *= p
                power_sum += pk
            result *= power_sum
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    def try_sieve(K):
        nonlocal best_M, best_k, best_count
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            return False
        
        # 1. Compute Sigma using NumPy (int64 safe for K up to ~10^7)
        sigma = np.zeros(K + 1, dtype=np.int64)
        for d in range(1, K + 1):
            sigma[d::d] += d
        
        if time.time() - start_time > deadline - 20:
            return False

        # 2. Compute M values as float64 for grouping (exact for < 2^53)
        ks_range = np.arange(1, K + 1, dtype=np.float64)
        Ms_float = ks_range * sigma[1:].astype(np.float64)
        
        if time.time() - start_time > deadline - 15:
            return False

        # 3. Group by M using np.unique on float64 (exact for integers < 2^53)
        unique_Ms, inverse, counts = np.unique(Ms_float, return_inverse=True, return_counts=True)
        
        # Find top candidates by count
        max_count_idx = np.argmax(counts)
        max_count = int(counts[max_count_idx])
        
        if max_count > best_count:
            candidate_M_float = unique_Ms[max_count_idx]
            candidate_M = int(candidate_M_float)
            
            # Verify exactly: find all k where k*sigma(k) == candidate_M
            k_indices = np.where(inverse == max_count_idx)[0]
            k_list = [int(x + 1) for x in k_indices]
            
            # Verify each k exactly
            verified_k = []
            for k in k_list:
                if k * _sigma_exact(k) == candidate_M:
                    verified_k.append(k)
            
            if len(verified_k) > best_count:
                best_M = candidate_M
                best_k = verified_k
                best_count = len(verified_k)
        
        # Also check second and third most common in case top has overflow issues
        if len(counts) >= 3:
            top3 = np.argsort(counts)[::-1][:3]
            for idx in top3[1:]:
                cnt = int(counts[idx])
                if cnt <= best_count:
                    break
                candidate_M_float = unique_Ms[idx]
                candidate_M = int(candidate_M_float)
                k_indices = np.where(inverse == idx)[0]
                k_list = [int(x + 1) for x in k_indices]
                verified_k = []
                for k in k_list:
                    if k * _sigma_exact(k) == candidate_M:
                        verified_k.append(k)
                if len(verified_k) > best_count:
                    best_M = candidate_M
                    best_k = verified_k
                    best_count = len(verified_k)
        
        # Free memory
        del sigma, Ms_float, unique_Ms, inverse, counts, ks_range
        return True
    
    # Try increasing K values, stop if time is running out
    for K in [5000000, 10000000, 15000000, 20000000]:
        if time.time() - start_time > deadline - 40:
            break
        try:
            ok = try_sieve(K)
            if not ok:
                break
        except MemoryError:
            continue
        except Exception:
            break
    
    # Final safety: ensure types are plain Python int and list
    best_k = [int(x) for x in best_k]
    best_M = int(best_M)
    
    return {'M': best_M, 'k': best_k}
