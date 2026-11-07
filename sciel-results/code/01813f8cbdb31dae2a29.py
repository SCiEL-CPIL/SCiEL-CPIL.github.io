   import numpy as np
   from math import isqrt

   def construct():
       N = 10000000
       sigma = np.zeros(N + 1, dtype=np.int64)
       
       for i in range(1, N + 1):
           sigma[i::i] += i
           
       k_arr = np.arange(1, N + 1, dtype=np.int64)
       M_arr = k_arr * sigma[1:N+1]
       
       mask = M_arr <= 10**18
       M_valid = M_arr[mask]
       k_valid = k_arr[mask]
       
       order = np.argsort(M_valid)
       M_sorted = M_valid[order]
       k_sorted = k_valid[order]
       
       unique_M, counts, start_indices = np.unique(M_sorted, return_counts=True, return_index=True)
       
       best_M = 1
       best_k = [1]
       best_count = 1
       
       for i in range(len(unique_M)):
           M_val = int(unique_M[i])
           start = start_indices[i]
           count = counts[i]
           end = start + count
           
           sqrt_M = isqrt(M_val)
           
           # Extract k's
           # k_sorted[start:end]
           # Check k <= sqrt_M
           # Since count is usually small, simple loop is fine
           # But we can use numpy boolean mask on the slice
           ks_slice = k_sorted[start:end]
           valid_mask = ks_slice <= sqrt_M
           valid_ks = ks_slice[valid_mask]
           
           if len(valid_ks) > best_count:
               best_count = len(valid_ks)
               best_M = M_val
               best_k = sorted([int(x) for x in valid_ks])
               
       return {'M': best_M, 'k': best_k}
   