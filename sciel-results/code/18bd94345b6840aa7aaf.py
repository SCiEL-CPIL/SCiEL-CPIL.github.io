import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin from 150s limit
    
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3
    
    # Try increasing N with SPF-based sigma computation (O(N) per N)
    target_ns = [5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000]
    
    for N in target_ns:
        if time.time() > deadline:
            break
        
        try:
            # Compute sigma using SPF DP: O(N)
            # sigma[1] = 1
            # For k > 1, if k = p^a * m where p does not divide m,
            # sigma(k) = sigma(p^a) * sigma(m) = (p^{a+1}-1)/(p-1) * sigma(m)
            
            # First, compute smallest prime factor (SPF) for all numbers up to N
            spf = np.zeros(N + 1, dtype=np.int64)
            spf[1] = 1
            for i in range(2, N + 1):
                if spf[i] == 0:
                    # i is prime
                    spf[i] = i
                    if i * i <= N:
                        for j in range(i * i, N + 1, i):
                            if spf[j] == 0:
                                spf[j] = i
            
            # Now compute sigma using DP
            sigma = np.zeros(N + 1, dtype=np.int64)
            sigma[1] = 1
            
            # We need to track p^a part for each number
            # Let's use: for each k, factor using spf
            # But that's O(log k) per k, total O(N log N)
            
            # Better: use the recurrence
            # If k = p * m where p is smallest prime factor of k,
            # and let k = p^a * m where p does not divide m,
            # then sigma(k) = sigma(p^a) * sigma(m)
            # We can store:
            #   p_power[k] = p^a (the highest power of spf[k] dividing k)
            #   sigma_p_power[k] = sigma(p^a) = (p^{a+1}-1)/(p-1)
            #   rest[k] = m = k / p^a
            # Then sigma[k] = sigma_p_power[k] * sigma[rest[k]]
            
            p_power = np.zeros(N + 1, dtype=np.int64)
            sigma_p_power = np.zeros(N + 1, dtype=np.int64)
            rest = np.zeros(N + 1, dtype=np.int64)
            
            for k in range(2, N + 1):
                p = spf[k]
                if k % p == 0:
                    k_div_p = k // p
                    if k_div_p % p == 0:
                        # p divides k_div_p, so p_power[k] = p * p_power[k_div_p]
                        p_power[k] = p * p_power[k_div_p]
                        sigma_p_power[k] = sigma_p_power[k_div_p] + p_power[k]
                        rest[k] = rest[k_div_p]
                    else:
                        # p does not divide k_div_p
                        p_power[k] = p
                        sigma_p_power[k] = 1 + p
                        rest[k] = k_div_p
                else:
                    # This shouldn't happen since spf[k] = p divides k
                    p_power[k] = 1
                    sigma_p_power[k] = 1
                    rest[k] = k
                
                if rest[k] > 1:
                    sigma[k] = sigma_p_power[k] * sigma[rest[k]]
                else:
                    sigma[k] = sigma_p_power[k]
            
            # Compute M = k * sigma(k)
            k_vals = np.arange(1, N + 1, dtype=np.int64)
            M_vals = k_vals * sigma
            
            # Filter M <= 10^18
            mask = M_vals <= 10**18
            M_valid = M_vals[mask]
            k_valid = k_vals[mask]
            
            if len(M_valid) == 0:
                continue
            
            # Find most frequent M
            unique_M, counts, inverse = np.unique(M_valid, return_counts=True, return_inverse=True)
            
            if len(unique_M) == 0:
                continue
            
            max_idx = np.argmax(counts)
            current_count = int(counts[max_idx])
            current_M = int(unique_M[max_idx])
            
            if current_count > best_count:
                best_count = current_count
                best_M = current_M
                k_indices = np.where(inverse == max_idx)[0]
                best_k_list = [int(x) for x in k_valid[k_indices]]
                
                # Check time
                if time.time() > deadline:
                    break
                    
        except MemoryError:
            break
        except Exception:
            break
    
    # Ensure types are correct
    return {'M': int(best_M), 'k': best_k_list}
