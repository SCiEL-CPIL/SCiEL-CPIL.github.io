import time
import math
import numpy as np

def _compute_lpf_sieve(upper):
    """Compute largest prime factor for all integers in [0, upper] using a sieve."""
    lpf = np.zeros(upper + 1, dtype=np.int32)
    for i in range(2, upper + 1):
        if lpf[i] == 0:
            lpf[i::i] = i
    return lpf

def _lpf_trial(n):
    """Compute largest prime factor of n by trial division (for verification)."""
    if n < 2:
        return 1
    result = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            result = d
            n //= d
        d += 1
    if n > 1:
        result = n
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin before 90s kill
    upper = 1000001
    lower = 10000

    # Precompute largest prime factor for all numbers up to upper
    lpf = _compute_lpf_sieve(upper)
    
    # Compute max(lpf[n], lpf[n+1]) for n in [lower, 1000000]
    lpf_n = lpf[lower:upper]
    lpf_np1 = lpf[lower+1:upper+1]
    
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the candidate with minimum max LPF
    min_idx = int(np.argmin(max_lpf))
    best_n = lower + min_idx
    best_val = int(max_lpf[min_idx])
    
    # Verify the result with trial division and check for ties
    # that trial division might resolve differently
    if time.time() < deadline - 5:
        # Verify best candidate
        v_n = _lpf_trial(best_n)
        v_np1 = _lpf_trial(best_n + 1)
        verified_val = max(v_n, v_np1)
        
        # If there's a discrepancy, the sieve might have an issue
        if verified_val != best_val:
            # Re-scan with trial division in a wider window
            window = 500
            start = max(lower, best_n - window)
            end = min(1000000, best_n + window)
            
            local_best_n = best_n
            local_best_val = verified_val
            
            for n in range(start, end + 1):
                if time.time() > deadline - 5:
                    break
                v = max(_lpf_trial(n), _lpf_trial(n + 1))
                if v < local_best_val:
                    local_best_val = v
                    local_best_n = n
            
            best_n = local_best_n
            best_val = local_best_val
        else:
            # Check for ties - find all n with the same best_val
            # and pick the smallest n (or verify if a different n is truly better)
            tie_mask = max_lpf == best_val
            tie_indices = np.where(tie_mask)[0]
            if len(tie_indices) > 0:
                # The argmin already picks the first occurrence, which is the smallest n
                # Verify the first few ties to ensure correctness
                for idx in tie_indices[:5]:
                    n_check = lower + int(idx)
                    v_check = max(_lpf_trial(n_check), _lpf_trial(n_check + 1))
                    if v_check < best_val:
                        best_n = n_check
                        best_val = v_check
                        break

    return int(best_n)
