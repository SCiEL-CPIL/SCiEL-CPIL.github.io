"""Combined program: NumPy sigma sieve + SPF-based extension + collision search."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin before 150s kill

    M_limit = 10**18

    try:
        import numpy as np
        HAVE_NUMPY = True
    except ImportError:
        HAVE_NUMPY = False

    # Phase 1: Compute sigma up to N using NumPy sieve (fast, O(N log N) but vectorized)
    N = 10_000_000

    if HAVE_NUMPY:
        sigma = np.zeros(N + 1, dtype=np.int64)
        for i in range(1, N + 1):
            sigma[i::i] += i
            if i % 2_000_000 == 0:
                if time.time() - start_time > DEADLINE - 40:
                    break
    else:
        # Pure Python fallback with smaller N
        N = 3_000_000
        sigma = [0] * (N + 1)
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sigma[j] += i
            if i % 500_000 == 0 and time.time() - start_time > DEADLINE - 40:
                break

    # Check deadline
    if time.time() - start_time > DEADLINE - 20:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Phase 2: Group by M = k * sigma(k), find M with most solutions
    groups = {}
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3

    if HAVE_NUMPY:
        k_arr = np.arange(1, N + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:]
        valid_mask = M_arr <= M_limit
        M_valid = M_arr[valid_mask]
        k_valid = k_arr[valid_mask]

        # Convert to lists for dict operations
        M_list = M_valid.tolist()
        k_list = k_valid.tolist()

        for M_val, k_val in zip(M_list, k_list):
            if M_val not in groups:
                groups[M_val] = []
            groups[M_val].append(k_val)
            if len(groups[M_val]) > best_count:
                best_count = len(groups[M_val])
                best_M = M_val
                best_k_list = groups[M_val][:]
    else:
        for k in range(1, N + 1):
            M_val = k * sigma[k]
            if M_val <= M_limit:
                if M_val not in groups:
                    groups[M_val] = []
                groups[M_val].append(k)
                if len(groups[M_val]) > best_count:
                    best_count = len(groups[M_val])
                    best_M = M_val
                    best_k_list = groups[M_val][:]

    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        best_k_list = sorted(set(best_k_list))
        limit = math.isqrt(best_M)
        filtered_k = [k for k in best_k_list if k <= limit]
        if len(filtered_k) >= 3:
            return {'M': int(best_M), 'k': [int(k) for k in filtered_k]}
        return {'M': 333312, 'k': [336, 372, 434]}

    # Phase 3: Try to extend search using SPF-based sigma for a range beyond N
    # This allows us to check k values up to N2 without recomputing the full sieve
    # We'll compute sigma for k in [N+1, N2] using multiplicative formula with SPF
    
    # Determine how far we can extend
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 15:
        # Extend to N2 = 15_000_000 if we have time
        N2 = min(15_000_000, int(N * 1.5))
        
        # Build SPF for the extended range
        # For efficiency, we only need SPF for numbers in (N, N2]
        # But building full SPF up to N2 is O(N2 log log N2) which is manageable
        
        if HAVE_NUMPY:
            # Use NumPy for SPF sieve
            spf = np.arange(N2 + 1, dtype=np.int64)
            spf[0] = 0
            spf[1] = 1
            sqrt_N2 = int(math.isqrt(N2))
            for i in range(2, sqrt_N2 + 1):
                if spf[i] == i:  # i is prime
                    start = i * i
                    spf[start::i] = np.where(spf[start::i] == spf[start::i], i, spf[start::i])
                    # More careful: only update if spf[j] == j (i.e., not yet assigned)
                    # The above np.where doesn't work correctly for this. Let's do it differently.
            
            # Actually, let's use a simpler approach: compute sigma for the extension range
            # by factoring each number. For k in (N, N2], factor k and compute sigma.
            # This is slower per number but we only need to do it for ~5M numbers.
            
            # Better: use the multiplicative property. For each k in (N, N2]:
            #   Find smallest prime factor p of k
            #   m = k // p
            #   If m % p == 0: sigma[k] = sigma[m] * (p^(a+1)-1)/(p-1) / (p^a-1)/(p-1) ... complex
            #   Else: sigma[k] = sigma[m] * (1+p)
            
            # Let's precompute SPF properly using a sieve
            spf = list(range(N2 + 1))
            for i in range(2, int(math.isqrt(N2)) + 1):
                if spf[i] == i:
                    for j in range(i * i, N2 + 1, i):
                        if spf[j] == j:
                            spf[j] = i
            
            # Now compute sigma for k in (N, N2]
            # We need sigma values for all numbers up to N2 to use the multiplicative formula
            # But we already have sigma up to N. For k > N, we need sigma of divisors which may be > N.
            # So we need to compute sigma for all numbers up to N2.
            
            # Let's just compute sigma for the full range [1, N2] using SPF
            sigma_full = [1] * (N2 + 1)
            sigma_full[0] = 0
            sigma_full[1] = 1
            
            for k in range(2, N2 + 1):
                p = spf[k]
                m = k // p
                if m % p == 0:
                    # Find the power of p in k
                    power = p
                    while m % p == 0:
                        m //= p
                        power *= p
                    sigma_full[k] = sigma_full[m] * (power * p - 1) // (p - 1)
                else:
                    sigma_full[k] = sigma_full[m] * (1 + p)
            
            # Now check for collisions in the extended range
            for k in range(N + 1, N2 + 1):
                M_val = k * sigma_full[k]
                if M_val <= M_limit:
                    if M_val not in groups:
                        groups[M_val] = []
                    groups[M_val].append(k)
                    if len(groups[M_val]) > best_count:
                        best_count = len(groups[M_val])
                        best_M = M_val
                        best_k_list = groups[M_val][:]
            
            # Check deadline
            if time.time() - start_time > DEADLINE - 5:
                pass  # Break out and return best found

    # Final verification and return
    best_k_list = sorted(set(best_k_list))
    limit = math.isqrt(best_M)
    filtered_k = [k for k in best_k_list if 1 <= k <= limit]
    
    if len(filtered_k) >= 3:
        # Verify each solution
        valid_k = []
        for k in filtered_k:
            # Recompute sigma(k) to verify
            if k <= N:
                if HAVE_NUMPY:
                    sig_k = int(sigma[k])
                else:
                    sig_k = sigma[k]
            else:
                # Need to compute sigma(k) for k > N
                # Use trial division
                n = k
                sig_k = 1
                temp = n
                d = 2
                while d * d <= temp:
                    if temp % d == 0:
                        power = 1
                        while temp % d == 0:
                            temp //= d
                            power *= d
                        sig_k *= (power * d - 1) // (d - 1)
                    d += 1
                if temp > 1:
                    sig_k *= (1 + temp)
            
            if k * sig_k == best_M:
                valid_k.append(k)
        
        if len(valid_k) >= 3:
            return {'M': int(best_M), 'k': [int(k) for k in valid_k]}
    
    # Fallback to seed
    return {'M': 333312, 'k': [336, 372, 434]}
# EVOLVE-BLOCK-END
