import time
import numpy as np
from math import isqrt
from collections import defaultdict

def _is_powerful(n):
    """Check if n is a powerful number using exact factorization."""
    if n == 1:
        return True
    temp = n
    f = 2
    while f * f <= temp:
        if temp % f == 0:
            count = 0
            while temp % f == 0:
                temp //= f
                count += 1
            if count == 1:
                return False
        f += 1
    if temp > 1:
        return False
    return True

def _generate_powerful_numbers_up_to(M):
    """
    Generate all powerful numbers up to M.
    A powerful number is of the form u^2 * v^3 where v is square-free.
    """
    if M < 1:
        return set()
    
    powerful = set()
    # v^3 <= M => v <= M^(1/3)
    max_v = int(M ** (1/3)) + 2
    
    # We need u^2 <= M / v^3 => u <= sqrt(M / v^3)
    for v in range(1, max_v + 1):
        v_cubed = v ** 3
        if v_cubed > M:
            break
        max_u = isqrt(M // v_cubed)
        for u in range(1, max_u + 1):
            val = u * u * v_cubed
            if val <= M:
                powerful.add(val)
    return powerful

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Strategy:
    # 1. Choose a powerful constant C.
    # 2. We want N = C * K.
    # 3. We want pairs (a, b) such that a + b = N, a = C * u, b = C * v.
    # 4. Condition: C*u is powerful AND C*v is powerful.
    # 5. If C is powerful, C*u is powerful iff u is powerful AND gcd(u, C/rad(C)) conditions?
    #    Actually, let C = prod p_i^{e_i}. Let u = prod p_i^{f_i} * w where w is coprime to C.
    #    C*u = prod p_i^{e_i + f_i} * w.
    #    For C*u to be powerful:
    #    - For p_i | C: e_i + f_i must be >= 2 (or 0, but f_i>=0 so >=2 usually holds if e_i>=2).
    #      If e_i >= 2, then e_i + f_i >= 2 is always true for f_i >= 0.
    #    - For p | w (p not in C): exponent in w must be >= 2.
    #    So if C is a powerful number where every prime exponent is >= 2, 
    #    then C*u is powerful if and only if u is powerful.
    #    Proof: Let C = prod p^{e_p} with e_p >= 2.
    #    Let u = prod p^{f_p} * w, where w is coprime to C.
    #    C*u = prod p^{e_p + f_p} * w.
    #    For p|C, exp = e_p + f_p >= 2.
    #    For p|w, exp = exp_in_w.
    #    C*u is powerful iff w is powerful.
    #    Since w is coprime to C, u is powerful iff w is powerful (and the p-part of u doesn't matter for the "powerful" check of the whole number if we just look at exponents? No, u being powerful requires f_p >= 2 for all p|u. But we don't need u to be powerful, we need C*u to be powerful.
    #    Wait, the condition is just that C*u is powerful.
    #    So we need to find u, v such that C*u and C*v are powerful.
    #    This is equivalent to finding u, v such that the "new" prime factors introduced by u (those not in C) appear with exponent >= 2, and for primes in C, the total exponent is >= 2 (which is guaranteed if C has exp >= 2).
    #    So u doesn't have to be powerful itself, but its part coprime to C must be powerful.
    
    # Let's pick C = 2^6 * 3^4 * 5^2 * 7^2 = 64 * 81 * 25 * 49
    # C = 64 * 81 = 5184
    # C = 5184 * 25 = 129600
    # C = 129600 * 49 = 6350400
    # C = 6,350,400
    
    C = 6350400
    
    # We want N <= 10^18.
    # Max K = 10^18 // C ~ 1.57 * 10^11.
    # We are looking for u + v = K.
    # Constraints on u, v:
    # Let u = u_c * u_p, where u_c is the part sharing primes with C, u_p is coprime to C.
    # u_p must be powerful.
    # u_c can be anything?
    # C*u = C * u_c * u_p.
    # Primes in C: exponents in C are >= 2. Adding exponents from u_c keeps them >= 2.
    # Primes in u_p: exponents must be >= 2.
    # Primes in u_c that are NOT in C? If u_c has a prime q not in C, then q is in u_p effectively.
    # So simply: Let u = u' * u'', where u'' is the part of u coprime to C.
    # u'' must be powerful.
    # u' can be any integer?
    # If u' has a prime factor q not in C, then q is part of u''.
    # So u' must only have prime factors from C.
    # Let P_C be the set of prime factors of C.
    # u = u_{P_C} * u_{coprime}
    # u_{coprime} must be powerful.
    # u_{P_C} can be any number whose prime factors are in P_C.
    
    # This structure is complex to enumerate directly for FFT because u_{P_C} can be large.
    # However, note that if we restrict u and v to be powerful numbers themselves, then C*u and C*v are definitely powerful.
    # Why?
    # If u is powerful, all prime exponents in u are >= 2.
    # C has exponents >= 2.
    # C*u has exponents >= 2 + 2 = 4 >= 2.
    # So the set of powerful numbers is a subset of valid u's.
    # Is it a dense subset?
    # The density of powerful numbers is ~ x^{1/2}.
    # The density of valid u's (where coprime part is powerful) is higher?
    # Actually, let's just use powerful numbers for u and v first. It's simpler and guaranteed correct.
    # If we use powerful numbers u, v, then a = C*u, b = C*v.
    # a+b = C(u+v) = N.
    # We need to find powerful u, v such that u+v = K, and C*K <= 10^18.
    # Max K = 10^18 // C.
    # We can generate powerful numbers up to Max_K.
    # Then use FFT to find K with many pairs (u, v) of powerful numbers.
    
    Max_K = 10**18 // C
    # Max_K is approx 1.57 * 10^11.
    # Generating powerful numbers up to 1.57 * 10^11:
    # Count is approx (1.57*10^11)^{1/2} * const ~ 400,000.
    # This is very feasible.
    
    # Let's refine Max_K to fit in memory for FFT.
    # We need FFT size ~ 2 * Max_K.
    # If Max_K is 10^10, size is 2*10^10. Too big for FFT (20 billion floats).
    # We need to reduce the search space for K.
    # We can search for K in a range [K_min, K_max].
    # But FFT requires the full range if we want to find any K.
    # Alternatively, we can just pick a specific K_max that fits in memory, e.g., K_max = 5 * 10^8.
    # Then N_max = C * K_max = 6.35 * 10^6 * 5 * 10^8 = 3 * 10^15.
    # This is well within 10^18.
    # The number of powerful numbers up to 5*10^8 is ~ sqrt(5*10^8) ~ 22,000.
    # FFT size for 5*10^8 is ~ 10^9. 1 billion floats = 8 GB. Might be too much.
    # Let's use K_max = 10^8.
    # FFT size 2*10^8 = 200 million. 1.6 GB. Feasible.
    # Powerful numbers up to 10^8: ~ 10,000.
    
    # Let's try multiple scales of K_max.
    target_Ks = [
        50_000_000,   # 50M
        20_000_000,   # 20M
        10_000_000,   # 10M
        5_000_000     # 5M
    ]
    
    for K_max in target_Ks:
        if time.time() > deadline - 20:
            break
            
        try:
            # Generate powerful numbers up to K_max
            powerful_set = _generate_powerful_numbers_up_to(K_max)
            powerful_list = sorted(powerful_set)
            K_count = len(powerful_list)
            
            if K_count < 2:
                continue
            
            # FFT setup
            size = 1
            while size < 2 * K_max + 1:
                size *= 2
                
            # Memory check: size * 8 bytes (float64)
            # For K_max=50M, size ~ 2^26 = 67M. 67M * 8 = 536 MB. OK.
            # For K_max=100M, size ~ 2^28 = 268M. 268M * 8 = 2.1 GB. Borderline.
            if size > 2**27:
                continue
                
            A = np.zeros(size, dtype=np.float64)
            for p in powerful_list:
                A[p] = 1.0
                
            F = np.fft.rfft(A)
            F_sq = F * F
            counts_float = np.fft.irfft(F_sq, n=size)
            
            # Search for hotspots in K
            # We want K such that N = C*K is large? Or just many pairs?
            # The score is the number of pairs.
            # Larger K tends to have more representations?
            # Let's check the top counts.
            
            # Focus on upper half of K range to get larger N?
            # N = C*K. Larger K -> Larger N.
            # But the count of pairs is what matters.
            # Let's just find the max count in the valid range.
            
            search_start = K_max // 2
            search_end = min(K_max, size - 1)
            
            if search_end <= search_start:
                continue
                
            counts_segment = counts_float[search_start:search_end]
            
            if counts_segment.size == 0:
                continue
            
            max_val = np.max(counts_segment)
            if max_val < 1:
                continue
            
            # Find top candidates
            top_k = min(100, counts_segment.size)
            top_local_indices = np.argpartition(counts_segment, -top_k)[-top_k:]
            top_indices = top_local_indices + search_start
            
            # Verify top candidates
            # For each candidate K, N = C*K.
            # Pairs are (C*u, C*v) where u+v=K and u,v in powerful_set.
            # Note: u <= v implies C*u <= C*v.
            
            for idx in top_indices:
                if time.time() > deadline - 10:
                    break
                
                K = int(idx)
                N = C * K
                
                if N > 10**18:
                    continue
                    
                half_K = K // 2
                pairs = []
                count = 0
                
                for u in powerful_list:
                    if u > half_K:
                        break
                    v = K - u
                    if v in powerful_set:
                        # Check if C*u and C*v are powerful?
                        # Since u and v are powerful, and C is powerful with exps>=2,
                        # C*u and C*v are powerful.
                        a = C * u
                        b = C * v
                        pairs.append([a, b])
                        count += 1
                        
                if count > best_count:
                    best_count = count
                    best_N = N
                    best_pairs = pairs
                    
        except MemoryError:
            continue
        except Exception as e:
            #print(f"Error: {e}")
            continue
            
    # Fallback if nothing found
    if best_count <= 1:
        # Try small N manually
        # N = C * 1? No, 1 is powerful.
        # Pairs for K=1: u=1, v=0? 0 is not positive.
        # u+v=1. u=1, v=0 (invalid).
        # Smallest K with 2 powerful numbers: 1+1=2.
        # K=2 -> N=2C.
        # u=1, v=1. a=C, b=C.
        # C is powerful.
        # So N=2C is a valid candidate with 1 pair.
        
        small_Ks = [2, 4, 5, 8, 9, 10, 12, 16, 18, 20, 25, 27, 32, 36, 45, 48, 50, 64, 81, 98, 100]
        for K in small_Ks:
            if time.time() > deadline - 2:
                break
            N = C * K
            if N > 10**18:
                continue
            half_K = K // 2
            p_set = _generate_powerful_numbers_up_to(K)
            pairs = []
            for u in sorted(p_set):
                if u > half_K:
                    break
                v = K - u
                if v in p_set:
                    a = C * u
                    b = C * v
                    pairs.append([a, b])
            if len(pairs) > best_count:
                best_count = len(pairs)
                best_N = N
                best_pairs = pairs

    return {'N': best_N, 'pairs': best_pairs}
