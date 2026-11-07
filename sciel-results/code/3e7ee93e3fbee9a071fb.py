import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer for the 120s limit
    
    # Strategy: Construct k as 2^a * 3^b * 5^c
    # sigma(k) = sigma(2^a) * sigma(3^b) * sigma(5^c)
    # sigma(p^e) = (p^(e+1)-1)/(p-1)
    
    # We want to find M with many k such that k*sigma(k) = M.
    # This is equivalent to finding collisions in the function f(a,b,c) = (2^a 3^b 5^c) * sigma(2^a) * sigma(3^b) * sigma(5^c).
    
    # Let's determine the range of exponents.
    # k <= 10^9 is a good target to keep M <= 10^18 (since sigma(k) is typically k log log k, so M ~ k^2 log^2 k, 
    # if k=10^9, M ~ 10^18 * (20)^2 which is too big. 
    # Actually sigma(n) < 2n for n>1? No, sigma(n) can be larger. 
    # For highly composite, sigma(n) ~ n log log n.
    # If k ~ 10^7, sigma(k) ~ 10^7 * 3 = 3*10^7. M ~ 3*10^14. This is safe.
    # If k ~ 10^8, sigma(k) ~ 10^8 * 4 = 4*10^8. M ~ 4*10^16. Safe.
    # If k ~ 10^9, sigma(k) ~ 10^9 * 5 = 5*10^9. M ~ 5*10^18. Too big.
    # So we should restrict k to be roughly <= 5*10^8 to be safe.
    
    # Let's iterate a, b, c such that 2^a 3^b 5^c <= 5*10^8.
    
    # Precompute sigma powers
    # sigma(2^a) = 2^(a+1) - 1
    # sigma(3^b) = (3^(b+1) - 1) / 2
    # sigma(5^c) = (5^(c+1) - 1) / 4
    
    # Max exponents:
    # 2^a <= 5*10^8 => a <= 28
    # 3^b <= 5*10^8 => b <= 18
    # 5^c <= 5*10^8 => c <= 12
    
    # We will iterate a from 0 to 28
    # For each a, iterate b from 0 to 18
    # For each a, b, iterate c from 0 to 12
    # Check if k = 2^a 3^b 5^c is valid (<= limit)
    
    # To avoid duplicates and speed up, we can precompute terms.
    
    # Let's use a dictionary to store collisions.
    # Key: M = k * sigma(k)
    # Value: list of (a, b, c) or just k. We need k.
    
    # Optimization: 
    # Precompute arrays for 2^a, sigma(2^a)
    # Precompute arrays for 3^b, sigma(3^b)
    # Precompute arrays for 5^c, sigma(5^c)
    
    max_k_val = 500_000_000  # 5 * 10^8
    
    pow2 = [1] * 30
    sig2 = [1] * 30
    for i in range(1, 30):
        pow2[i] = pow2[i-1] * 2
        sig2[i] = sig2[i-1] * 2 + 1  # sigma(2^i) = 2^(i+1)-1
        
    pow3 = [1] * 20
    sig3 = [1] * 20
    for i in range(1, 20):
        pow3[i] = pow3[i-1] * 3
        sig3[i] = (pow3[i] * 3 - 1) // 2
        
    pow5 = [1] * 15
    sig5 = [1] * 15
    for i in range(1, 15):
        pow5[i] = pow5[i-1] * 5
        sig5[i] = (pow5[i] * 5 - 1) // 4
        
    # Now iterate
    collisions = {}
    
    # To improve performance, we can iterate a, then b, then c.
    # We can prune if 2^a * 3^b * 5^c > max_k_val.
    
    for a in range(30):
        if pow2[a] > max_k_val:
            break
        for b in range(20):
            if pow2[a] * pow3[b] > max_k_val:
                break
            for c in range(15):
                k_part = pow2[a] * pow3[b] * pow5[c]
                if k_part > max_k_val:
                    break
                
                # k = k_part
                # sigma_k = sig2[a] * sig3[b] * sig5[c]
                sigma_k = sig2[a] * sig3[b] * sig5[c]
                M = k_part * sigma_k
                
                if M > 10**18:
                    # If M is too big, larger c will only make it bigger.
                    break
                
                # Check if k <= sqrt(M)
                # k^2 <= M is k^2 <= k*sigma(k) => k <= sigma(k).
                # For k > 1, sigma(k) >= k+1 > k. So this is always true for k > 1.
                # For k=1, sigma(1)=1, M=1. 1<=1. True.
                # So the condition k <= sqrt(M) is automatically satisfied for all k >= 1.
                
                if M in collisions:
                    collisions[M].append(k_part)
                else:
                    collisions[M] = [k_part]
                    
                # Time check
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    # Find the best M
    best_M = None
    best_k = []
    
    for M_val, ks in collisions.items():
        if len(ks) > len(best_k):
            # Verify distinctness (they are distinct k values from different (a,b,c) combos? 
            # Wait, (a,b,c) uniquely determines k. So ks are distinct.)
            best_k = ks
            best_M = M_val
            
    # Fallback if we didn't find anything better than 3
    # Known collision: 333312 with k=[336, 372, 434]
    # 336 = 2^4 * 3 * 7. Not of form 2^a 3^b 5^c.
    # So my grid might not find this specific one.
    # But it might find others.
    
    if best_M is None or len(best_k) < 3:
        # Use the known good one
        best_M = 333312
        best_k = [336, 372, 434]
    else:
        # Ensure types
        best_M = int(best_M)
        best_k = [int(k) for k in best_k]
        
    # Final sanity check
    # M <= 10^18
    if best_M > 10**18:
        best_M = 333312
        best_k = [336, 372, 434]
        
    # Check k <= isqrt(M)
    sqrt_M = math.isqrt(best_M)
    valid_k = [k for k in best_k if k <= sqrt_M]
    
    if len(valid_k) < len(best_k):
        # If some were invalid, filter them
        best_k = valid_k
        
    if len(best_k) == 0:
        # Should not happen
        best_M = 333312
        best_k = [336, 372, 434]

    return {'M': best_M, 'k': best_k}

# EVOLVE-BLOCK-END
