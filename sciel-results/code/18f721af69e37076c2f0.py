    def _compute_sigma_linear(limit):
        sigma = [0] * (limit + 1)
        sigma[1] = 1
        primes = []
        is_prime = [True] * (limit + 1)
        # sp[i] will be the highest power of the smallest prime factor of i
        # e.g., if i = 12 = 2^2 * 3, sp[12] = 4
        sp = [0] * (limit + 1)
        sp[1] = 1
        
        for i in range(2, limit + 1):
            if is_prime[i]:
                primes.append(i)
                sigma[i] = i + 1
                sp[i] = i
            for p in primes:
                if i * p > limit:
                    break
                is_prime[i * p] = False
                if i % p == 0:
                    # p is the smallest prime factor of i
                    # sp[i] is the highest power of p dividing i
                    # sigma[i * p] = sigma[i] + p * sp[i] * (sigma[i] // (sp[i] + ... ))? 
                    # No.
                    # Let i = p^k * m. sp[i] = p^k.
                    # sigma[i] = sigma(m) * (1 + ... + p^k)
                    # i * p = p^(k+1) * m
                    # sigma[i * p] = sigma(m) * (1 + ... + p^(k+1))
                    # sigma[i * p] = sigma[i] + sigma(m) * p^(k+1)
                    # We know sigma[m] = sigma[i] / (1 + ... + p^k)
                    # So sigma[i * p] = sigma[i] + (sigma[i] / (1 + ... + p^k)) * p^(k+1)
                    # This involves division which might be slow or inexact if not careful, but sigma is integer.
                    # Actually, there is a simpler recurrence:
                    # sigma[i * p] = sigma[i] + p * sp[i] * (sigma[i] // (sp[i] * p - 1) * (p - 1))? No.
                    
                    # Let's use the identity:
                    # sigma(n) = sigma(n/p) + sigma(n/p^k) * p^k
                    # where n = p^k * m.
                    # Here n = i * p.
                    # If i % p == 0, let i = p^k * m.
                    # Then i * p = p^(k+1) * m.
                    # sigma(i * p) = sigma(i) + sigma(m) * p^(k+1)
                    # We have sp[i] = p^k.
                    # sigma(m) = sigma(i) / (1 + p + ... + p^k)
                    # This requires calculating the sum of geometric series.
                    
                    # Better approach for linear sieve sigma:
                    # Keep `sigma` and `sp` (highest power of smallest prime).
                    # If i % p == 0:
                    #   sp[i*p] = sp[i] * p
                    #   sigma[i*p] = sigma[i] + sp[i*p] * (sigma[i] // (sp[i] * p - 1) * (p - 1))? No.
                    
                    # Let's use the multiplicative property directly with spf.
                    # Since we are doing a linear sieve, we can just calculate sigma[i*p] from sigma[i] and sp[i].
                    # If i % p == 0:
                    #   sp[i*p] = sp[i] * p
                    #   # sigma(i) = sigma(m) * S_k, where S_k = 1+...+p^k
                    #   # sigma(i*p) = sigma(m) * S_{k+1}
                    #   # S_{k+1} = S_k + p^(k+1)
                    #   # sigma(i*p) = sigma(m) * (S_k + p^(k+1)) = sigma(i) + sigma(m) * p^(k+1)
                    #   # sigma(m) = sigma(i) / S_k
                    #   # This is tricky.
                    
                    # Let's switch to a simpler O(N log N) sieve if linear is too complex to get right quickly.
                    # N=10^7. O(N log N) might be ~ 10^7 * 16 = 1.6 * 10^8 operations. In Python, this is too slow (approx 10-20s+).
                    # We need the linear sieve.
                    
                    # Correct Linear Sieve for Sigma:
                    # sigma[1] = 1
                    # For i from 2 to N:
                    #   If i is prime:
                    #     sigma[i] = i + 1
                    #     sp[i] = i
                    #   For each prime p in primes:
                    #     if i * p > N: break
                    #     if i % p == 0:
                    #       sp[i * p] = sp[i] * p
                    #       sigma[i * p] = sigma[i] + sp[i * p] * (sigma[i] // (sp[i] * p - 1) * (p - 1))?
                    #       Let's derive:
                    #       i = p^k * m. sp[i] = p^k.
                    #       sigma[i] = sigma(m) * (p^(k+1)-1)/(p-1)
                    #       i*p = p^(k+1) * m. sp[i*p] = p^(k+1).
                    #       sigma[i*p] = sigma(m) * (p^(k+2)-1)/(p-1)
                    #       sigma[i*p] = sigma[i] + sigma(m) * p^(k+1)
                    #       sigma(m) = sigma[i] * (p-1) / (p^(k+1)-1)
                    #       So sigma[i*p] = sigma[i] + sigma[i] * (p-1) / (p^(k+1)-1) * p^(k+1)
                    #       This involves division.
                    
                    # Alternative:
                    # sigma[i*p] = sigma[i] + p * sp[i] * (sigma[i] // (sp[i] * p - 1) * (p - 1))?
                    # Let's check for i=4, p=2.
                    # i=4, sp[4]=4, sigma[4]=7.
                    # i*p=8. sp[8]=8.
                    # sigma[8] should be 15.
                    # Formula: sigma[8] = sigma[4] + p * sp[4] * (sigma[4] // (sp[4] * p - 1) * (p - 1))?
                    # sp[4]*p - 1 = 7. 7 // 7 = 1. 1 * (2-1) = 1.
                    # sigma[8] = 7 + 2 * 4 * 1 = 15. Correct.
                    
                    # Let's check i=2, p=2.
                    # i=2, sp[2]=2, sigma[2]=3.
                    # i*p=4. sp[4]=4.
                    # sigma[4] = sigma[2] + 2 * 2 * (sigma[2] // (2*2-1) * (2-1))
                    # = 3 + 4 * (3 // 3 * 1) = 3 + 4 = 7. Correct.
                    
                    # So the formula is:
                    # if i % p == 0:
                    #   sp[i*p] = sp[i] * p
                    #   sigma[i*p] = sigma[i] + p * sp[i] * (sigma[i] // (sp[i] * p - 1) * (p - 1))
                    #   Wait, the term is sigma(m) * p^(k+1).
                    #   sigma(m) = sigma[i] * (p-1) / (p^(k+1)-1)
                    #   p^(k+1) = sp[i] * p
                    #   So term = sigma[i] * (p-1) / (sp[i]*p - 1) * (sp[i]*p)
                    #   My previous check:
                    #   i=4, p=2. sp[i]=4. sp[i]*p = 8.
                    #   term = 7 * 1 / 7 * 8 = 8.
                    #   sigma[8] = 7 + 8 = 15.
                    #   The formula I wrote in the check was: `p * sp[i] * (sigma[i] // (sp[i] * p - 1) * (p - 1))`
                    #   This equals `2 * 4 * (7 // 7 * 1)` = `8 * 1` = 8.
                    #   Yes, it works because `sigma[i]` is divisible by `sp[i]*p - 1`?
                    #   `sigma[i] = sigma(m) * (p^(k+1)-1)/(p-1)`.
                    #   So `sigma[i] * (p-1) = sigma(m) * (p^(k+1)-1)`.
                    #   So `sigma[i] * (p-1)` is divisible by `p^(k+1)-1`.
                    #   So `sigma[i] // (sp[i]*p - 1)` is NOT necessarily integer.
                    #   `sigma[i] / (sp[i]*p - 1)` is `sigma(m) / (p-1)`.
                    #   So we need `(sigma[i] * (p-1)) // (sp[i]*p - 1)`.
                    
                    # Correct formula:
                    # term = (sigma[i] * (p - 1)) // (sp[i] * p - 1) * (sp[i] * p)
                    # sigma[i*p] = sigma[i] + term
                    
                    # Let's re-verify i=4, p=2.
                    # sigma[4]=7. p=2. sp[4]=4.
                    # term = (7 * 1) // (4*2 - 1) * (4*2) = 7 // 7 * 8 = 1 * 8 = 8.
                    # sigma[8] = 7 + 8 = 15. Correct.
                    
                    # Let's re-verify i=2, p=2.
                    # sigma[2]=3. p=2. sp[2]=2.
                    # term = (3 * 1) // (2*2 - 1) * (2*2) = 3 // 3 * 4 = 1 * 4 = 4.
                    # sigma[4] = 3 + 4 = 7. Correct.
                    
                    # If i % p != 0:
                    #   sp[i*p] = p
                    #   sigma[i*p] = sigma[i] * (p + 1)
                    
                    break
            