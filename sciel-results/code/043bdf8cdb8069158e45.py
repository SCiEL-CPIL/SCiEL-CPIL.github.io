import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    def check_time():
        return time.time() > deadline

    # Linear Sieve for sigma(n)
    # We want to compute sigma(k) for k up to L
    # sigma(n) is multiplicative.
    # For n = prod p_i^e_i, sigma(n) = prod (p_i^(e_i+1)-1)/(p_i-1)
    
    # To handle the recurrence efficiently in a linear sieve:
    # Let n = p^e * m where p is the smallest prime factor of n, and gcd(p, m) = 1.
    # sigma(n) = sigma(p^e) * sigma(m)
    
    # We maintain:
    # lp[i]: smallest prime factor of i
    # pe[i]: p^e (the part of i consisting of powers of lp[i])
    # sig_pe[i]: sigma(pe[i])
    # rest[i]: m (the part of i coprime to lp[i])
    # sigma[i]: sigma(i)
    
    L = 10**7
    
    lp = [0] * (L + 1)
    pe = [1] * (L + 1)
    sig_pe = [1] * (L + 1)
    rest = [1] * (L + 1)
    sigma = [0] * (L + 1)
    
    primes = []
    sigma[1] = 1
    sig_pe[1] = 1
    rest[1] = 1
    pe[1] = 1
    
    m_counts = defaultdict(int)
    m_examples = defaultdict(list)
    best_m = 0
    best_count = 0
    
    for i in range(2, L + 1):
        if lp[i] == 0:
            lp[i] = i
            primes.append(i)
            pe[i] = i
            sig_pe[i] = 1 + i
            rest[i] = 1
            sigma[i] = 1 + i
        
        for p in primes:
            if p > lp[i] or i * p > L:
                break
            
            lp[i * p] = p
            
            if i % p == 0:
                # p is the smallest prime factor of i
                pe[i * p] = pe[i] * p
                sig_pe[i * p] = sig_pe[i] + pe[i * p]
                rest[i * p] = rest[i]
                sigma[i * p] = sig_pe[i * p] * sigma[rest[i * p]]
                break
            else:
                # p < lp[i], so p does not divide i
                pe[i * p] = p
                sig_pe[i * p] = 1 + p
                rest[i * p] = i
                sigma[i * p] = sig_pe[i * p] * sigma[i]
        
        # Check M = i * sigma(i)
        if i * sigma[i] <= 10**18:
            M = i * sigma[i]
            m_counts[M] += 1
            if len(m_examples[M]) < 100: # Limit list size to avoid memory blowup if one M has many
                m_examples[M].append(i)
            
            if m_counts[M] > best_count:
                best_count = m_counts[M]
                best_m = M
                
        if check_time():
            break

    # If we broke early, we might have missed some, but we have a candidate.
    # We should verify if there are any other M with same count but we just pick the first max found.
    # Actually, the loop updates best_m whenever a strictly higher count is found.
    # If multiple M have the same max count, we keep the first one found.
    
    if best_m > 0 and best_count > 0:
        # Ensure we return a list of ints
        k_list = [int(k) for k in m_examples[best_m]]
        # The problem requires distinct k. The sieve generates each i exactly once, so they are distinct.
        # The problem requires k <= sqrt(M).
        # As derived before, if k*sigma(k)=M, then k < sqrt(M) is automatically satisfied for k > 1.
        # For k=1, M=2, 1 <= sqrt(2) is true.
        # So we just return the list.
        
        # Double check types
        M_out = int(best_m)
        k_out = list(k_list)
        
        # Final sanity check: verify the first few solutions to be safe?
        # The sieve is exact, so no need.
        
        return {'M': M_out, 'k': k_out}
    
    # Fallback if sieve didn't find anything (shouldn't happen)
    return {'M': 333312, 'k': [336, 372, 434]}

if __name__ == "__main__":
    result = construct()
    print(result)
