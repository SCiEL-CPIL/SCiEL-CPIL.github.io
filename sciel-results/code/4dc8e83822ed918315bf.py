import math
from collections import defaultdict

def construct():
    # Primes to use
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    m_to_k = defaultdict(list)
    
    # Recursive generation
    # State: current_index, current_k, current_sigma, current_M
    # We need to prune if current_M > 10^18
    
    def gen(idx, k, sigma_k, M_val):
        if M_val > 10**18:
            return
        
        if idx == len(primes):
            if M_val > 1: # k > 1
                m_to_k[M_val].append(k)
            return
        
        p = primes[idx]
        # Try exponents e = 0, 1, ...
        # We need to know sigma(p^e) and p^e
        # Start with e=0: p^0=1, sigma(1)=1
        # Then e=1: p^1=p, sigma(p)=1+p
        # e=2: p^2, sigma(p^2)=1+p+p^2
        
        pe = 1
        sigma_pe = 1
        
        while True:
            new_k = k * pe
            new_sigma = sigma_k * sigma_pe
            new_M = new_k * new_sigma
            
            if new_M > 10**18:
                break
            
            # If we include this prime power, recurse to next prime
            # Note: we must allow e=0 (not using this prime)
            # The loop handles e=0 in the first iteration?
            # No, the loop starts with e=0.
            # If e=0, pe=1, sigma_pe=1.
            # Then we recurse.
            gen(idx + 1, new_k, new_sigma, new_M)
            
            # Update for next exponent
            # Check if we can go higher
            # p^(e+1) might overflow? No, we check M_val.
            # But we need to calculate next pe and sigma_pe.
            # pe_next = pe * p
            # sigma_pe_next = sigma_pe + pe_next
            
            # Optimization: if p is large, we might only have e=0,1.
            # We break when new_M > 10^18.
            
            # Prepare for next e
            # We need to be careful with the loop condition.
            # Let's just increment e and check.
            
            # Actually, the `while True` with `break` on M_val is fine.
            # But we need to update pe and sigma_pe at the end of the loop.
            
            # However, if we break, we don't update.
            # So we should update at the end, but check limit before recursing?
            # No, we check limit, then recurse, then update.
            
            # Wait, if I update pe and sigma_pe, I need to make sure I don't overflow Python ints (they don't) 
            # or take too long.
            
            # Let's restructure slightly for clarity and speed.
            # The recursion depth is 10.
            
            # Update for next exponent
            pe *= p
            sigma_pe += pe
            
            # Safety check for infinite loop if p=1? No, p>=2.
            # The M_val check will break it quickly.
            
    gen(0, 1, 1, 1)
    
    # Find best M
    best_M = 1
    best_ks = []
    for M, ks in m_to_k.items():
        # Filter k <= isqrt(M) (should be all, but just in case)
        sqrt_M = math.isqrt(M)
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > len(best_ks):
            best_M = M
            best_ks = valid_ks
            
    return {'M': best_M, 'k': best_ks}
