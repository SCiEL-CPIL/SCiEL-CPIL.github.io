import math
import time

# Generate primes
def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_up_to(100)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    max_n = 10**18
    best_tau = 0
    best_n = 1
    best_exps = []
    
    # DFS to find number with max divisors
    # Exponents must be non-increasing
    def dfs(idx, current_n, current_tau, max_exp):
        nonlocal best_tau, best_n, best_exps
        
        if time.time() > deadline:
            return
            
        if current_tau > best_tau:
            best_tau = current_tau
            best_n = current_n
            best_exps = [0] * len(_PRIMES)
            # Reconstruct exponents is tricky in recursive closure without passing list
            # We'll just store the n and recalculate or store path
            
        # Try increasing exponent for prime _PRIMES[idx]
        if idx >= len(_PRIMES):
            return
            
        p = _PRIMES[idx]
        
        # e can be 0 to max_exp
        # But we must have e <= max_exp
        # And current_n * p^e <= max_n
        
        # Optimization: if current_n * p > max_n, we can't even take e=1
        # But we can stop if p is too large?
        
        # We iterate e from 1 to max_exp
        # Note: We can also choose e=0, which means skipping this prime?
        # No, for HCN, we use consecutive primes. If we skip a prime, we can use a larger one,
        # but larger prime gives smaller n for same exponent, so we can fit more exponents?
        # Actually, HCNs always use the smallest primes.
        # So we don't skip primes. We just stop when we can't add more.
        
        # However, if we set e=0 for prime i, we can't use prime i+1 with e>0?
        # The condition is e_i >= e_{i+1}. If e_i=0, then e_{i+1}=0.
        # So we effectively stop.
        
        # So we just try e from 1 to max_exp.
        # If current_n * p > max_n, break.
        
        # We also need to consider the case where we don't use this prime at all?
        # That's equivalent to stopping the recursion.
        # The base case (or current state) already records the score.
        # So we just recurse for e >= 1.
        
        for e in range(1, max_exp + 1):
            if current_n * p > max_n:
                break
                
            new_n = current_n * p
            new_tau = current_tau * (e + 1) # Wait, current_tau is product of (prev_e + 1).
            # This logic is slightly flawed if we don't track exponents separately.
            # current_tau passed in should be the tau of the number formed by primes 0..idx-1.
            # Then new_tau = current_tau * (e + 1).
            
            dfs(idx + 1, new_n, new_tau, e)
            
            # Check deadline
            if time.time() > deadline:
                return

    # Initial call:
    # idx=0, n=1, tau=1, max_exp=60 (since 2^60 > 1e18)
    dfs(0, 1, 1, 60)
    
    return best_n
