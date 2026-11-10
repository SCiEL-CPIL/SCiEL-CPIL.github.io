import math
from collections import defaultdict

def construct():
    # We search for M <= 10^18 with many k such that k*sigma(k) = M.
    # We generate k from a set of small primes.
    
    LIMIT_M = 10**18
    
    # Primes to include in the factorization of k
    # Including more primes increases the chance of collisions but increases search space.
    # We stick to a moderate set to keep runtime low.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # To speed up sigma calculation, we precompute sigma(p^a) for small a.
    # But we can compute it on the fly.
    
    solutions = defaultdict(list)
    
    def sigma_prime_power(p, a):
        # Sum of divisors of p^a is (p^(a+1) - 1) / (p - 1)
        if a == 0:
            return 1
        # Using integer arithmetic to avoid floating point errors
        # (p^(a+1) - 1) is divisible by (p-1)
        num = pow(p, a + 1) - 1
        den = p - 1
        return num // den

    def dfs(idx, current_k, current_sigma, current_M_part):
        # current_M_part is the product of (p^a * sigma(p^a)) for processed primes?
        # No, M = k * sigma(k).
        # Let's track k and sigma(k) separately.
        # M = k * sigma(k).
        
        # Pruning: if current_k * current_sigma > LIMIT_M, stop.
        # Also, we only add to solutions if we have processed all primes?
        # No, k can have any subset of primes.
        # We should record the solution at every step where we decide to stop adding more primes?
        # Actually, k is defined by its prime factorization.
        # If we stop at prime index idx, the remaining primes have exponent 0.
        # So we record the solution at each leaf of the DFS tree?
        # Or better: iterate through primes, and at each step, try exponents 0..max_a.
        # If we choose exponent 0 for all remaining primes, we have a complete k.
        
        # Optimization: 
        # If we pick exponent a for prime p_i, the contribution to M is p_i^a * sigma(p_i^a).
        # Let T_i = p_i^a * sigma(p_i^a).
        # M = product T_i.
        
        pass # Placeholder, will rewrite below

    # Let's use a more direct recursive generation of M factors.
    # M = \prod_{p \in P} (p^{a_p} \sigma(p^{a_p}))
    # We want to find many distinct sets of exponents {a_p} that give the same M.
    
    # We'll use a dictionary to map M to the list of k's.
    # k = \prod p^{a_p}
    
    def solve():
        best_M = None
        best_ks = []
        
        # To limit search space, we only consider primes up to a certain limit
        # and exponents up to a certain limit.
        
        # We'll do a DFS over the primes.
        # At each prime p, we choose exponent a >= 0.
        # We stop if M > 10^18.
        
        # We need to be careful with the number of combinations.
        # With 15 primes, even with small max exponents, it can be large.
        # However, M grows very fast.
        # For p=2, a=1 -> 2*3=6. a=2 -> 4*7=28. a=3 -> 8*15=120.
        # For p=3, a=1 -> 3*4=12.
        
        # We will use a recursive function that passes the current M and current k.
        
        def recurse(i, curr_M, curr_k):
            nonlocal best_M, best_ks
            
            if curr_M > LIMIT_M:
                return
            
            if i == len(primes):
                # All primes considered (exponent 0 for the rest)
                # Check if curr_M is a candidate
                if curr_M > 1: # M=1 is trivial
                    if curr_M not in solutions:
                        solutions[curr_M] = []
                    # Add curr_k if distinct
                    if curr_k not in solutions[curr_M]:
                        solutions[curr_M].append(curr_k)
                        # Update best if this M has more solutions
                        if len(solutions[curr_M]) > len(best_ks):
                            best_ks = solutions[curr_M][:]
                            best_M = curr_M
                return
            
            p = primes[i]
            
            # Determine max exponent a for this prime
            # We need curr_M * (p^a * sigma(p^a)) <= LIMIT_M
            a = 0
            term = 1 # p^0 * sigma(p^0) = 1
            k_part = 1 # p^0
            
            while True:
                next_term = term * p * (p + 1) # This is wrong for a>1
                # Let's calculate properly
                # term_a = p^a * sigma(p^a)
                # term_0 = 1
                # term_1 = p * (1+p)
                # term_a = term_{a-1} * p * (1 + p + ... + p^{a-1} * (1+p)? No.
                # sigma(p^a) = sigma(p^{a-1}) * (1+p+...+p^{a-1})? No.
                # sigma(p^a) = 1 + p + ... + p^a.
                # sigma(p^a) = sigma(p^{a-1}) + p^a.
                # p^a * sigma(p^a) = p^a * (sigma(p^{a-1}) + p^a) = p^a * sigma(p^{a-1}) + p^{2a}.
                # This is getting complicated to update incrementally for the product M.
                
                # Let's just compute T(p, a) from scratch for small a.
                # Since LIMIT_M is 10^18, a will be small.
                # For p=2, 2^a * sigma(2^a) ~ 2^a * 2^{a+1} = 2^{2a+1}.
                # 2^{2a+1} <= 10^18 => 2a+1 <= 60 => a <= 30.
                # So a is small.
                
                break # Logic needs restructuring

            # Better approach: iterate a from 0 upwards.
            for a in range(0, 50): # 50 is safe upper bound
                # Calculate T = p^a * sigma(p^a)
                # sigma(p^a) = (p^(a+1)-1)/(p-1)
                if a == 0:
                    T = 1
                    k_p = 1
                else:
                    p_pow_a = pow(p, a)
                    sigma_val = (pow(p, a + 1) - 1) // (p - 1)
                    T = p_pow_a * sigma_val
                    k_p = p_pow_a
                
                new_M = curr_M * T
                if new_M > LIMIT_M:
                    break
                
                new_k = curr_k * k_p
                recurse(i + 1, new_M, new_k)
        
        recurse(0, 1, 1)
        
        if best_M is None:
            return {'M': 1, 'k': [1]}
            
        # Ensure k_i <= sqrt(M)
        # As discussed, this is always true for k>1.
        # For k=1, M=1, sqrt(1)=1.
        
        # Sort and distinct (already distinct in dict list)
        best_ks = list(set(best_ks)) # Just in case
        
        return {'M': best_M, 'k': best_ks}

    return solve()
