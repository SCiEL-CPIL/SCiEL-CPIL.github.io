import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin below 150s wall-clock kill
    
    # The optimal n is expected to use the smallest primes.
    # Product of first 15 primes: 2*3*5*...*47 ~ 6e17.
    # Product of first 16 primes: *53 ~ 3.4e19 > 10^18.
    # So k is at most 15.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    max_val = 10**18
    
    best_n = 1
    best_score = -1
    
    # Cache for divisor generation if needed, but direct calc is likely faster for one-shot.
    
    def tau_perp(n):
        """
        Calculate the score tau_perp(n).
        """
        if n == 1:
            return 0
            
        # Factorize n using the known primes (since n is generated from them)
        divisors = [1]
        temp = n
        
        for p in primes:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                # Extend divisors
                current_divs = list(divisors)
                factor = 1
                for _ in range(count):
                    factor *= p
                    for d in current_divs:
                        divisors.append(d * factor)
            
            # If temp becomes 1, we have factored n completely
            if temp == 1:
                break
        
        # If temp > 1, n has a prime factor larger than our list (should not happen for our candidates)
        # But for safety, handle it if needed. In our construction, n is always from 'primes'.
        
        divisors.sort()
        
        score = 0
        # Optimization: use math.gcd
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
                
        return score

    def tau_perp_upper_bound(exponents):
        """
        Fast upper bound on tau_perp for a candidate with given exponents.
        The maximum possible coprime adjacency count is bounded by:
          - Number of divisors minus 1 (trivial upper bound)
          - More tightly: each divisor d can be coprime to at most 2 neighbors.
            So tau_perp <= 2 * (number of divisors) but that's loose.
        
        A better bound: consider the divisor lattice. In the sorted divisor list,
        coprime adjacencies are limited. For n = prod p_i^e_i, the total number
        of divisors is prod(e_i+1). The score is at most prod(e_i+1) - 1.
        
        For pruning, we use: if current best score >= prod(e_i+1) - 1, skip.
        This is a valid upper bound since you can't have more coprime pairs
        than total adjacent pairs.
        """
        total_divs = 1
        for e in exponents:
            total_divs *= (e + 1)
        return total_divs - 1

    def dfs(idx, max_exp, current_n, exponents_so_far):
        nonlocal best_n, best_score
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        # If we have included at least one prime, it's a valid candidate
        # We evaluate all candidates found.
        if current_n > 1:
            # Use upper bound to prune: if even the max possible score
            # for this partial assignment (extended to full 15 primes with e=1)
            # can't beat best_score, skip.
            # Actually, let's just evaluate directly for simplicity, but add
            # a quick check: total divisors of current_n gives an upper bound
            # on how many more coprime pairs we could get.
            # For now, evaluate directly.
            score = tau_perp(current_n)
            if score > best_score:
                best_score = score
                best_n = current_n
        
        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Determine max exponent for this prime
        # current_n * p^e <= max_val
        # p^e <= max_val // current_n
        limit = max_val // current_n
        
        e_max_possible = 0
        temp_product = 1
        while temp_product * p <= limit and e_max_possible < max_exp:
            temp_product *= p
            e_max_possible += 1
        
        e_limit = min(e_max_possible, max_exp)
        
        # Try exponents from e_max down to 1 (higher exponents first for better pruning)
        for e in range(e_limit, 0, -1):
            next_n = current_n * (p ** e)
            dfs(idx + 1, e, next_n, exponents_so_far + [e])
            
            if time.time() - start_time > TIME_LIMIT:
                return

    # Start DFS
    # Max exponent for 2 is log2(10^18) ~ 59
    # We pass max_exp = 60 to allow any reasonable start.
    dfs(0, 60, 1, [])
    
    return best_n

if __name__ == "__main__":
    # Optional: run a quick self-test
    n = construct()
    # print(f"Best n: {n}")
    # print(f"Score: {tau_perp(n)}")
    pass
