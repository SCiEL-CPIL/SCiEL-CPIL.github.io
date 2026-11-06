import math
import time
from itertools import product

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin below 150s wall-clock kill
    
    max_val = 10**18
    
    # Primes to consider
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_n = 1
    best_score = -1
    best_divisors = None
    
    def compute_score(divisors):
        """Compute tau_perp from sorted divisor list."""
        if len(divisors) < 2:
            return 0
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score
    
    def generate_divisors(exponents, primes_subset):
        """Generate all divisors from prime factorization."""
        if not exponents:
            return [1]
        
        divisors = [1]
        for i, p in enumerate(primes_subset):
            e = exponents[i]
            if e == 0:
                continue
            current_divs = list(divisors)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divisors.append(d * factor)
        
        divisors.sort()
        return divisors
    
    def tau_perp_from_exponents(exponents, primes_subset):
        """Calculate the score tau_perp(n) given the prime factors and their exponents."""
        if not exponents:
            return 0
        divisors = generate_divisors(exponents, primes_subset)
        if len(divisors) < 2:
            return 0
        return compute_score(divisors)
    
    # Strategy: Search over exponent vectors with non-increasing exponents
    # (since we use smallest primes, optimal has e1 >= e2 >= ... >= ek)
    
    def dfs(idx, max_exp, current_n, exponents_so_far, primes_used):
        nonlocal best_n, best_score, best_divisors
        
        if time.time() - start_time > TIME_LIMIT:
            return
        
        # Evaluate current candidate if valid
        if current_n > 1 and len(primes_used) >= 1:
            # Quick upper bound: max possible score is len(divisors)-1
            # tau = product of (e+1)
            tau = 1
            for e in exponents_so_far:
                tau *= (e + 1)
            if tau - 1 <= best_score:
                # Can't beat current best even in best case
                pass
            else:
                # Compute actual score
                divs = generate_divisors(exponents_so_far, primes_used)
                score = compute_score(divs)
                if score > best_score:
                    best_score = score
                    best_n = current_n
                    best_divisors = divs
        
        if idx >= len(primes):
            return
        
        p = primes[idx]
        
        # Determine max exponent for this prime
        limit = max_val // current_n
        
        e_max_possible = 0
        temp_product = 1
        while temp_product * p <= limit and e_max_possible < max_exp:
            temp_product *= p
            e_max_possible += 1
        
        e_limit = min(e_max_possible, max_exp)
        
        if e_limit < 1:
            return
        
        # Try exponents from high to low (dense structures first)
        # But also ensure we explore the space well
        # Key insight: for many primes, e=1 (squarefree) is often optimal
        # Let's try e=1 first, then other values
        
        # Actually, let's try all values but with pruning
        # The number of divisors grows as product of (e_i+1)
        # We want many divisors but also good coprime structure
        
        # Try e from e_limit down to 1
        # But limit the search: if we've already used many primes with e=1,
        # adding more with e=1 is good
        
        # Heuristic: prioritize e=1 for later primes
        if idx >= 5:  # For later primes, try e=1 first
            e_values = [1] + list(range(e_limit, 1, -1))
        else:
            e_values = list(range(e_limit, 0, -1))
        
        for e in e_values:
            if e > max_exp:
                continue
            next_n = current_n * (p ** e)
            if next_n > max_val:
                continue
            dfs(idx + 1, e, next_n, exponents_so_far + [e], primes_used + [p])
            
            if time.time() - start_time > TIME_LIMIT:
                return
    
    # Start DFS
    # Max exponent for 2 is log2(10^18) ~ 59
    # But we want many primes, so keep exponents small
    # Start with a reasonable max_exp
    dfs(0, 20, 1, [], [])
    
    # If we haven't found a good solution, try a targeted search
    # around known highly composite numbers
    if time.time() - start_time < TIME_LIMIT - 30:
        # Try specific highly composite structures
        # n = 2^a * 3^b * 5^c * 7^d * ... with a>=b>=c>=d>=...
        
        # Generate candidates with many small primes
        # Squarefree with k primes: product of first k primes
        # k=15: 2*3*5*...*47 ≈ 6.1e17
        # k=14: ≈ 1.1e16
        
        # Try mixed exponents
        candidate_exponents = [
            # (a, b, c, d, e, f, ...)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # squarefree 15 primes
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],     # squarefree 14 primes
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],        # 2^2 * squarefree 12
            [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],              # 2^2*3^2 * squarefree 9
            [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],              # 2^3 * squarefree 10
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],           # 2^2 * squarefree 11
            [4, 1, 1, 1, 1, 1, 1, 1, 1, 1],                 # 2^4 * squarefree 8
            [2, 2, 1, 1, 1, 1, 1, 1, 1, 1],                 # 2^2*3^2 * squarefree 8
            [3, 2, 1, 1, 1, 1, 1, 1, 1],                    # 2^3*3^2 * squarefree 7
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],              # 2^2 * squarefree 10
        ]
        
        for exps in candidate_exponents:
            if time.time() - start_time > TIME_LIMIT:
                break
            
            n_candidate = 1
            valid = True
            for i, e in enumerate(exps):
                if i >= len(primes):
                    valid = False
                    break
                n_candidate *= primes[i] ** e
                if n_candidate > max_val:
                    valid = False
                    break
            
            if not valid or n_candidate < 2:
                continue
            
            # Check non-increasing
            for i in range(len(exps) - 1):
                if exps[i] < exps[i+1]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            divs = generate_divisors(exps, primes[:len(exps)])
            score = compute_score(divs)
            
            if score > best_score:
                best_score = score
                best_n = n_candidate
                best_divisors = divs
    
    return best_n

if __name__ == "__main__":
    n = construct()
    # print(f"Best n: {n}")
    pass
