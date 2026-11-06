import math
import time
from functools import lru_cache

T12 = math.factorial(12)
PRIMES = [2, 3, 5, 7, 11]
INITIAL_EXP = (10, 5, 2, 1, 1)

def exp_to_val(exp):
    result = 1
    for i, e in enumerate(exp):
        if e > 0:
            result *= PRIMES[i] ** e
    return result

def val_to_exp(n):
    exp = [0, 0, 0, 0, 0]
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            exp[i] += 1
            n //= p
    if n != 1:
        return None
    return tuple(exp)

# Precompute all divisors for all exponent states
divisors_by_exp = {}
for e2 in range(11):
    for e3 in range(6):
        for e5 in range(3):
            for e7 in range(2):
                for e11 in range(2):
                    exp = (e2, e3, e5, e7, e11)
                    val = exp_to_val(exp)
                    divs = []
                    # Generate divisors of val
                    # Since val is small, we can just iterate 1..val? No, val is up to 479M.
                    # Better to generate from prime factors.
                    divs = [1]
                    for i, e in enumerate(exp):
                        if e > 0:
                            p = PRIMES[i]
                            powers = [1]
                            for k in range(1, e + 1):
                                powers.append(powers[-1] * p)
                            new_divs = []
                            for d in divs:
                                for pow_val in powers:
                                    new_divs.append(d * pow_val)
                            divs = new_divs
                    divisors_by_exp[exp] = sorted(divs)

def solve():
    deadline = time.time() + 55
    best_solution = None
    
    # M must be at least 22 because of the factor 11
    # 11 must be in a factor > 12. Smallest is 22.
    for M in range(22, 1000):
        if time.time() > deadline:
            break
            
        # Memoization for this M
        # Key: (rem_exp, last_factor)
        # Value: True/False or the list of factors
        # Using a set for visited states to avoid cycles? No, it's a tree (decreasing rem_exp).
        # But different paths can lead to same (rem_exp, last).
        memo = {}
        
        def dfs(rem_exp, last):
            if rem_exp == (0, 0, 0, 0, 0):
                return []
            
            key = (rem_exp, last)
            if key in memo:
                return memo[key]
            
            # Get divisors of rem_exp
            divs = divisors_by_exp[rem_exp]
            
            # Filter divisors: d > last, d <= M
            # Also, we should check if it's possible to form the remainder.
            # Pruning:
            # If rem_exp has a prime factor P, and the smallest multiple of P > last is > M, 
            # and P is not "covered" by a composite d <= M?
            # Actually, simpler: just try all valid d.
            
            found = None
            # Try divisors in descending order? 
            # Let's try descending. It might find a solution faster by reducing the remainder quickly.
            # Or ascending? 
            # If we pick small d, we have many more factors to pick.
            # If we pick large d, we have fewer.
            # Let's try descending.
            for d in reversed(divs):
                if d <= last:
                    continue
                if d > M:
                    continue
                
                # Check if d is a valid factor (it is, by construction)
                # Calculate new rem_exp
                d_exp = val_to_exp(d)
                if d_exp is None:
                    continue
                
                new_rem_exp = tuple(rem_exp[i] - d_exp[i] for i in range(5))
                if any(x < 0 for x in new_rem_exp):
                    continue
                
                # Recurse
                res = dfs(new_rem_exp, d)
                if res is not None:
                    found = [d] + res
                    break
            
            memo[key] = found
            return found
        
        res = dfs(INITIAL_EXP, 0)
        if res is not None:
            # Verify
            prod = 1
            for f in res:
                prod *= f
            if prod == T12 and all(f > 12 for f in res) and len(set(res)) == len(res) and max(res) <= M:
                return res

    return [T12]

def construct():
    try:
        sol = solve()
        return sol
    except:
        return [T12]
