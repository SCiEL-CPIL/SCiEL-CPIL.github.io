import math
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    max_exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of N
    divisors = [1]
    for i, p in enumerate(primes):
        new_divs = []
        power = 1
        for _ in range(max_exponents[i] + 1):
            for d in divisors:
                new_divs.append(d * power)
            power *= p
        divisors = new_divs
        
    divisors.sort()
    divisors_set = set(divisors)
    
    # Map divisor to its prime exponent vector for fast subtraction
    # This is the key for the CSP approach using exponent vectors
    divisor_to_exps = {}
    for d in divisors:
        exps = []
        temp = d
        for p in primes:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            exps.append(count)
        divisor_to_exps[d] = tuple(exps)
        
    # Precompute valid divisors for each exponent vector state?
    # No, we will filter on the fly. The number of divisors is 792.
    
    # We want to minimize the largest factor M.
    # Binary search on M.
    
    # Lower bound for M:
    # We need at least 1 factor. If 1 factor, M=N.
    # If 2 factors a*b=N, a,b > 12. Minimize max(a,b).
    # Best is sqrt(N) ~ 21886.
    # With more factors, M can be smaller.
    # 13 factors is max 12.
    # Let's estimate a lower bound.
    # Product of 12 distinct integers > 12.
    # Min product: 13*14*...*24 = 24!/12!
    # 24!/12! is huge.
    # Let's just start low.
    # If M=13, only factor 13 is allowed. 13 divides N. N/13 = 36846276.
    # Next factor must be > 13.
    
    def can_decompose(exps_remaining, min_factor, max_factor, depth, used_factors):
        """
        Check if we can decompose the number represented by exps_remaining
        into distinct factors in [min_factor, max_factor].
        exps_remaining: tuple of exponents for primes [2, 3, 5, 7, 11]
        min_factor: minimum value for the next factor
        max_factor: maximum value for any factor
        depth: number of factors already chosen
        used_factors: set of already chosen factors (for distinctness, though min_factor ensures distinctness if we pick in increasing order)
        """
        if time.time() > deadline:
            return False
            
        if sum(exps_remaining) == 0:
            return True
            
        if depth >= 12:
            return False
            
        if min_factor > max_factor:
            return False
            
        # Check if any divisor d of the remaining number (represented by exps_remaining)
        # is in [min_factor, max_factor] and its exponents are <= exps_remaining
        
        # Iterate through all divisors of N
        # Optimization: We can iterate through divisors of the remaining number.
        # But we don't have a direct list. We can filter divisors of N.
        
        # Pruning:
        # The product of the remaining factors must be >= min_factor * (min_factor+1) * ...
        # Actually, we just need to find ONE valid next factor.
        
        # To ensure distinctness, we pick factors in increasing order.
        # So the next factor d must be >= min_factor.
        # And d must be a divisor of the remaining number.
        
        # Let's iterate through divisors of N that are >= min_factor and <= max_factor.
        # And check if they divide the remaining number.
        
        # Efficient check:
        # d divides remaining if for all i, exps_d[i] <= exps_remaining[i].
        
        found = False
        # Sort candidates?
        # We want to find a solution.
        # Trying smaller d first allows more room for larger factors?
        # Or trying larger d first reduces the remaining product faster?
        # Let's try ascending order of d.
        
        # Precomputed list of divisors sorted
        # We can precompute a list of (d, exps_d) for all divisors
        # And filter by range
        
        for d, exps_d in divisor_list:
            if d < min_factor:
                continue
            if d > max_factor:
                break  # Since divisor_list is sorted by d
            
            # Check if d divides remaining
            valid = True
            for i in range(len(primes)):
                if exps_d[i] > exps_remaining[i]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Try picking d
            new_exps = tuple(exps_remaining[i] - exps_d[i] for i in range(len(primes)))
            
            if can_decompose(new_exps, d + 1, max_factor, depth + 1, used_factors | {d}):
                return True
                
        return False

    # Precompute divisor_list: list of (d, exps_d) sorted by d
    divisor_list = [(d, divisor_to_exps[d]) for d in divisors]
    divisor_list.sort(key=lambda x: x[0])
    
    # To speed up can_decompose, we can precompute for each min_factor index,
    # but min_factor changes dynamically.
    
    # Let's try binary search on M
    low = 13
    high = N
    best_M = N
    
    # Initial check: is M=N possible? Yes, [N]. But we want min.
    # Let's find a tighter upper bound first?
    # Maybe just binary search.
    
    # Optimization for can_decompose:
    # If the remaining number is 1, return True.
    # If depth == 12, return False.
    # If min_factor > max_factor, return False.
    
    # We can also prune if the product of the smallest possible distinct factors
    # starting from min_factor exceeds the remaining number?
    # No, the remaining number is fixed. We are decomposing it.
    
    # Let's add a quick check:
    # If remaining number is prime (in terms of divisors > 1), and > max_factor, fail.
    # But "prime" here means no divisors in range.
    
    # Let's run the binary search
    while low < high:
        if time.time() > deadline:
            break
            
        mid = (low + high) // 2
        
        if can_decompose(tuple(max_exponents), 13, mid, 0, set()):
            best_M = mid
            high = mid
        else:
            low = mid + 1
            
    # Now find the actual factors for best_M
    def find_factors(exps_remaining, min_factor, max_factor, depth):
        if time.time() > deadline:
            return None
            
        if sum(exps_remaining) == 0:
            return []
            
        if depth >= 12:
            return None
            
        if min_factor > max_factor:
            return None
            
        for d, exps_d in divisor_list:
            if d < min_factor:
                continue
            if d > max_factor:
                break
                
            valid = True
            for i in range(len(primes)):
                if exps_d[i] > exps_remaining[i]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            new_exps = tuple(exps_remaining[i] - exps_d[i] for i in range(len(primes)))
            sub_factors = find_factors(new_exps, d + 1, max_factor, depth + 1)
            if sub_factors is not None:
                return [d] + sub_factors
                
        return None

    factors = find_factors(tuple(max_exponents), 13, best_M, 0)
    
    if factors is None:
        # Fallback
        return [N]
        
    return factors
