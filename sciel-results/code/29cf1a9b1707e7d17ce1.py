import math

def construct():
    # We aim to maximize the number of divisors, which generally correlates with the score.
    # We search for exponents e_i for primes p_i such that product p_i^e_i <= 10^18
    # and product (e_i + 1) is maximized.
    # We assume e_1 >= e_2 >= ... to avoid duplicate factorizations.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    limit = 10**18
    
    best_score = 0
    best_n = 1
    
    # Recursive search for exponents
    # idx: index of current prime
    # remaining_limit: max allowed value for remaining part
    # current_n: product so far
    # current_divisors: product of (e+1) so far
    # max_exp: max allowed exponent for current prime (to enforce non-increasing)
    
    def search(idx, remaining_limit, current_n, current_divisors, max_exp):
        nonlocal best_score, best_n
        
        if idx >= len(primes):
            if current_divisors > best_score:
                best_score = current_divisors
                best_n = current_n
            return
            
        p = primes[idx]
        
        # Try exponents from max_exp down to 0
        # We can prune if p**e > remaining_limit
        # Also, if current_divisors * (max_exp + 1) ** (len(primes) - idx) <= best_score, we might prune? 
        # Simple pruning: if even with max exponents for all remaining primes we can't beat best_score.
        
        # Calculate max possible divisors if we used max_exp for all remaining (optimistic bound)
        # This bound is loose but helps.
        
        for e in range(max_exp, -1, -1):
            if e > 0:
                # Check if p**e fits
                # Use log to avoid overflow issues in intermediate steps, or just integer math
                # Since limit is 10^18, and p is small, p**e is manageable.
                val = p ** e
                if val > remaining_limit:
                    continue
                
                new_n = current_n * val
                new_divs = current_divisors * (e + 1)
                
                # Optimization: if new_divs * (max_exp + 1) ** (len(primes) - idx - 1) <= best_score:
                #    continue # Prune branch if max possible is not better
                
                search(idx + 1, remaining_limit // val, new_n, new_divs, e)
            else:
                # e=0 case
                # If we pick 0, all subsequent exponents must be 0
                # So we can just update best and return
                if current_divisors > best_score:
                    best_score = current_divisors
                    best_n = current_n
                return

    # Start search
    # Max exponent for 2 is floor(log2(10^18)) = 59
    search(0, limit, 1, 1, 59)
    
    return best_n
