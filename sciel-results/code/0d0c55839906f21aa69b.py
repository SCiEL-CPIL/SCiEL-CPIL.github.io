import time
import math
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Known good solution
    best_solution = [10, 15, 930]
    best_max_denom = 930
    
    def find_3term_solutions(max_denom):
        """Find 3-term Egyptian fraction solutions using algebraic completion."""
        solutions = []
        # We want 1/a + 1/b + 1/c = 5/31 with 1 <= a < b < c <= max_denom
        # Rearranging: 1/b + 1/c = 5/31 - 1/a = (5a - 31)/(31a)
        # Let p = 5a - 31, q = 31a
        # We need 1/b + 1/c = p/q, so (b+c)/(bc) = p/q
        # This gives: p*b*c = q*(b+c)
        # Rearranging: p*bc - q*b - q*c = 0
        # Multiply by p: p^2*bc - p*q*b - p*q*c = 0
        # Add q^2: p^2*bc - p*q*b - p*q*c + q^2 = q^2
        # (p*b - q)(p*c - q) = q^2
        # So we need to find divisors d of q^2 such that:
        #   p*b - q = d, p*c - q = q^2/d
        #   b = (d + q) / p, c = (q^2/d + q) / p
        #   b and c must be positive integers, b < c, and b > a
        
        for a in range(1, max_denom + 1):
            p = 5 * a - 31
            if p <= 0:
                continue
            q = 31 * a
            
            # Find all divisors of q^2
            q_sq = q * q
            # Get divisors of q_sq
            divisors = []
            i = 1
            while i * i <= q_sq:
                if q_sq % i == 0:
                    divisors.append(i)
                    if i != q_sq // i:
                        divisors.append(q_sq // i)
                i += 1
            
            for d in divisors:
                # b = (d + q) / p
                # c = (q_sq // d + q) / p
                num_b = d + q
                num_c = q_sq // d + q
                
                if num_b % p != 0 or num_c % p != 0:
                    continue
                
                b = num_b // p
                c = num_c // p
                
                if b <= a or c <= b:
                    continue
                if c > max_denom:
                    continue
                
                # Verify the solution
                if a * b * c * 5 == 31 * (b * c + a * c + a * b):
                    sol = [a, b, c]
                    solutions.append(sol)
        
        return solutions
    
    def find_2term_completion(max_denom, a, b):
        """Find c such that 1/a + 1/b + 1/c = 5/31."""
        # 1/c = 5/31 - 1/a - 1/b
        # 1/c = (5ab - 31b - 31a) / (31ab)
        # c = 31ab / (5ab - 31b - 31a)
        num = 5 * a * b - 31 * b - 31 * a
        if num <= 0:
            return None
        den = 31 * a * b
        if den % num != 0:
            return None
        c = den // num
        if c <= b or c > max_denom:
            return None
        # Verify
        if a * b * c * 5 == 31 * (b * c + a * c + a * b):
            return c
        return None
    
    # First, exhaustively search for 3-term solutions with small max_denom
    # Start with a reasonable bound and increase
    for max_d in [200, 300, 400, 500, 700, 900, 1200, 1500, 2000, 3000, 5000]:
        if time.time() - start_time > time_limit:
            break
        
        solutions = find_3term_solutions(max_d)
        for sol in solutions:
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
        
        # Also try 4-term solutions by fixing first two and completing
        if time.time() - start_time > time_limit:
            break
        
        # For 4-term: 1/a + 1/b + 1/c + 1/d = 5/31
        # Try small a, b and complete with c, d
        for a in range(1, min(50, max_d)):
            if time.time() - start_time > time_limit:
                break
            for b in range(a + 1, min(100, max_d)):
                if time.time() - start_time > time_limit:
                    break
                # Find c, d such that 1/c + 1/d = 5/31 - 1/a - 1/b
                rem_num = 5 * a * b - 31 * b - 31 * a
                rem_den = 31 * a * b
                if rem_num <= 0:
                    continue
                
                # Need 1/c + 1/d = rem_num/rem_den
                # (c+d)/(cd) = rem_num/rem_den
                # rem_num*cd = rem_den*(c+d)
                # rem_num^2*cd - rem_num*rem_den*c - rem_num*rem_den*d = 0
                # (rem_num*c - rem_den)(rem_num*d - rem_den) = rem_den^2
                rem_den_sq = rem_den * rem_den
                i = 1
                while i * i <= rem_den_sq:
                    if rem_den_sq % i == 0:
                        for d_val in [i, rem_den_sq // i]:
                            if d_val == i:
                                continue
                            num_c = d_val + rem_den
                            num_d = rem_den_sq // d_val + rem_den
                            if num_c % rem_num == 0 and num_d % rem_num == 0:
                                c = num_c // rem_num
                                d = num_d // rem_num
                                if b < c < d <= max_d:
                                    # Verify
                                    sol = [a, b, c, d]
                                    if a * b * c * d * 5 == 31 * (b * c * d + a * c * d + a * b * d + a * b * c):
                                        actual_max = max(sol)
                                        if actual_max < best_max_denom:
                                            best_max_denom = actual_max
                                            best_solution = sol
                    i += 1
    
    # Finally, use DFS as a fallback for potentially better solutions
    def dfs(n, d, min_d, path):
        if time.time() - start_time > time_limit:
            return None
        if n == 0:
            return path
        if n < 0:
            return None
        
        # Lower bound for x
        lower = max(min_d, (d + n - 1) // n)
        
        if lower > best_max_denom - 1:
            return None
        
        for x in range(lower, best_max_denom):
            num_next = n * x - d
            if num_next == 0:
                return path + [x]
            if num_next < 0:
                break
            
            den_next = d * x
            g = gcd(num_next, den_next)
            num_next //= g
            den_next //= g
            
            # Prune: Can we form num_next/den_next with denoms in [x+1, best_max_denom-1]?
            upper_bound_sum = (best_max_denom - 1 - x) / (x + 1)
            if upper_bound_sum < num_next / den_next:
                break
                
            res = dfs(num_next, den_next, x + 1, path + [x])
            if res is not None:
                return res
        return None
    
    sol = dfs(5, 31, 1, [])
    if sol:
        actual_max = max(sol)
        if actual_max < best_max_denom:
            best_max_denom = actual_max
            best_solution = sol
    
    return best_solution
