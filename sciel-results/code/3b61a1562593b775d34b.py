import time
from fractions import Fraction
from math import gcd

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    time_limit = 55.0
    
    best_solution = [8, 31, 248]
    best_max_denom = 248
    
    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == target
    
    # Search 3-term solutions: 1/a + 1/b + 1/c = 5/31, a < b < c
    # For each a >= 7, solve 1/b + 1/c = (5a-31)/(31a)
    # Let p = 5a-31, q = 31a
    # (p*b - q)*(p*c - q) = q^2
    # Let X = p*b - q, Y = p*c - q
    # X*Y = q^2, X < Y
    # X = k*p - q
    # Since X < q, k*p - q < q => k*p < 2q => k < 2q/p
    # Since X > 0, k*p > q => k > q/p
    
    for a in range(7, 100):
        if time.time() - start_time > time_limit:
            break
        
        p = 5 * a - 31
        q = 31 * a
        
        if p <= 0:
            continue
        
        g = gcd(p, q)
        p_r = p // g
        q_r = q // g
        
        # k must be integer such that:
        # q_r/p_r < k < 2*q_r/p_r
        k_min = int(q_r / p_r) + 1
        k_max = int(2 * q_r / p_r)
        
        for k in range(k_min, k_max + 1):
            X = k * p_r - q_r
            if X <= 0 or X >= q_r:
                continue
            
            q_r_sq = q_r * q_r
            if q_r_sq % X != 0:
                continue
            
            Y = q_r_sq // X
            
            # b = (X + q_r) / p_r
            # c = (Y + q_r) / p_r
            # Check divisibility
            if (X + q_r) % p_r != 0:
                continue
            if (Y + q_r) % p_r != 0:
                continue
            
            b = (X + q_r) // p_r
            c = (Y + q_r) // p_r
            
            if b <= a or c <= b:
                continue
            
            if c < best_max_denom:
                best_solution = [a, b, c]
                best_max_denom = c
    
    if verify(best_solution):
        return best_solution
    
    return [8, 31, 248]
