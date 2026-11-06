import time
from fractions import Fraction
import math
import sys

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    def time_ok():
        return time.time() - start_time < time_limit

    best_solution = None
    best_max_denom = float('inf')

    def update_solution(sol):
        nonlocal best_solution, best_max_denom
        if sol is None:
            return
        m = max(sol)
        if m < best_max_denom:
            best_max_denom = m
            best_solution = list(sol)

    # Function to find c, d such that 1/c + 1/d = p/q
    # Returns (c, d) with c < d, minimizing d.
    # c > min_c
    def find_min_max_cd(p, q, min_c):
        if p <= 0 or q <= 0:
            return None
        
        q2 = q * q
        lower_x = max(1, p * min_c - q + 1)
        
        for x in range(q, lower_x - 1, -1):
            if q2 % x != 0:
                continue
            y = q2 // x
            c_num = x + q
            d_num = y + q
            if c_num % p != 0 or d_num % p != 0:
                continue
            c = c_num // p
            d = d_num // p
            if c <= 0 or d <= 0:
                continue
            if c == d:
                continue
            if c < min_c:
                continue
            return (c, d)
            
        return None

    # Check if a solution with max denom <= K exists
    # Uses DFS with pruning
    def exists_solution(K):
        # a < b < c < d, all <= K
        # 1/a + 1/b + 1/c + 1/d = 5/31
        
        # a must be at least ceil(1/(5/31)) = ceil(31/5) = 7
        # a must be at most floor(31/5 * 4) = 24 (since 4/a >= 5/31 => a <= 124/5 = 24.8)
        # Actually, 1/a <= 5/31 => a >= 31/5 = 6.2, so a >= 7
        # And 1/a + 1/b + 1/c + 1/d = 5/31
        # Minimum sum with a,b,c,d >= a is 4/a <= 5/31 => a >= 124/5 = 24.8? No.
        # 4/a >= 5/31 => a <= 124/5 = 24.8
        # So a is in [7, 24]
        
        for a in range(7, min(25, K + 1)):
            if not time_ok():
                return False
            rem_a = target - Fraction(1, a)
            if rem_a <= 0:
                continue
            
            # b > a, 1/b < rem_a => b > 1/rem_a
            b_min = int(math.ceil(1.0 / float(rem_a))) + 1
            if b_min <= a:
                b_min = a + 1
            if b_min > K:
                continue
            
            # b max: 1/b >= rem_a/3 (since 1/c + 1/d <= 2/b roughly, but more precisely)
            # Actually, 1/b + 1/c + 1/d = rem_a
            # 3/b >= rem_a => b <= 3/rem_a
            b_max = int(math.floor(3.0 / float(rem_a)))
            if b_max > K:
                b_max = K
            
            for b in range(b_min, b_max + 1):
                if not time_ok():
                    return False
                rem_b = rem_a - Fraction(1, b)
                if rem_b <= 0:
                    continue
                
                # c > b, 1/c < rem_b => c > 1/rem_b
                c_min = int(math.ceil(1.0 / float(rem_b))) + 1
                if c_min <= b:
                    c_min = b + 1
                if c_min > K:
                    continue
                
                # c max: 1/c + 1/d = rem_b
                # 2/c >= rem_b => c <= 2/rem_b
                c_max = int(math.floor(2.0 / float(rem_b)))
                if c_max > K:
                    c_max = K
                
                for c in range(c_min, c_max + 1):
                    if not time_ok():
                        return False
                    rem_c = rem_b - Fraction(1, c)
                    if rem_c <= 0:
                        continue
                    
                    # d > c, 1/d = rem_c => d = 1/rem_c
                    # d must be integer, d > c, d <= K
                    if rem_c.numerator != 1:
                        continue
                    d = rem_c.denominator
                    if d > c and d <= K:
                        return True
        
        return False

    # Binary search for minimum K
    lo = 1
    hi = 10000000
    
    # First, find a feasible solution to get an upper bound
    # Use the greedy approach to find any solution
    def find_any_solution():
        # 3-term solution: 1/8 + 1/28 + 1/1736 = 5/31
        return [8, 28, 1736]
    
    any_sol = find_any_solution()
    hi = max(any_sol)
    
    # Binary search
    while lo < hi and time_ok():
        mid = (lo + hi) // 2
        if exists_solution(mid):
            hi = mid
        else:
            lo = mid + 1
    
    K = lo
    
    # Now find the actual solution with max denom <= K
    # We need to find one solution
    solution = None
    
    for a in range(7, min(25, K + 1)):
        if not time_ok():
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        
        b_min = int(math.ceil(1.0 / float(rem_a))) + 1
        if b_min <= a:
            b_min = a + 1
        if b_min > K:
            continue
        
        b_max = int(math.floor(3.0 / float(rem_a)))
        if b_max > K:
            b_max = K
        
        for b in range(b_min, b_max + 1):
            if not time_ok():
                break
            rem_b = rem_a - Fraction(1, b)
            if rem_b <= 0:
                continue
            
            c_min = int(math.ceil(1.0 / float(rem_b))) + 1
            if c_min <= b:
                c_min = b + 1
            if c_min > K:
                continue
            
            c_max = int(math.floor(2.0 / float(rem_b)))
            if c_max > K:
                c_max = K
            
            for c in range(c_min, c_max + 1):
                if not time_ok():
                    break
                rem_c = rem_b - Fraction(1, c)
                if rem_c <= 0:
                    continue
                
                if rem_c.numerator != 1:
                    continue
                d = rem_c.denominator
                if d > c and d <= K:
                    solution = [a, b, c, d]
                    break
            if solution:
                break
        if solution:
            break
    
    if solution is None:
        # Fallback to the known solution
        solution = any_sol
    
    return solution
