import time
from fractions import Fraction
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 45
    
    target = Fraction(5, 31)
    
    # Known solution
    best_solution = [8, 31, 248]
    best_max_den = 248
    
    def update_best(dens):
        nonlocal best_solution, best_max_den
        if len(dens) > 0 and len(dens) <= 256:
            # Verify distinct and in range
            if len(set(dens)) != len(dens):
                return
            if any(d < 1 or d > 10000000 for d in dens):
                return
            # Verify sum
            s = sum(Fraction(1, d) for d in dens)
            if s == target:
                m = max(dens)
                if m < best_max_den:
                    best_max_den = m
                    best_solution = dens
                    
    # Search 3-term solutions
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # 1/a < 5/31 => a > 6.2 => a >= 7
    # 3/a > 5/31 => a < 18.6 => a <= 18
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem_num = 5 * a - 31
        rem_den = 31 * a
        if rem_num <= 0:
            continue
            
        # 1/b + 1/c = rem_num / rem_den
        # (rem_num*b - rem_den)(rem_num*c - rem_den) = rem_den^2
        # X * Y = rem_den^2
        # X = rem_num*b - rem_den => b = (X + rem_den) / rem_num
        # Y = rem_num*c - rem_den => c = (Y + rem_den) / rem_num
        
        D2 = rem_den * rem_den
        # Iterate X up to sqrt(D2) = rem_den
        # Optimization: X must be such that (X + rem_den) % rem_num == 0
        # X ≡ -rem_den (mod rem_num)
        
        # Find smallest X >= 1 satisfying condition
        # X = k * rem_num - rem_den
        # We need X > 0 => k * rem_num > rem_den => k > rem_den / rem_num
        # Also X <= rem_den (since X <= Y and XY=D2)
        
        start_k = (rem_den // rem_num) + 1
        # Actually X can be smaller if we don't assume X <= Y? 
        # We assume X <= Y => X <= sqrt(D2) = rem_den.
        
        for k in range(start_k, (rem_den // rem_num) + 2):
            X = k * rem_num - rem_den
            if X > rem_den:
                break
            if X <= 0:
                continue
            if D2 % X != 0:
                continue
            Y = D2 // X
            if Y < X:
                continue
            
            b_num = X + rem_den
            c_num = Y + rem_den
            
            if b_num % rem_num != 0:
                continue
            if c_num % rem_num != 0:
                continue
                
            b = b_num // rem_num
            c = c_num // rem_num
            
            # Check distinct and order
            if b == a or c == a or b == c:
                continue
            if b < a or c < b:
                continue
            if b < 1 or c < 1:
                continue
                
            update_best([a, b, c])

    # Search 4-term solutions
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # a >= 7
    # 4/a > 5/31 => a < 24.8 => a <= 24
    
    for a in range(7, 25):
        if time.time() > deadline:
            break
        rem1_num = 5 * a - 31
        rem1_den = 31 * a
        if rem1_num <= 0:
            continue
        
        # 1/b + 1/c + 1/d = rem1_num / rem1_den
        # b > a
        # 3/b > rem1_num / rem1_den => b < 3 * rem1_den / rem1_num
        # 1/b < rem1_num / rem1_den => b > rem1_den / rem1_num
        
        b_min = a + 1
        b_max = int(3 * rem1_den / rem1_num)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem2_num = rem1_num * b - rem1_den
            rem2_den = rem1_den * b
            if rem2_num <= 0:
                continue
            
            # 1/c + 1/d = rem2_num / rem2_den
            # (rem2_num*c - rem2_den)(rem2_num*d - rem2_den) = rem2_den^2
            # X * Y = rem2_den^2
            # X = rem2_num*c - rem2_den => c = (X + rem2_den) / rem2_num
            # Y = rem2_num*d - rem2_den => d = (Y + rem2_den) / rem2_num
            
            D2 = rem2_den * rem2_den
            # X <= rem2_den
            # X ≡ -rem2_den (mod rem2_num)
            
            start_k = (rem2_den // rem2_num) + 1
            # Max k such that X <= rem2_den
            # k * rem2_num - rem2_den <= rem2_den => k * rem2_num <= 2 * rem2_den => k <= 2 * rem2_den / rem2_num
            
            end_k = int(2 * rem2_den / rem2_num) + 1
            
            for k in range(start_k, end_k + 1):
                X = k * rem2_num - rem2_den
                if X > rem2_den:
                    break
                if X <= 0:
                    continue
                if D2 % X != 0:
                    continue
                Y = D2 // X
                if Y < X:
                    continue
                
                c_num = X + rem2_den
                d_num = Y + rem2_den
                
                if c_num % rem2_num != 0:
                    continue
                if d_num % rem2_num != 0:
                    continue
                    
                c = c_num // rem2_num
                d = d_num // rem2_num
                
                if c == a or c == b or c == d or d == a or d == b:
                    continue
                if c < b or d < c:
                    continue
                if c < 1 or d < 1:
                    continue
                    
                update_best([a, b, c, d])

    return best_solution
