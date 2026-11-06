import math
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    def search(rem, last_d, k, path):
        if k == 0:
            return path if rem == 0 else None
        
        if k == 1:
            if rem.denominator == 1:
                d = rem.numerator
                if d > last_d:
                    return path + [d]
            return None
            
        if k == 2:
            # 1/d1 + 1/d2 = rem
            # d2 = 1 / (rem - 1/d1)
            # d1 > last_d
            # d1 < 1/rem
            # Also d1 < d2 => 1/d1 > 1/d2 => rem - 1/d1 < 1/d1 => rem < 2/d1 => d1 < 2/rem
            # So d1 in (last_d, floor(2/rem))
            max_d1 = int(2 / rem)
            for d1 in range(last_d + 1, max_d1 + 1):
                term1 = Fraction(1, d1)
                if term1 >= rem:
                    continue
                rem2 = rem - term1
                # d2 = 1/rem2
                if rem2.denominator == 1:
                    d2 = rem2.numerator
                    if d2 > d1:
                        return path + [d1, d2]
            return None

        # Upper bound for d: k * (1/d) >= rem  => d <= k / rem
        max_d_frac = Fraction(k, 1) / rem
        if max_d_frac < 0:
            return None
        max_d = int(max_d_frac)
        
        d_min = last_d + 1
        
        # Check if min sum > rem
        current_sum = Fraction(0, 1)
        for i in range(k):
            current_sum += Fraction(1, d_min + i)
            
        while current_sum > rem:
            d_min += 1
            if d_min > max_d:
                return None
            current_sum = Fraction(0, 1)
            for i in range(k):
                current_sum += Fraction(1, d_min + i)

        for d in range(d_min, max_d + 1):
            if time.time() > deadline:
                return None
            term = Fraction(1, d)
            if term > rem:
                continue
            
            # Prune if max possible sum of remaining k-1 terms < rem - term
            if Fraction(k-1, 1) / (d + 1) < rem - term:
                continue
                
            res = search(rem - term, d, k - 1, path + [d])
            if res:
                return res
        return None

    for k in range(5, 25):
        res = search(Fraction(1, 1), 4, k, [])
        if res:
            return res
            
    # Fallback
    def greedy_egyptian(n, d, min_denom):
        res = []
        num = n
        den = d
        while num > 0:
            x = (den + num - 1) // num
            if x < min_denom:
                x = min_denom
            res.append(x)
            new_num = num * x - den
            new_den = den * x
            g = math.gcd(new_num, new_den)
            num = new_num // g
            den = new_den // g
            min_denom = x + 1
        return res

    return greedy_egyptian(1, 1, 5)
