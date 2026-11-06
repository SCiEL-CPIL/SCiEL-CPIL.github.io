"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    target_num = 5
    target_den = 31
    
    best = [8, 31, 248]
    best_max = 248
    
    # Helper: compute (n/d - 1/x) = (nx - d) / (dx), reduced
    def sub_recip(n, d, x):
        num = n * x - d
        den = d * x
        if num <= 0:
            return (0, 1)
        g = gcd(num, den)
        return (num // g, den // g)
    
    # Helper: upper bound for k remaining terms: k/n > rem = n/d → d*k > n*n_rem... 
    # Actually: we need 1/x < rem AND k/x > rem where k is terms remaining after x
    # 1/x < n/d → x > d/n → x_min = floor(d/n) + 1
    # k/x > n/d → x < k*d/n → x_max = ceil(k*d/n) - 1
    def bounds(n, d, k, prev):
        # x > d/n → x_min = floor(d/n) + 1
        x_min = d // n + 1
        if x_min <= prev:
            x_min = prev + 1
        # x < k*d/n → x_max = ceil(k*d/n) - 1 = (k*d + n - 1) // n - 1... 
        # Actually ceil(k*d/n) - 1. If k*d is divisible by n, ceil = k*d/n, so x_max = k*d/n - 1
        # Otherwise ceil(k*d/n) = (k*d + n - 1) // n
        if (k * d) % n == 0:
            x_max = (k * d) // n - 1
        else:
            x_max = (k * d + n - 1) // n - 1
        return x_min, x_max
    
    # 3-term search: 1/a + 1/b + 1/c = 5/31
    for a in range(7, 19):
        if time.time() > deadline:
            return best
        rn, rd = sub_recip(target_num, target_den, a)
        if rn == 0:
            continue
        b_min, b_max_raw = bounds(rn, rd, 2, a)
        b_max = min(b_max_raw, best_max - 1)
        for b in range(b_min, b_max + 1):
            cn, cd = sub_recip(rn, rd, b)
            if cn == 0:
                continue
            if cn != 1:
                continue
            c = cd
            if c <= b or c >= best_max:
                continue
            best_max = c
            best = [a, b, c]
    
    # 4-term search
    for a in range(7, min(25, best_max)):
        if time.time() > deadline:
            return best
        rn, rd = sub_recip(target_num, target_den, a)
        if rn == 0:
            continue
        b_min, b_max_raw = bounds(rn, rd, 3, a)
        b_max = min(b_max_raw, best_max - 2)
        for b in range(b_min, b_max + 1):
            cn, cd = sub_recip(rn, rd, b)
            if cn == 0:
                continue
            c_min, c_max_raw = bounds(cn, cd, 2, b)
            c_max = min(c_max_raw, best_max - 1)
            for c in range(c_min, c_max + 1):
                dn, dd = sub_recip(cn, cd, c)
                if dn == 0:
                    continue
                if dn != 1:
                    continue
                d = dd
                if d <= c or d >= best_max:
                    continue
                best_max = d
                best = [a, b, c, d]
    
    # 5-term search
    for a in range(7, min(31, best_max)):
        if time.time() > deadline:
            return best
        rn, rd = sub_recip(target_num, target_den, a)
        if rn == 0:
            continue
        b_min, b_max_raw = bounds(rn, rd, 4, a)
        b_max = min(b_max_raw, best_max - 3)
        for b in range(b_min, b_max + 1):
            cn, cd = sub_recip(rn, rd, b)
            if cn == 0:
                continue
            c_min, c_max_raw = bounds(cn, cd, 3, b)
            c_max = min(c_max_raw, best_max - 2)
            for c in range(c_min, c_max + 1):
                dn, dd = sub_recip(cn, cd, c)
                if dn == 0:
                    continue
                d_min, d_max_raw = bounds(dn, dd, 2, c)
                d_max = min(d_max_raw, best_max - 1)
                for d in range(d_min, d_max + 1):
                    en, ed = sub_recip(dn, dd, d)
                    if en == 0:
                        continue
                    if en != 1:
                        continue
                    e = ed
                    if e <= d or e >= best_max:
                        continue
                    best_max = e
                    best = [a, b, c, d, e]
    
    # 6-term search
    for a in range(7, min(37, best_max)):
        if time.time() > deadline:
            return best
        rn, rd = sub_recip(target_num, target_den, a)
        if rn == 0:
            continue
        b_min, b_max_raw = bounds(rn, rd, 5, a)
        b_max = min(b_max_raw, best_max - 4)
        for b in range(b_min, b_max + 1):
            cn, cd = sub_recip(rn, rd, b)
            if cn == 0:
                continue
            c_min, c_max_raw = bounds(cn, cd, 4, b)
            c_max = min(c_max_raw, best_max - 3)
            for c in range(c_min, c_max + 1):
                dn, dd = sub_recip(cn, cd, c)
                if dn == 0:
                    continue
                d_min, d_max_raw = bounds(dn, dd, 3, d if False else c)
                d_max = min(d_max_raw, best_max - 2)
                for d in range(d_min, d_max + 1):
                    en, ed = sub_recip(dn, dd, d)
                    if en == 0:
                        continue
                    e_min, e_max_raw = bounds(en, ed, 2, d)
                    e_max = min(e_max_raw, best_max - 1)
                    for e in range(e_min, e_max + 1):
                        fn, fd = sub_recip(en, ed, e)
                        if fn == 0:
                            continue
                        if fn != 1:
                            continue
                        f = fd
                        if f <= e or f >= best_max:
                            continue
                        best_max = f
                        best = [a, b, c, d, e, f]
    
    return best
# EVOLVE-BLOCK-END
