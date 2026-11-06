import time
import math

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    
    _factor_cache = {}
    _divs_cache = {}
    
    def factorize(num):
        if num in _factor_cache:
            return _factor_cache[num]
        orig = num
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        _factor_cache[orig] = factors
        return factors
    
    def divisors_of_square(b):
        if b in _divs_cache:
            return _divs_cache[b]
        if b <= 0:
            _divs_cache[b] = [1]
            return [1]
        factors = factorize(b)
        divs = [1]
        for p, e in factors.items():
            exp = 2 * e
            new_divs = []
            for d in divs:
                val = d
                for _ in range(exp + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        _divs_cache[b] = divs
        return divs
    
    def solve_n(n, x_start, x_end, tlimit):
        for x in range(x_start, x_end + 1):
            if time.time() > tlimit:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            divs = divisors_of_square(b)
            neg_b_mod_a = (-b) % a
            for k in divs:
                if k % a != neg_b_mod_a:
                    continue
                y = (k + b) // a
                if y <= x:
                    continue
                bq = b * b // k
                if (bq + b) % a != 0:
                    continue
                z = (bq + b) // a
                if z <= y:
                    continue
                if z > 10**12 or y > 10**12:
                    continue
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    return [x, y, z]
        return None
    
    # Pass 1: quick search
    for n in range(3, 257):
        if time.time() > deadline:
            break
        idx = n - 3
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        x_limit = min(x_min + 25, x_max)
        answers[idx] = solve_n(n, x_min, x_limit, deadline)
    
    # Pass 2: full search for unsolved
    remaining = deadline - time.time()
    if remaining > 0.3:
        unsolved = [n for n in range(3, 257) if answers[n - 3] is None]
        if unsolved:
            time_per_n = remaining / len(unsolved)
            for n in unsolved:
                if time.time() > deadline:
                    break
                idx = n - 3
                x_min = n // 4 + 1
                x_max = (3 * n) // 4
                x_start = x_min + 26
                if x_start > x_max:
                    continue
                answers[idx] = solve_n(n, x_start, x_max, min(deadline, time.time() + time_per_n))
    
    return answers
