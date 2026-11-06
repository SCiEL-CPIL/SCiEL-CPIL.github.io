def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 80
    
    target = Fraction(5, 31)
    best_solution = [7, 56, 1736]
    best_max = 1736
    
    path = []
    used = set()
    
    def solve_two(remaining, last_d, max_den):
        p, q = remaining.numerator, remaining.denominator
        q2 = q * q
        best = None
        max_f = min(q - 1, 200000)
        for f in range(1, max_f + 1):
            if q2 % f != 0:
                continue
            g = q2 // f
            if g <= f:
                continue
            num_a = f + q
            num_b = g + q
            if num_a % p != 0 or num_b % p != 0:
                continue
            a = num_a // p
            b = num_b // p
            if a <= last_d or b > max_den:
                continue
            if a in used or b in used:
                continue
            if a == b:
                continue
            if best is None or max(a, b) < max(best[0], best[1]):
                best = (a, b)
        return best
    
    def dfs(remaining, last_d, max_den, depth):
        nonlocal best_solution, best_max
        if time.time() > deadline:
            return None
        if remaining == 0:
            return list(path)
        if depth >= 256:
            return None
        if remaining.numerator == 1:
            d = remaining.denominator
            if d > last_d and d <= max_den and d not in used:
                return path + [d]
            return None
        if remaining.numerator == 2:
            pair = solve_two(remaining, last_d, max_den)
            if pair:
                a, b = pair
                return path + [a, b]
            return None
        
        min_sum = Fraction(0, 1)
        for i in range(1, remaining.numerator + 1):
            min_sum += Fraction(1, last_d + i)
        if remaining < min_sum:
            return None
        
        d_min = last_d + 1
        d_lower = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        d_min = max(d_min, d_lower)
        
        d_max = min(max_den, d_min + 2000)
        
        for d in range(d_min, d_max + 1):
            if d in used:
                continue
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            path.append(d)
            used.add(d)
            result = dfs(new_rem, d, max_den, depth + 1)
            used.remove(d)
            path.pop()
            if result is not None:
                return result
        return None
    
    for M in range(7, best_max):
        if time.time() > deadline:
            break
        path.clear()
        used.clear()
        result = dfs(target, 0, M, 0)
        if result is not None:
            best_solution = result
            best_max = M
            break
    
    return best_solution
