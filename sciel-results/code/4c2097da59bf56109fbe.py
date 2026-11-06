"""Rewritten program: Dijkstra-style search over rational remainders to minimize max denominator."""
import time
from fractions import Fraction
import heapq

def construct():
    deadline = time.time() + 55
    target = Fraction(5, 31)
    
    # We know a 3-term solution: [8, 31, 248] with max=248
    # Let's verify: 1/8 + 1/31 + 1/248 = 31/248 + 8/248 + 1/248 = 40/248 = 5/31. Yes!
    best_solution = [8, 31, 248]
    best_max = 248
    
    def verify(denoms):
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        return s == target
    
    # Dijkstra-like search: state = (current_max_denominator, number_of_terms, remainder, path)
    # We want to minimize the maximum denominator used.
    # 
    # Key insight: For Egyptian fractions, if we're looking for a solution with max denominator <= M,
    # we can try all possible first terms a in range(ceil(1/target), M+1) and recurse.
    # 
    # Better approach: BFS/DFS with pruning. Since we want to minimize max denominator,
    # we can binary search on M (the max denominator) and check if a solution exists with all
    # denominators <= M.
    #
    # Actually, let's think differently. The optimal solution likely has a small number of terms.
    # Let's try to find solutions with 2, 3, 4, 5 terms where the max denominator is minimized.
    #
    # For 2 terms: 1/a + 1/b = 5/31 => 31(a+b) = 5ab => 31a + 31b = 5ab
    # => 5ab - 31a - 31b = 0 => (5a - 31)(5b - 31) = 31^2 = 961
    # 961 = 1*961 = 31*31
    # 5a-31=1 => a=32/5 not integer
    # 5a-31=31 => a=12.4 not integer
    # 5a-31=961 => a=198.4 not integer
    # So no 2-term solution.
    #
    # For 3 terms, we already have [8,31,248]. Can we do better?
    # Let's search for 3-term solutions with max denominator < 248.
    #
    # 1/a + 1/b + 1/c = 5/31, a < b < c, c < 248
    # a must be >= ceil(31/5) = 7
    # For each a, 1/b + 1/c = 5/31 - 1/a = (5a-31)/(31a)
    # => 31a(b+c) = (5a-31)bc
    # => (5a-31)bc - 31ab - 31ac = 0
    # => ((5a-31)b - 31a)((5a-31)c - 31a) = (31a)^2
    #
    # Let's enumerate:
    
    # 3-term search with tighter bounds
    for a in range(7, 248):
        if time.time() > deadline:
            break
        num_a = 5 * a - 31
        if num_a <= 0:
            continue
        den_a = 31 * a
        # 1/b + 1/c = num_a / den_a
        # b > a, c > b
        # b must satisfy: b > a and b < 2*den_a/num_a
        b_min = a + 1
        b_max = min(2 * den_a // num_a, best_max - 1)
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            # 1/c = num_a/den_a - 1/b = (num_a*b - den_a) / (den_a*b)
            num_c = num_a * b - den_a
            den_c = den_a * b
            if num_c <= 0:
                continue
            if den_c % num_c != 0:
                continue
            c = den_c // num_c
            if c <= b:
                continue
            if c >= best_max:
                continue
            if verify([a, b, c]):
                if c < best_max:
                    best_max = c
                    best_solution = [a, b, c]
    
    # 4-term search - optimized with tighter bounds
    for a in range(7, min(30, best_max)):
        if time.time() > deadline:
            break
        num_a = 5 * a - 31
        if num_a <= 0:
            continue
        den_a = 31 * a
        b_min = max(a + 1, den_a // num_a + 1)
        b_max = min(3 * den_a // num_a, best_max - 1)
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            num_b = num_a * b - den_a
            den_b = den_a * b
            if num_b <= 0:
                continue
            c_min = max(b + 1, den_b // num_b + 1)
            c_max = min(2 * den_b // num_b, best_max - 1)
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                num_c = num_b * c - den_b
                den_c = den_b * c
                if num_c <= 0:
                    continue
                if den_c % num_c != 0:
                    continue
                d = den_c // num_c
                if d <= c:
                    continue
                if d >= best_max:
                    continue
                if verify([a, b, c, d]):
                    if d < best_max:
                        best_max = d
                        best_solution = [a, b, c, d]
    
    # 5-term search with tighter bounds
    for a in range(7, min(20, best_max)):
        if time.time() > deadline:
            break
        num_a = 5 * a - 31
        if num_a <= 0:
            continue
        den_a = 31 * a
        b_min = max(a + 1, den_a // num_a + 1)
        b_max = min(4 * den_a // num_a, best_max - 1)
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            num_b = num_a * b - den_a
            den_b = den_a * b
            if num_b <= 0:
                continue
            c_min = max(b + 1, den_b // num_b + 1)
            c_max = min(3 * den_b // num_b, best_max - 1)
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                num_c = num_b * c - den_b
                den_c = den_b * c
                if num_c <= 0:
                    continue
                d_min = max(c + 1, den_c // num_c + 1)
                d_max = min(2 * den_c // num_c, best_max - 1)
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        break
                    num_d = num_c * d - den_c
                    den_d = den_c * d
                    if num_d <= 0:
                        continue
                    if den_d % num_d != 0:
                        continue
                    e = den_d // num_d
                    if e <= d:
                        continue
                    if e >= best_max:
                        continue
                    if verify([a, b, c, d, e]):
                        if e < best_max:
                            best_max = e
                            best_solution = [a, b, c, d, e]
    
    # 6-term search with very tight bounds
    for a in range(7, min(15, best_max)):
        if time.time() > deadline:
            break
        num_a = 5 * a - 31
        if num_a <= 0:
            continue
        den_a = 31 * a
        b_min = max(a + 1, den_a // num_a + 1)
        b_max = min(5 * den_a // num_a, best_max - 1)
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            num_b = num_a * b - den_a
            den_b = den_a * b
            if num_b <= 0:
                continue
            c_min = max(b + 1, den_b // num_b + 1)
            c_max = min(4 * den_b // num_b, best_max - 1)
            for c in range(c_min, c_max + 1):
                if time.time() > deadline:
                    break
                num_c = num_b * c - den_b
                den_c = den_b * c
                if num_c <= 0:
                    continue
                d_min = max(c + 1, den_c // num_c + 1)
                d_max = min(3 * den_c // num_c, best_max - 1)
                for d in range(d_min, d_max + 1):
                    if time.time() > deadline:
                        break
                    num_d = num_c * d - den_c
                    den_d = den_c * d
                    if num_d <= 0:
                        continue
                    e_min = max(d + 1, den_d // num_d + 1)
                    e_max = min(2 * den_d // num_d, best_max - 1)
                    for e in range(e_min, e_max + 1):
                        if time.time() > deadline:
                            break
                        num_e = num_d * e - den_d
                        den_e = den_d * e
                        if num_e <= 0:
                            continue
                        if den_e % num_e != 0:
                            continue
                        f = den_e // num_e
                        if f <= e:
                            continue
                        if f >= best_max:
                            continue
                        if verify([a, b, c, d, e, f]):
                            if f < best_max:
                                best_max = f
                                best_solution = [a, b, c, d, e, f]
    
    return best_solution
