from fractions import Fraction
import time

def construct():
    target = Fraction(5, 31)
    
    def check_m3(K):
        for d1 in range(7, K + 1):
            r1 = target - Fraction(1, d1)
            if r1 <= 0:
                break
            for d2 in range(d1 + 1, K + 1):
                r2 = r1 - Fraction(1, d2)
                if r2 <= 0:
                    break
                if r2.numerator == 1:
                    d3 = r2.denominator
                    if d3 > d2 and d3 <= K:
                        return [d1, d2, d3]
                else:
                    if r2.denominator % r2.numerator == 0:
                        d3 = r2.denominator // r2.numerator
                        if d3 > d2 and d3 <= K:
                            return [d1, d2, d3]
        return None

    def check_dfs(K, max_terms=10):
        def dfs(S, last, depth, path):
            if S == 0:
                return path
            if depth == max_terms:
                return None
            
            rem_terms = max_terms - depth
            max_possible = Fraction(0, 1)
            for i in range(rem_terms):
                max_possible += Fraction(1, last + 1 + i)
            if S > max_possible:
                return None
                
            min_possible = Fraction(0, 1)
            for i in range(rem_terms):
                min_possible += Fraction(1, K - i)
            if S < min_possible:
                return None

            min_d = max(last + 1, (S.denominator + S.numerator - 1) // S.numerator)
            
            for d in range(min_d, K + 1):
                if Fraction(1, d) > S:
                    continue
                res = dfs(S - Fraction(1, d), d, depth + 1, path + [d])
                if res:
                    return res
            return None
            
        return dfs(target, 0, 0, [])

    start_time = time.time()
    # We know [7, 62, 434] is a solution.
    # Check K from 20 to 433.
    for K in range(20, 434):
        if time.time() - start_time > 80:
            break
        res = check_m3(K)
        if res:
            return res
        res = check_dfs(K)
        if res:
            return res
            
    return [7, 62, 434]
