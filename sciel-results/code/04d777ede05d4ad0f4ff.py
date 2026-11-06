import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 55.0

    # Known baseline solution: 8 + 28 + 1736 -> max 1736, score 8/1736 ≈ 0.0046 -> 9.33
    best_solution = [8, 28, 1736]
    best_max_denom = 1736

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
        return s == Fraction(5, 31)

    # For 2-term: 1/a + 1/b = N/D
    # => b = D*k / (N*k - D) for k = 1, 2, ...
    # We need N*k - D > 0, so k > D/N
    # a = D*k / (N*k - D) ... wait, let me re-derive.
    # 1/a + 1/b = N/D => (a+b)/(ab) = N/D => D(a+b) = Nab => Na b - D b = Da => b(Na - D) = Da => b = Da/(Na-D)
    # So for each a, b = Da/(Na - D). We need Na > D and (Na - D) | Da.
    # Let g = gcd(N, D). N' = N/g, D' = D/g.
    # Na - D = g(N'a - D'). b = D'a / (N'a - D'). We need (N'a - D') | D'a.
    # Since gcd(N', D') = 1, gcd(N'a - D', D') = gcd(N'a, D') = 1 (since gcd(N',D')=1).
    # Wait, that's not right. gcd(N'a - D', D') = gcd(N'a, D') = gcd(N', D') * a ... no.
    # gcd(N'a - D', D') = gcd(N'a, D') since D' | D'. And gcd(N'a, D') divides gcd(N', D') = 1.
    # So gcd(N'a - D', D') = 1. Therefore (N'a - D') | D'a implies (N'a - D') | a.
    # Let a = t*(N'a - D') + r ... actually, (N'a - D') | a means a = m*(N'a - D') for some integer m.
    # Wait, that's not right either. Let me reconsider.
    # We need (N'a - D') | D'a. Since gcd(N'a - D', D') = 1, we need (N'a - D') | a.
    # Let h = N'a - D'. Then h | a and h = N'a - D'.
    # a = h * s for some positive integer s.
    # h = N' * h * s - D' => h(1 - N'*s) = -D' => h = D'/(N'*s - 1).
    # So N'*s - 1 must divide D'.
    # For each s such that N'*s - 1 | D', we get h = D'/(N'*s - 1), a = h*s, b = D'*a/h = D'*s.
    # Wait, let me recheck: b = D'*a/(N'a - D') = D'*(h*s)/h = D'*s.
    # And a = h*s = D'*s/(N'*s - 1).
    # So the pair is (a, b) = (D'*s/(N'*s - 1), D'*s) where N'*s - 1 | D'.
    
    # For 3-term: 1/a + 1/b + 1/c = 5/31
    # Fix a. Then 1/b + 1/c = 5/31 - 1/a = (5a - 31)/(31a).
    # Let N = 5a - 31, D = 31a. Need N > 0, so a >= 7.
    # Reduce: g = gcd(N, D). N' = N/g, D' = D/g.
    # Then 1/b + 1/c = N'/D'.
    # Using the 2-term formula: for each s such that N'*s - 1 | D',
    # b = D'*s/(N'*s - 1), c = D'*s.
    # We need b and c to be positive integers, distinct from a, and distinct from each other.

    # First, check for 2-term solutions (unlikely to be optimal but easy)
    # 1/a + 1/b = 5/31. N=5, D=31. gcd(5,31)=1. N'=5, D'=31.
    # Need 5*s - 1 | 31. 5s-1 divides 31, so 5s-1 is in {1, 31}.
    # 5s-1=1 => s=0.4 (not integer). 5s-1=31 => s=32/5 (not integer).
    # No 2-term solution.

    # Check 1-term: 1/a = 5/31 => a = 31/5 (not integer). No 1-term solution.

    # Search 3-term solutions
    # a ranges from 7 up to some bound. Since we want to minimize max(a,b,c),
    # and a is the smallest, max >= a. If a >= best_max_denom, stop.
    # Also, b and c >= a (we can assume a <= b <= c without loss of generality by ordering).
    # Actually, we should enumerate a, then for each a find (b,c) with b >= a, c >= b.
    
    for a in range(7, best_max_denom):
        if time.time() - start_time > time_limit:
            break
        
        N = 5 * a - 31
        D = 31 * a
        if N <= 0:
            continue
        
        g = gcd(N, D)
        Np = N // g
        Dp = D // g
        
        # Find all s such that Np*s - 1 | Dp
        # Np*s - 1 is a positive divisor of Dp
        # Np*s - 1 >= 1 => s >= 2/Np => s >= 1
        # Dp divisors: iterate through divisors of Dp
        # For efficiency, factorize Dp or just iterate divisors up to sqrt(Dp)
        
        # Find all divisors of Dp
        # Dp = D/g = 31*a/gcd(5a-31, 31a)
        # Let's just iterate through potential s values.
        # Np*s - 1 | Dp means Np*s - 1 = d where d | Dp.
        # So s = (d+1)/Np must be a positive integer.
        # Iterate through divisors d of Dp.
        
        # For performance, let's factorize Dp
        # But Dp can be large. Let's just iterate d from 1 to sqrt(Dp)
        # Actually, the number of divisors is small. Let's find them.
        
        # Find divisors of Dp
        divisors = []
        i = 1
        while i * i <= Dp:
            if Dp % i == 0:
                divisors.append(i)
                if i != Dp // i:
                    divisors.append(Dp // i)
            i += 1
        
        for d in divisors:
            # d = Np*s - 1 => s = (d+1)/Np
            if (d + 1) % Np != 0:
                continue
            s = (d + 1) // Np
            if s < 1:
                continue
            
            # b = Dp*s / (Np*s - 1) = Dp*s/d
            # c = Dp*s
            if (Dp * s) % d != 0:
                continue
            b = (Dp * s) // d
            c = Dp * s
            
            # Check distinctness and bounds
            if b <= 0 or c <= 0:
                continue
            if a == b or a == c or b == c:
                continue
            if b > 10000000 or c > 10000000:
                continue
            
            # We want b >= a and c >= b for ordering (to avoid duplicates)
            # Actually, we just need them distinct. Let's just check max.
            mx = max(a, b, c)
            if mx < best_max_denom:
                # Verify
                sol = [a, b, c]
                if verify(sol):
                    best_max_denom = mx
                    best_solution = sol
    
    # Also try 4-term and more? The 3-term search should be sufficient if optimal is 3-term.
    # But let's also do a quick DFS for up to 5 terms with tight bounds if time allows.
    
    if time.time() - start_time < time_limit - 5.0:
        # DFS for solutions with max < best_max_denom
        def dfs(num, den, min_d, current):
            nonlocal best_solution, best_max_denom
            
            if time.time() - start_time > time_limit:
                return
            
            if num == 0:
                if max(current) < best_max_denom:
                    best_max_denom = max(current)
                    best_solution = list(current)
                return
            
            if num < 0:
                return
            
            # Lower bound for next denominator
            lower_bound = max(min_d, (den + num - 1) // num)
            if lower_bound >= best_max_denom:
                return
            
            # Upper bound: we need at least 1 more term
            # 1/d <= num/den => d >= den/num (already handled)
            # Also, if we use d, remainder must be formable by >= 1 term with denominators > d
            # Simple: d < best_max_denom
            
            # Additional pruning: if we have enough remaining capacity
            # For now, just iterate
            for d in range(lower_bound, best_max_denom):
                if time.time() - start_time > time_limit:
                    return
                if den > num * d:
                    continue
                
                next_num = num * d - den
                next_den = den * d
                g2 = gcd(next_num, next_den)
                if g2 > 1:
                    next_num //= g2
                    next_den //= g2
                
                if next_num < 0:
                    continue
                
                dfs(next_num, next_den, d + 1, current + [d])
        
        dfs(5, 31, 1, [])
    
    if verify(best_solution):
        return best_solution
    
    return [8, 28, 1736]
