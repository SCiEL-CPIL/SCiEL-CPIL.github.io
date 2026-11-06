"""Find distinct denominators summing reciprocally to 5/31, minimizing the largest denominator."""
import time
from fractions import Fraction
from math import isqrt

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 60s kill

    TARGET = Fraction(5, 31)

    def valid_solution(denoms):
        """Check distinct, in range, sum to target."""
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        s = set(denoms)
        if len(s) != len(denoms):
            return False
        for d in denoms:
            if d < 1 or d > 10000000:
                return False
        total = Fraction(0, 1)
        for d in denoms:
            total += Fraction(1, d)
        return total == TARGET

    # Greedy solution as fallback
    def greedy_solution(target):
        denoms = []
        r = target
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d <= 0:
                d = 1
            if r.denominator % r.numerator == 0:
                # r is a unit fraction or integer
                d = r.denominator // r.numerator
            denoms.append(d)
            r -= Fraction(1, d)
            if len(denoms) > 256:
                break
        return denoms

    # Find all 2-term completions: 1/a + 1/b = r
    def two_term_completions(r, max_d):
        """Find pairs (a,b) with a < b, a,b <= max_d, 1/a + 1/b = r."""
        if r <= 0 or r >= 1:
            return []
        results = []
        n, d = r.numerator, r.denominator
        # 1/a + 1/b = n/d => d(a+b) = n*a*b => n*a*b - d*a - d*b = 0
        # (n*a - d)(n*b - d) = d^2
        # Let x = n*a - d, y = n*b - d, then x*y = d^2, a = (x+d)/n, b = (y+d)/n
        # Need x | d^2, y = d^2/x, a = (x+d)/n must be positive integer, same for b
        # Also need a < b => x < y
        d2 = d * d
        # x must be a divisor of d^2
        # a = (x + d) / n > 0 => x > -d, and a >= 1 => x >= n - d
        # b = (d^2/x + d) / n > 0 => d^2/x > -d => x > -d (same)
        # b >= 1 => d^2/x + d >= n => d^2/x >= n - d => x <= d^2 / (n - d) if n > d
        # a <= max_d => (x+d)/n <= max_d => x <= n*max_d - d
        # b <= max_d => (d^2/x + d)/n <= max_d => d^2/x <= n*max_d - d => x >= d^2 / (n*max_d - d)
        
        # Iterate over divisors x of d^2
        # x can be negative too, but a > 0 requires x > -d, and since a >= 1, x >= n - d
        # For practical purposes, x > 0 typically
        # Let's iterate x from 1 to sqrt(d^2) and check both x and d^2//x
        
        for i in range(1, isqrt(d2) + 1):
            if d2 % i == 0:
                for x in [i, d2 // i]:
                    if x <= 0:
                        continue
                    a_num = x + d
                    if a_num % n != 0:
                        continue
                    a = a_num // n
                    if a < 1 or a > max_d:
                        continue
                    y = d2 // x
                    b_num = y + d
                    if b_num % n != 0:
                        continue
                    b = b_num // n
                    if b < 1 or b > max_d:
                        continue
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    # Verify
                    if Fraction(1, a) + Fraction(1, b) == r:
                        results.append((a, b))
        return results

    # 1-term completion: just 1/d = r
    def one_term_completion(r, max_d):
        if r <= 0:
            return []
        if r.numerator == 1:
            d = r.denominator
            if 1 <= d <= max_d:
                return [(d,)]
        return []

    # 3-term completion via direct search
    def three_term_completions(r, max_d):
        """Find triples (a,b,c) with a<b<c, all <= max_d, 1/a+1/b+1/c = r."""
        if r <= 0 or r >= 1:
            return []
        results = []
        n, d = r.numerator, r.denominator
        
        # For each a, find 2-term completions for r - 1/a
        # a must satisfy: 1/a < r => a > 1/r = d/n
        # Also 3/a >= r (since b,c > a, 1/b + 1/c < 2/a, so 1/a + 2/a > r => 3/a > r => a < 3/r)
        # Actually: 1/a + 1/b + 1/c = r with a<b<c implies 1/a < r < 3/a
        # So a > 1/r and a < 3/r
        a_min = (d + n) // n + 1  # a > d/n
        a_max = (3 * d) // n  # a < 3d/n
        
        # Also need a >= 2 (since b > a >= 1, and we want distinct)
        a_min = max(a_min, 2)
        
        for a in range(a_min, min(a_max + 1, max_d + 1)):
            r2 = r - Fraction(1, a)
            if r2 <= 0:
                continue
            # Find (b,c) with b > a, b < c, b,c <= max_d, 1/b + 1/c = r2
            pairs = two_term_completions(r2, max_d)
            for b, c in pairs:
                if b > a and b < c:
                    results.append((a, b, c))
        return results

    # Try to find solutions with decreasing maximum denominator
    # Start with a reasonable upper bound and work down
    # The greedy solution gives us an initial bound
    greedy = greedy_solution(TARGET)
    if greedy and valid_solution(greedy):
        best = greedy
        best_max = max(greedy)
    else:
        best = [8]
        r = TARGET - Fraction(1, 8)
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            best.append(d)
            r -= Fraction(1, d)
        best_max = max(best)

    # Try to improve by searching for solutions with smaller max denominator
    # Search K from best_max - 1 downward
    # For each K, try to find any valid solution with all denominators <= K
    
    # Strategy: try small number of terms (2, 3, 4, 5) first since they're faster to verify
    # For a given K, check if there's a solution with max denom <= K
    
    for K in range(min(best_max - 1, 100), 5, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Try 2 terms
        if time.time() > deadline - 5:
            break
        pairs = two_term_completions(TARGET, K)
        if pairs:
            a, b = pairs[0]
            candidate = [a, b]
            if valid_solution(candidate):
                best = candidate
                best_max = K
                found = True
                break
        
        # Try 3 terms
        if not found and time.time() > deadline - 5:
            break
        # Limit search for 3-term
        triples = three_term_completions(TARGET, K)
        if triples:
            a, b, c = triples[0]
            candidate = [a, b, c]
            if valid_solution(candidate):
                best = candidate
                best_max = K
                found = True
                break
        
        if time.time() > deadline - 10:
            break
    
    # Verify final solution
    if not valid_solution(best):
        # Fallback to a known working solution
        best = [8]
        r = TARGET - Fraction(1, 8)
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            best.append(d)
            r -= Fraction(1, d)
    
    return best
