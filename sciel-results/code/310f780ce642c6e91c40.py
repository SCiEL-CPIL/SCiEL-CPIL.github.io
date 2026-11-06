import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80

    def find_last_two(r, prev_d):
        """
        Find two distinct integers d7, d8 > prev_d such that 1/d7 + 1/d8 = r.
        Returns (d7, d8) or None.
        """
        a, b = r.numerator, r.denominator
        if a <= 0 or b <= 0:
            return None
        # 1/d7 + 1/d8 = a/b
        # (d7 + d8) / (d7 * d8) = a/b
        # b * (d7 + d8) = a * d7 * d8
        # a * d7 * d8 - b * d7 - b * d8 = 0
        # (a * d7 - b) * (a * d8 - b) = b^2
        # Let x = a * d7 - b, y = a * d8 - b. Then x * y = b^2.
        # d7 = (x + b) / a, d8 = (y + b) / a.
        # We need d7, d8 to be integers > prev_d.
        # So x + b must be divisible by a, and y + b must be divisible by a.
        # Also x < y (assuming d7 < d8).
        
        # Iterate over divisors x of b^2.
        # b can be large, so b^2 is very large. This might be slow.
        # Alternative: iterate d7.
        # 1/d8 = a/b - 1/d7 = (a * d7 - b) / (b * d7)
        # d8 = b * d7 / (a * d7 - b)
        # Let num = a * d7 - b.
        # We need num > 0 and num divides b * d7.
        # min_d7: 1/d7 < a/b => d7 > b/a.
        # max_d7: 2/d7 > a/b => d7 < 2*b/a.
        
        min_d7 = max(prev_d + 1, b // a + (1 if b % a else 0))
        max_d7 = min(10000000, 2 * b // a)
        
        for d7 in range(min_d7, max_d7 + 1):
            num = a * d7 - b
            if num <= 0:
                continue
            if (d7 * b) % num != 0:
                continue
            d8 = (d7 * b) // num
            if d8 <= d7 or d8 > 10000000:
                continue
            return d7, d8
        return None

    def dfs(n_terms, prev_d, current, remaining_frac):
        """
        Find n_terms distinct denominators > prev_d summing to remaining_frac.
        Returns list of denominators or None.
        """
        if time.time() > deadline:
            return None
            
        if n_terms == 2:
            return find_last_two(remaining_frac, prev_d)
        
        if n_terms < 2:
            return None
            
        # Pruning:
        # Max possible sum with n_terms terms is sum(1/(prev_d + 1 + i) for i in range(n_terms))
        max_possible = sum(Fraction(1, prev_d + 1 + i) for i in range(n_terms))
        if max_possible < remaining_frac:
            return None
            
        # Min possible sum is very small, usually not a constraint.
        
        # Range for the next denominator d
        # 1/d <= remaining_frac / n_terms  => d >= n_terms / remaining_frac
        # 1/d >= remaining_frac / 1 (if n_terms=1, but we handle n_terms>=2)
        # Actually, 1/d must be small enough to leave room for the rest.
        # 1/d <= remaining_frac - sum(1/(d + 1 + i) for i in range(n_terms - 1))
        # This is hard to solve directly.
        
        # Simple bounds:
        # d must be at least prev_d + 1
        # d must be at most such that 1/d is not too small.
        # If 1/d is too small, the remaining n_terms-1 terms (which are even smaller) can't sum to the rest.
        # Max sum of remaining n_terms-1 terms is (n_terms-1)/(d+1).
        # So we need remaining_frac - 1/d <= (n_terms-1)/(d+1).
        # Let's just iterate d from min_d to max_d.
        
        min_d = max(prev_d + 1, remaining_frac.denominator // remaining_frac.numerator + (1 if remaining_frac.denominator % remaining_frac.numerator else 0))
        # max_d: 1/d >= remaining_frac / n_terms => d <= n_terms * remaining_frac.denominator // remaining_frac.numerator
        max_d = min(10000000, n_terms * remaining_frac.denominator // remaining_frac.numerator)
        
        if max_d < min_d:
            return None
            
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            new_rem = remaining_frac - Fraction(1, d)
            if new_rem <= 0:
                continue
            # Prune: check if remaining n_terms-1 terms can sum to new_rem
            # Max possible sum for n_terms-1 terms with denominators > d
            max_possible2 = sum(Fraction(1, d + 1 + i) for i in range(n_terms - 1))
            if max_possible2 < new_rem:
                continue
                
            current.append(d)
            result = dfs(n_terms - 1, d, current, new_rem)
            if result is not None:
                return [d] + result
            current.pop()
            
        return None

    # Search for minimal n
    for n in range(5, 20):
        if time.time() > deadline:
            break
        # Try to find n terms
        # Start with d1 from 5 upwards
        # We can optimize by trying d1 values that are more likely to work.
        # But for small n, the search space is small.
        for d1 in range(5, 50):
            if time.time() > deadline:
                break
            r_after_d1 = Fraction(1) - Fraction(1, d1)
            if r_after_d1 <= 0:
                continue
            # Check if n-1 more terms can sum to r_after_d1
            max_possible = sum(Fraction(1, d1 + 1 + i) for i in range(n - 1))
            if max_possible < r_after_d1:
                continue
                
            current = [d1]
            result = dfs(n - 1, d1, current, r_after_d1)
            if result is not None:
                return current + result

    # Fallback: use the DP approach from inspiration 1
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no baseline representation")
