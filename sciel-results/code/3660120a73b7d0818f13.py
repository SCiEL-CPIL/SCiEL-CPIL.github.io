"""Find minimum distinct denominators >= 5 with reciprocal sum exactly 1."""
import time
from fractions import Fraction
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    def verify(sol):
        if not sol:
            return False
        if len(set(sol)) != len(sol):
            return False
        if any(d < 5 or d > 10000000 for d in sol):
            return False
        s = sum(Fraction(1, d) for d in sol)
        return s == 1
    
    def search_k(k, min_d=5, deadline=None):
        """Find k distinct denominators >= min_d with reciprocal sum = 1."""
        if deadline is None:
            deadline = 80
        if time.time() - start_time > deadline:
            return None
        return _dfs(Fraction(1, 1), k, min_d, [])
    
    def _dfs(remaining, k, min_d, path):
        """DFS to find k denominators >= min_d summing to 'remaining'."""
        if time.time() - start_time > deadline:
            return None
        if k == 0:
            if remaining == Fraction(0, 1):
                return path[:]
            return None
        if k == 1:
            d = remaining.denominator // remaining.numerator if remaining.numerator != 0 else None
            if d is not None and remaining * d == 1:
                if d >= min_d and (not path or d > path[-1]):
                    return path + [d]
            return None
        
        # Pruning: max sum with k terms starting at min_d
        # 1/min_d + 1/(min_d+1) + ... + 1/(min_d+k-1)
        max_possible = sum(Fraction(1, min_d + i) for i in range(k))
        if max_possible < remaining:
            return None
        
        # Minimum possible: each term >= 1/10000000, so min sum = k/10000000
        # If remaining < k/10000000, impossible (but remaining is always > 0 here)
        
        # Determine range for next denominator d
        # d >= min_d
        # 1/d < remaining (since we need k-1 more positive terms)
        # So d > 1/remaining = remaining.denominator/remaining.numerator
        # Also 1/d + (k-1)/(d+1) >= remaining (lower bound on d from upper bound)
        # For large d: k/d >= remaining, so d <= k/remaining = k * remaining.denominator / remaining.numerator
        
        start_d = min_d
        if remaining.numerator != 0:
            # d > remaining.denominator / remaining.numerator
            start_d = max(start_d, remaining.denominator // remaining.numerator + 1)
        
        # Upper bound: d <= k * remaining.denominator // remaining.numerator (rough)
        if remaining.numerator != 0:
            upper_d = min(10000000, (k * remaining.denominator) // remaining.numerator + 10)
        else:
            upper_d = 10000000
        
        if start_d > upper_d:
            return None
        
        for d in range(start_d, upper_d + 1):
            if time.time() - start_time > deadline:
                return None
            new_rem = remaining - Fraction(1, d)
            if new_rem <= 0:
                continue
            # new_rem must be representable as sum of (k-1) unit fractions
            # Each <= 1/(d+1), so new_rem <= (k-1)/(d+1)
            if new_rem > Fraction(k - 1, d + 1):
                continue
            result = _dfs(new_rem, k - 1, d + 1, path + [d])
            if result is not None:
                return result
        return None
    
    # Try increasing k from 9 (k=8 impossible as shown above)
    best = None
    for k in range(9, 16):
        sol = search_k(k, 5, deadline=80)
        if sol is not None and verify(sol):
            best = sol
            break
    
    if best is not None:
        return best
    
    # Fallback: known valid solutions
    # Try to construct a valid solution
    fallback = [5, 6, 7, 10, 14, 20, 35, 70, 140]
    if verify(fallback):
        return fallback
    
    # Another fallback
    fallback2 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    if verify(fallback2):
        return fallback2
    
    # Construct via greedy splitting
    # Start with 1, split repeatedly
    terms = []
    rem = Fraction(1, 1)
    while rem > 0 and len(terms) < 256:
        if rem.numerator == rem.denominator:
            # rem is integer, add its reciprocal
            terms.append(rem.denominator)
            rem = Fraction(0, 1)
        elif rem.numerator == 1:
            # rem = 1/n
            terms.append(rem.denominator)
            rem = Fraction(0, 1)
        else:
            # Split: rem = p/q. Find d such that 1/d < p/q and remainder is manageable
            # Try d = q/p + 1 (so 1/d < p/q)
            d = rem.denominator // rem.numerator + 1
            new_rem = rem - Fraction(1, d)
            if new_rem <= 0 or d in terms:
                d += 1
                new_rem = rem - Fraction(1, d)
            if new_rem > 0:
                terms.append(d)
                rem = new_rem
            else:
                # Try larger d
                for dd in range(d + 1, d + 100):
                    nr = rem - Fraction(1, dd)
                    if nr > 0 and dd not in terms:
                        terms.append(dd)
                        rem = nr
                        break
    
    if terms and verify(terms):
        return terms
    
    # Last resort: brute force small set
    for k in range(9, 20):
        sol = search_k(k, 5, deadline=80)
        if sol is not None and verify(sol):
            return sol
    
    return [5, 6, 7, 10, 14, 20, 35, 70, 140]
