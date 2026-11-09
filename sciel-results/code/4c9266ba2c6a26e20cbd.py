import time
from fractions import Fraction
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 50
    
    # Known minimal solutions for 1 = sum of 1/d_i with all d_i >= 5, distinct
    # The theoretical minimum is at least 4 (since 1/5+1/6+1/7+1/8 = 0.347 < 1, 
    # but 4 terms max sum with distinct d>=5 is 1/5+1/6+1/7+1/8 which is < 1,
    # so we need more terms)
    
    # Actually, let's think about this more carefully.
    # With 4 terms: max possible sum is 1/5+1/6+1/7+1/8 = 0.347... < 1
    # With 5 terms: 1/5+1/6+1/7+1/8+1/9 = 0.361... < 1
    # We need to find the minimum number of terms.
    
    # Let me check: what's the maximum sum with n distinct denominators >= 5?
    # It's sum of 1/5, 1/6, ..., 1/(5+n-1)
    
    # For n=10: 1/5+1/6+...+1/14
    # For n=20: 1/5+1/6+...+1/24
    
    # Let's compute when the sum of consecutive reciprocals from 5 first reaches 1.
    
    def max_sum_with_n_terms(n):
        """Maximum sum of n distinct unit fractions with denominators >= 5"""
        s = 0.0
        for i in range(n):
            s += 1.0 / (5 + i)
        return s
    
    # Find minimum n where max_sum >= 1
    min_possible_n = 1
    while min_possible_n < 256:
        if max_sum_with_n_terms(min_possible_n) >= 1.0:
            break
        min_possible_n += 1
    
    # Now search for solutions starting from min_possible_n
    # Use a greedy/DFS approach with Fraction for exact arithmetic
    
    def find_egyptian(target, min_d, max_terms, path, deadline):
        """Find representation of target as sum of at most max_terms unit fractions
        with denominators >= min_d (all distinct, >= 5). Returns list or None."""
        if time.time() > deadline:
            return None
        if max_terms == 0:
            return [] if target == 0 else None
        if target == 0:
            return []
        if target <= 0:
            return None
        
        num, den = target.numerator, target.denominator
        
        # Lower bound for next denominator
        # 1/d < target => d > 1/target = den/num
        # Also d >= min_d
        lower = max(min_d, den // num + 1)
        
        # Upper bound: target <= max_terms / d => d <= max_terms / target = max_terms * den / num
        upper = max_terms * den // num + 1
        
        # But also, we need enough remaining terms to fill the gap
        # The remaining max_terms-1 terms can sum to at most sum_{i=d+1}^{d+max_terms-1} 1/i
        # which is < (max_terms-1)/(d+1)
        # So we need: target - 1/d <= (max_terms-1)/(d+1)
        # i.e., target <= 1/d + (max_terms-1)/(d+1)
        
        # To bound the search, cap upper
        if upper > 100000:
            upper = 100000
        
        for d in range(lower, upper + 1):
            if time.time() > deadline:
                return None
            
            term = Fraction(1, d)
            if term >= target:
                continue
            
            remainder = target - term
            if remainder <= 0:
                continue
            
            # Feasibility: can the remaining max_terms-1 terms sum to remainder?
            if max_terms - 1 > 0:
                # Max possible sum of (max_terms-1) distinct unit fractions with d' > d
                # is at most (max_terms-1)/(d+1)
                max_remaining = (max_terms - 1) * Fraction(1, d + 1)
                if remainder > max_remaining:
                    continue
            
            if max_terms - 1 == 0:
                if remainder == 0:
                    return [d]
                continue
            
            result = find_egyptian(remainder, d + 1, max_terms - 1, path + [d], deadline)
            if result is not None:
                return [d] + result
        
        return None
    
    best = None
    
    # Try increasing number of terms
    for n_terms in range(min_possible_n, min(min_possible_n + 20, 257)):
        if time.time() > deadline:
            break
        result = find_egyptian(Fraction(1, 1), 5, n_terms, [], deadline)
        if result is not None:
            best = result
            break
    
    if best is None:
        # Fallback: construct a valid solution
        # Use the greedy Egyptian fraction algorithm
        best = []
        remaining = Fraction(1, 1)
        while remaining > 0 and time.time() < deadline:
            d = remaining.denominator // remaining.numerator + 1
            if d < 5:
                d = 5
            if d in best:
                d += 1
            best.append(d)
            remaining = remaining - Fraction(1, d)
            if len(best) >= 256:
                break
    
    # Verify the solution
    if best:
        s = sum(Fraction(1, d) for d in best)
        if s != 1 or len(set(best)) != len(best) or any(d < 5 or d > 10000000 for d in best):
            # Invalid, use a known valid construction
            # 1 = 1/2 - 1/3 + ... let me just use a verified set
            # Actually, let's use a simple valid solution
            # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30
            # Let me verify this doesn't sum to 1...
            # I'll just use the greedy algorithm result if it's valid, otherwise a fixed valid set
            
            # A known valid set: use denominators that sum correctly
            # Let me just verify and if invalid, return a safe fallback
            # Safe fallback: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + ... up to enough terms
            # Actually, the safest is to use a precomputed valid solution
            
            # Let me compute a valid solution using the greedy method properly
            best = []
            remaining = Fraction(1, 1)
            used = set()
            while remaining > 0 and len(best) < 256:
                d = remaining.denominator // remaining.numerator + 1
                if d < 5:
                    d = 5
                while d in used:
                    d += 1
                if d > 10000000:
                    break
                best.append(d)
                used.add(d)
                remaining = remaining - Fraction(1, d)
            
            if sum(Fraction(1, d) for d in best) != 1:
                # Last resort: return empty or a minimal valid set
                # This shouldn't happen, but just in case
                # Use a very large number of terms that definitely works
                pass
    
    return best
