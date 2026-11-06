"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    # Greedy Egyptian fraction with strictly increasing denominators >= 5
    # We want the fewest terms such that sum of 1/d_i = 1, d_i distinct, d_i >= 5.
    
    # The minimum number of terms: with denominators >= 5, the maximum sum with n terms
    # using the smallest possible distinct denominators 5,6,7,...,n+4 is sum_{k=5}^{n+4} 1/k.
    # For n=2: 1/5 + 1/6 = 11/30 < 1. So we need at least 3 terms.
    # For n=3: 1/5 + 1/6 + 1/7 = 41/210 ≈ 0.195... wait that's way less than 1.
    # Actually we need the SUM to be exactly 1. With large denominators, terms are small.
    # Let's think differently: we want sum of 1/d_i = 1.
    # 
    # For 3 terms: 1/a + 1/b + 1/c = 1 with 5 <= a < b < c.
    # 1/a < 1, so a >= 2, but a >= 5. 
    # 1/5 + 1/b + 1/c = 1 => 1/b + 1/c = 4/5
    # 1/b < 4/5 => b > 5/4 = 1.25, so b >= 2, but b > a = 5, so b >= 6.
    # 1/b + 1/c = 4/5 => (b+c)/(bc) = 4/5 => 5(b+c) = 4bc => 4bc - 5b - 5c = 0
    # (4b-5)(4c-5) = 25/4... let me redo: multiply by 4: 16bc - 20b - 20c = 0
    # (4b-5)(4c-5) = 25. Since b,c > 5, 4b-5 > 15, 4c-5 > 15, product > 225 > 25. No solution.
    # 
    # Try a=5: 1/b + 1/c = 4/5. As shown, no integer solution with b>5.
    # Try a=6: 1/b + 1/c = 5/6. (b+c)/(bc)=5/6 => 6b+6c=5bc => 5bc-6b-6c=0
    # (5b-6)(5c-6)=36. b>6 so 5b-6>24, 5c-6>24, product > 576 > 36. No.
    # Try a=7: 1/b+1/c=6/7. 7b+7c=6bc => 6bc-7b-7c=0 => (6b-7)(6c-7)=49.
    # b>7 => 6b-7>35, product > 1225 > 49. No.
    # In general for 3 terms with a>=5: 1/b+1/c = 1-1/a = (a-1)/a.
    # (a-1)bc = a(b+c) => (a-1)bc - ab - ac = 0 => multiply by (a-1):
    # (a-1)^2 bc - a(a-1)b - a(a-1)c = 0 => (a(a-1)-a)(a(a-1)-c)... 
    # Standard: (a-1)bc - a(b+c) = 0 => b(a-1)c - ab - ac = 0
    # => b c (a-1) = a(b+c) => 1/b + 1/c = (a-1)/a
    # => c = ab / ((a-1)b - a)
    # For c to be integer and c > b > a >= 5:
    # (a-1)b - a > 0 => b > a/(a-1) which is always true for b >= a+1.
    # c = ab / ((a-1)b - a). Let's check a=5: c = 5b/(4b-5). For b>5:
    # b=6: c=30/19 not int. b=7: 35/23 no. b=8: 40/27 no. b=9: 45/31 no. b=10: 50/35=10/7 no.
    # As b increases, c approaches 5/(4) = 1.25 from above, so c < b for large b, but we need c > b.
    # c > b iff ab > b((a-1)b - a) iff a > (a-1)b - a iff 2a > (a-1)b iff b < 2a/(a-1).
    # For a=5: b < 10/4 = 2.5, but b > a = 5. Contradiction. So no 3-term solution with a>=5.
    # 
    # So minimum is at least 4 terms. Let's check if 4 terms work.
    # Actually, let's just use the greedy algorithm with backtracking to find the minimum count.
    
    def greedy_fraction(remaining, min_denom, max_terms, path):
        """Find representation of 'remaining' as sum of at most max_terms unit fractions
        with denominators >= min_denom, strictly increasing."""
        if max_terms == 0:
            return None if remaining != 0 else list(path)
        if remaining == 0:
            return list(path)
        
        # Lower bound on denominator: 1/d <= remaining => d >= 1/remaining
        # Also d >= min_denom
        # Upper bound: we need remaining - 1/d to be representable with max_terms-1 terms
        # each at least 1/(d+1), so remaining - 1/d <= (max_terms-1)/(d+1)
        # => remaining <= 1/d + (max_terms-1)/(d+1)
        
        # Greedy: try the smallest valid denominator first
        # d must satisfy: d >= min_denom and 1/d <= remaining
        # So d >= ceil(1/remaining) if remaining < 1, else d >= min_denom
        
        if remaining > 1:
            return None  # Can't represent fraction > 1 with unit fractions that sum to it if we're already partial? 
                         # Actually remaining should always be <= 1 in our setup.
        
        if remaining == 0:
            return list(path)
        
        # Minimum denominator
        if remaining < 1:
            min_d = max(min_denom, (1 // remaining.numerator) + (1 if 1 % remaining.numerator else 0))
            # ceil(1/remaining) = ceil(remaining.denominator / remaining.numerator)
            min_d = max(min_denom, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        else:
            min_d = min_denom
        
        # Maximum denominator: 1/d >= remaining/max_terms => d <= max_terms/remaining
        # Also need to be able to form the rest with max_terms-1 terms
        # Simple upper bound: d <= max_terms * remaining.denominator // remaining.numerator
        max_d = max(min_d, max_terms * remaining.denominator // remaining.numerator + 1)
        # Cap to avoid huge searches
        max_d = min(max_d, 10000000)
        
        # Try denominators from min_d to max_d
        # For efficiency, try a limited range first
        for d in range(min_d, min(max_d + 1, min_d + 1000)):
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            if new_remaining == 0:
                if max_terms >= 1:
                    return list(path) + [d]
            if max_terms - 1 >= 1 and new_remaining > 0:
                # Check if new_remaining can be represented with max_terms-1 terms
                # Lower bound check: (max_terms-1) * (1/(d+1)) >= new_remaining? Not exactly.
                # Just recurse
                result = greedy_fraction(new_remaining, d + 1, max_terms - 1, path + [d])
                if result is not None:
                    return result
        
        # If greedy didn't work in the small range, try wider but limited
        for d in range(min_d + 1000, max_d + 1):
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            if new_remaining == 0:
                if max_terms >= 1:
                    return list(path) + [d]
            if max_terms - 1 >= 1 and new_remaining > 0:
                result = greedy_fraction(new_remaining, d + 1, max_terms - 1, path + [d])
                if result is not None:
                    return result
        
        return None
    
    # Try with increasing number of terms starting from minimum possible
    target = Fraction(1, 1)
    
    # Minimum terms: we established at least 4 (actually let's just try from 3 up)
    for num_terms in range(3, 10):
        result = greedy_fraction(target, 5, num_terms, [])
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == target and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: use a known representation
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/140 ... let me just find one computationally
    # Actually, let's use a simpler approach: start with 1/5+1/6+1/7+1/42 = 1
    # 1/5+1/6+1/7 = 41/210, 1 - 41/210 = 169/210. Not a single unit fraction.
    # Let me just return a valid one found by the algorithm or a hardcoded one.
    
    # Hardcoded fallback: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/14 + 1/35 + 1/45 + 1/63 + 1/90 + 1/126 + 1/180 + 1/315 + 1/540 + 1/840 + 1/1260 + 1/1890 + 1/2520 + 1/4200
    # This is getting complicated. Let's just trust the greedy with more terms.
    
    # Try up to 20 terms
    for num_terms in range(3, 21):
        result = greedy_fraction(target, 5, num_terms, [])
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            if s == target and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
