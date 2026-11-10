"""
Optimized Egyptian fraction search for 5/31 minimizing largest denominator.
Uses identity-based decomposition with tight bounds and early termination.
"""
import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_result = None
    
    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)
    
    def time_ok():
        return time.time() < deadline - 3.0
    
    # For a given fraction p/q (in lowest terms), find all unit fraction
    # decompositions with at most 'max_terms' terms, all denominators in (prev, M]
    # Use the identity: p/q = 1/a + (pa-q)/(qa)
    # For the last term, we need the remainder to be a unit fraction.
    
    def search(num, den, prev, max_terms, M, path):
        """Recursively search for decomposition of num/den using at most max_terms
        more unit fractions, all with denominators in (prev, M]."""
        nonlocal best_max, best_result
        
        if max_terms == 0:
            if num == 0:
                mx = max(path) if path else 0
                if mx < best_max:
                    best_max = mx
                    best_result = list(path)
            return
        
        if num <= 0 or den <= 0:
            return
        
        # Reduce fraction
        from math import gcd
        g = gcd(num, den)
        num //= g
        den //= g
        
        if num == 0:
            return
        
        # If max_terms == 1, check if num/den is a unit fraction
        if max_terms == 1:
            if num == 1:
                d = den
                if d > prev and d <= M:
                    path.append(d)
                    mx = max(path)
                    if mx < best_max:
                        best_max = mx
                        best_result = list(path)
                    path.pop()
            return
        
        # For a in range, we need: 1/a <= num/den, so a >= den/num
        # Also a > prev and a <= M
        # And we need the remainder (num*a - den) / (den*a) to be decomposable
        # in at most (max_terms - 1) terms with denominators > a
        
        # Lower bound for a: a > prev, a >= ceil(den/num)
        min_a = max(prev + 1, (den + num - 1) // num)
        # Upper bound: a <= M, and we need at least (max_terms-1) more terms
        # The smallest possible sum of (max_terms-1) terms with denominators > a
        # is 1/(a+1) + 1/(a+2) + ... + 1/(a+(max_terms-1))
        # We need num/den - 1/a >= that minimum
        # Also, 1/a < num/den (strictly, since we need positive remainder)
        # So a > den/num, which is already covered by min_a
        
        # Upper bound from needing positive remainder:
        # num*a - den > 0 => a > den/num (already have)
        # Also need num*a - den < den*a (always true for positive num, den)
        
        # Practical upper bound: a <= M
        max_a = min(M, (den * max_terms) // num)  # rough upper bound
        
        for a in range(min_a, max_a + 1):
            if not time_ok():
                return
            
            rem_num = num * a - den
            rem_den = den * a
            
            if rem_num <= 0:
                continue
            
            # Check if rem can be decomposed in at most (max_terms-1) terms
            # Quick feasibility: rem_num/rem_den must be > 0 and <= sum of largest possible
            # The maximum sum of (max_terms-1) unit fractions with denominators > a
            # is 1/(a+1) + 1/(a+2) + ... (but this is an upper bound on what's possible)
            
            # Reduce rem
            g2 = gcd(rem_num, rem_den)
            rn = rem_num // g2
            rd = rem_den // g2
            
            # For the remainder to be expressible as sum of (max_terms-1) unit fractions
            # with denominators > a, we need rn/rd to be "small enough"
            # Quick check: rn/rd < (max_terms-1) / (a+1) roughly
            if rn * (a + 1) >= rd * (max_terms - 1):
                # Too large, skip (this is a necessary condition)
                # Actually this might be too aggressive, let's be more careful
                pass
            
            path.append(a)
            search(rn, rd, a, max_terms - 1, M, path)
            path.pop()
            
            if best_max <= a:
                # Can't do better if current max is already <= a
                # and we're trying larger a values
                # Actually we can still find solutions with same max but different composition
                # But if best_max <= a, any solution found will have max >= a+1 > best_max
                # Wait, no: the current path has max = max(prev_path_max, a)
                # If best_max <= a, then any completion will have max >= a >= best_max
                # So we can break if we're confident
                # But we might find a solution with the same max... let's not break early
                pass
    
    # Strategy: Try increasing M from small values upward
    # For each M, try to find any valid decomposition
    # Start with M = 7 (minimum possible since 1/7 < 5/31 < 1/6... wait)
    # 5/31 ≈ 0.1613, 1/7 ≈ 0.1429, 1/6 ≈ 0.1667
    # So the largest term must be at least 1/7 (denominator at most 7 for the first term)
    # But we need multiple terms. The smallest denominator in the set must satisfy
    # 1/d <= 5/31, so d >= 31/5 = 6.2, so d >= 7.
    
    # Try M from 7 upward
    for M in range(7, 300):
        if not time_ok():
            break
        
        # Try with up to 10 terms (generous, but the search will prune heavily)
        # Start with fewer terms for speed, then increase
        for max_terms in range(3, 11):
            if not time_ok():
                break
            
            search(5, 31, 0, max_terms, M, [])
            
            if best_max <= M:
                # Found a solution with max <= M, can stop increasing M
                break
        
        if best_max <= M:
            # Found a solution, continue searching for better (smaller M)
            # Actually we're going upward, so if we found one at M, 
            # we should continue to see if there's a smaller one... 
            # But we're going from small M upward, so the first solution found
            # at the smallest M is optimal!
            # Wait, no: we try M=7 first, then M=8, etc.
            # The first M where we find a solution gives the minimum largest denominator.
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: use a known valid decomposition
    # 5/31 = 1/7 + 1/21 + 1/147 + ... let me compute
    # Actually, let's use the greedy algorithm as fallback
    remaining = Fraction(5, 31)
    denoms = []
    while remaining > 0 and len(denoms) < 256:
        d = (31 * remaining.denominator) // (5 * remaining.numerator)
        if d < 1:
            d = 1
        # Greedy: d = ceil(den/num)
        d = -(-remaining.denominator // remaining.numerator)
        if d in denoms:
            d += 1
        denoms.append(d)
        remaining -= Fraction(1, d)
        if remaining < 0:
            break
        if d > 10000000:
            break
    
    if remaining == 0 and all(d <= 10000000 for d in denoms) and len(set(denoms)) == len(denoms):
        return denoms
    
    # Last resort: hard-coded valid solution
    # Let me verify: 1/7 + 1/21 + 1/147 + 1/1029 + ...
    # 5/31 - 1/7 = (35-31)/217 = 4/217
    # 4/217 - 1/21 = (84-217)/4557... no
    # Let me just use a simple valid one
    # 5/31 = 1/7 + 4/217
    # 4/217: try 1/55: 4/217 - 1/55 = (220-217)/11935 = 3/11935
    # 3/11935: try 1/3979: 3/11935 - 1/3979 = (11937-11935)/47489365 = 2/47489365
    # This is getting large. Let me use a different approach.
    
    # Known: 5/31 can be written with small denominators
    # Let me try: 1/7 + 1/21 + 1/147 + 1/1029 + 1/7203 + ...
    # 5/31 = 0.16129032...
    # 1/7 = 0.14285714...
    # diff = 0.01843318... = 4/217
    # 1/21 = 0.04761905... too big
    # 1/55 = 0.01818182...
    # 4/217 - 1/55 = (4*55 - 217)/(217*55) = (220-217)/11935 = 3/11935
    # 1/3979 = 0.00025132...
    # 3/11935 ≈ 0.00025136...
    # 3/11935 - 1/3979 = (3*3979 - 11935)/(11935*3979) = (11937-11935)/47489365 = 2/47489365
    # 1/23744683 = ...
    # 2/47489365 - 1/23744683 = (2*23744683 - 47489365)/(47489365*23744683) = (47489366-47489365)/... = 1/...
    # So: 5/31 = 1/7 + 1/55 + 1/3979 + 1/23744683 + 1/(47489365*23744683)
    # That last one is way too big.
    
    # Let me try a different decomposition
    # 5/31 = 1/8 + 1/10 + 1/... 
    # 1/8 + 1/10 = 9/40 = 0.225 > 0.1613, too big
    # 1/9 + 1/11 + 1/13 = (143+117+99)/1287 = 359/1287 ≈ 0.279, too big
    # 1/10 + 1/12 + 1/14 = (42+35+30)/420 = 107/420 ≈ 0.2548, too big
    # 1/11 + 1/13 + 1/15 = (195+165+143)/2145 = 503/2145 ≈ 0.2345, too big
    # 1/12 + 1/14 + 1/16 = (56+48+42)/336 = 146/336 ≈ 0.4345... wait that's wrong
    # 1/12 + 1/14 + 1/16 = (56+48+42)/336 = 146/336 = 73/168 ≈ 0.4345, way too big
    
    # Let me be more careful. 5/31 ≈ 0.16129
    # 1/7 ≈ 0.14286, remainder ≈ 0.01843
    # 1/55 ≈ 0.01818, remainder ≈ 0.00025
    # 1/3979 ≈ 0.0002513, very close
    
    # Alternative: 1/7 + 1/54 + ...
    # 4/217 - 1/54 = (216-217)/(217*54) = -1/11718, negative!
    
    # 1/7 + 1/56 + ...
    # 4/217 - 1/56 = (224-217)/(217*56) = 7/12152 = 1/1736
    # So 5/31 = 1/7 + 1/56 + 1/1736
    # Check: 1/7 + 1/56 + 1/1736 = (248 + 31 + 1)/1736 = 280/1736 = 35/217 = 5/31 ✓
    # Max denominator = 1736
    
    # Can we do better? Let's try 1/7 + 1/55 + 1/3979 + 1/23744683 (too big)
    # 1/7 + 1/57 + ...
    # 4/217 - 1/57 = (228-217)/(217*57) = 11/12369
    # 11/12369: 1/1124 = 0.000890, 11/12369 ≈ 0.000889
    # 11/12369 - 1/1124 = (12369*... let me compute: 11*1124 = 12364, 12369-12364=5
    # So 11/12369 - 1/1124 = 5/(12369*1124) = 5/13899756
    # 1/2779951 = ...
    # 5/13899756 - 1/2779951 = (5*2779951 - 13899756)/(13899756*2779951) = (13899755-13899756)/... = -1/...
    # Negative! So 1/2779951 is slightly too big.
    
    # Let me try: 5/13899756, need 1/d where d ≈ 13899756/5 = 2779951.2
    # d = 2779952: 5/13899756 - 1/2779952 = (5*2779952 - 13899756)/(13899756*2779952)
    # = (13899760 - 13899756)/(...) = 4/(13899756*2779952)
    # = 1/(3474939*2779952) which is huge.
    
    # So 1/7 + 1/56 + 1/1736 gives max 1736.
    # Let me try to find something with max < 1736.
    
    # The search above should find it if it exists. If not, return the known solution.
    return [7, 56, 1736]
