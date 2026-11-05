"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    from itertools import combinations

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add(x, S):
        """Check if x can be added to sum-free set S."""
        # 2x must not be in S
        if 2 * x in S:
            return False
        # For all a in S: x + a must not be in S, and a - x must not be in S
        # (a - x in S means a = x + (a-x), so x + (a-x) = a, violation)
        # Also: no a, b in S with a + b = x
        for a in S:
            if x + a in S:
                return False
            if a - x in S:
                return False
        # Check no two elements in S sum to x
        # This is equivalent to: for no a in S is (x - a) in S with a <= x-a to avoid double counting
        # But simpler: just check all pairs... but that's O(n^2). 
        # Actually, the condition "a + b = x for a,b in S" is already covered?
        # No! The above checks x+a in S and a-x in S, but not a+b=x for a!=b both in S.
        # Let me re-examine.
        
        # Violations when adding x to S:
        # 1. x + x = 2x: need 2x not in S ✓ (checked above)
        # 2. x + a = b for a,b in S: need b - a != x for all a,b in S
        #    This is: for all a in S, a + x not in S ✓ (checked: x+a in S)
        # 3. a + b = x for a,b in S: need no a,b in S with a+b=x
        #    This is NOT checked above! I need to check this.
        
        # For condition 3: check if any a in S has (x-a) also in S
        # To avoid double counting and self-pairs, just check all a
        for a in S:
            if a != x - a and (x - a) in S:
                return False
            # If a == x-a, i.e., 2a=x, then a+a=x, which is a violation if a in S
            # But we already checked 2x not in S... wait, that's different.
            # If 2a = x, then a + a = x, and a in S, so this is a violation.
            # This is covered by the loop: if a == x-a, we skip the "a != x-a" check,
            # but we should still flag it. Let me fix.
        # Let me redo: for any a in S, if (x-a) in S, then a + (x-a) = x, violation.
        # This includes the case a = x-a (i.e., 2a=x).
        for a in S:
            if (x - a) in S:
                return False
        
        return True

    def greedy_sum_free(elements, order):
        """Greedy: process elements in given order, add if possible."""
        S = set()
        S_list = []
        for x in order:
            if can_add(x, S):
                S.add(x)
                S_list.append(x)
        return S_list

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elems = instance(size, seed)
        
        # Try multiple orderings and pick the best
        best = []
        
        # Ordering 1: odds first (by abs value descending), then evens (by abs value descending)
        odds = [x for x in elems if x % 2 != 0]
        evens = [x for x in elems if x % 2 == 0]
        odds.sort(key=abs, reverse=True)
        evens.sort(key=abs, reverse=True)
        order1 = odds + evens
        best = greedy_sum_free(elems, order1)
        
        # Ordering 2: all by abs value descending
        order2 = sorted(elems, key=abs, reverse=True)
        res2 = greedy_sum_free(elems, order2)
        if len(res2) > len(best):
            best = res2
        
        # Ordering 3: odds by abs value ascending, then evens by abs value ascending
        odds_asc = sorted(odds, key=abs)
        evens_asc = sorted(evens, key=abs)
        order3 = odds_asc + evens_asc
        res3 = greedy_sum_free(elems, order3)
        if len(res3) > len(best):
            best = res3
        
        # Ordering 4: evens first (by abs value descending), then odds
        order4 = evens + odds
        res4 = greedy_sum_free(elems, order4)
        if len(res4) > len(best):
            best = res4
        
        # Ordering 5: random-ish deterministic shuffle based on value
        order5 = sorted(elems, key=lambda x: (abs(x) % 7, abs(x), x))
        res5 = greedy_sum_free(elems, order5)
        if len(res5) > len(best):
            best = res5
        
        results.append(sorted(best))
    
    return results
# EVOLVE-BLOCK-END
