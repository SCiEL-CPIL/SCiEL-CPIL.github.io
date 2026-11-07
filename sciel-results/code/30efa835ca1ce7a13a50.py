import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 128
    
    # Precompute all forbidden triples (a, b, c) with a < b < c
    # where a+b, a+c, b+c are all <= 128 (so they could be in the set)
    # The constraint: if a,b,c are in S AND a+b, a+c, b+c are in S, that's forbidden.
    # But we want to avoid having all six in S.
    # Actually the constraint is: no three distinct a,b,c in S such that a+b, a+c, b+c are ALL in S.
    
    # For a given set S, a triple (a,b,c) is "active" if a,b,c in S and a+b,a+c,b+c in S.
    # We need to avoid any active triple.
    
    # Strategy: precompute all potential triples (a,b,c) with a<b<c and a+b<=128, a+c<=128, b+c<=128
    # Then for each such triple, if all of {a,b,c,a+b,a+c,b+c} are in S, it's a violation.
    
    # Hitting set approach: for each forbidden configuration, we need to remove at least one of the 6 elements.
    # But that's complex. Let's think differently.
    
    # Actually, let's use a different structural approach:
    # The constraint only triggers when we have 6 specific values all present.
    # Key observation: if we ensure that for any triple (a,b,c), at least one of {a,b,c,a+b,a+c,b+c} is missing, we're safe.
    
    # Let's precompute all "critical sets" - minimal sets of values that would cause a violation if all present.
    # A critical set is {a,b,c,a+b,a+c,b+c} for distinct a<b<c with all sums <= 128.
    
    # But there are many such sets. Let's count them.
    # a < b < c, a+b <= 128, a+c <= 128, b+c <= 128
    # Since b < c, b+c < 2c, so c < 64. And a+b < 2b, so b < 64. And a < b < 64.
    
    # Let's precompute all such critical 6-sets (or smaller if some values coincide, but a,b,c distinct and sums > max(a,b,c) so all 6 are distinct)
    
    critical_sets = []
    for a in range(1, 129):
        for b in range(a+1, 129):
            if a + b > 128:
                break
            for c in range(b+1, 129):
                if a + c > 128 or b + c > 128:
                    break
                # All six values: a, b, c, a+b, a+c, b+c
                # They are all distinct since a<b<c and sums > c
                s = frozenset([a, b, c, a+b, a+c, b+c])
                critical_sets.append(s)
    
    # Now we want to find a large subset S of {1..128} such that no critical_set is a subset of S.
    # This is a maximum independent set in a hypergraph, or equivalently, a hitting set complement.
    
    # Greedy approach: start with all 128 elements, then remove elements to hit all critical sets.
    # Then try to add back.
    
    # Better: use a greedy algorithm that builds up the set, checking against critical sets.
    # But that's what the current program does essentially.
    
    # Let's try a different structural approach: use the fact that odd numbers work perfectly.
    # Start with all odds (64 elements), then try to add some evens.
    
    # When we add an even number e, it can only form a violation with two odds o1, o2 if:
    # {e, o1, o2, e+o1, e+o2, o1+o2} are all in S.
    # But o1+o2 is even, and e+o1, e+o2 are odd. So we need e+o1, e+o2, o1+o2 all in S.
    # Since we start with all odds, o1+o2 (even) is NOT in S initially. So adding evens one by one,
    # we need to check if the new even e combined with existing odds creates a violation.
    
    # Let's build from odds and try to add evens.
    
    S = set(range(1, 129, 2))  # All odds, size 64
    
    def is_valid_set(S):
        """Check if S is valid using precomputed critical sets."""
        S_frozen = frozenset(S)
        for cs in critical_sets:
            if cs.issubset(S_frozen):
                return False
        return True
    
    # Verify odds are valid
    # For any critical set, it contains a,b,c (all distinct) and a+b,a+c,b+c.
    # If a,b,c are all odd, then a+b, a+c, b+c are all even.
    # So the critical set contains 3 odds and 3 evens.
    # If S is all odds, it contains the 3 odds but not the 3 evens, so no critical set is fully contained.
    # Good.
    
    # Now try to add evens
    evens = list(range(2, 129, 2))
    random.seed(42)
    
    # Try adding each even in various orders
    best_S = set(S)
    best_size = len(S)
    
    # Greedy: try to add evens in ascending order
    for e in evens:
        if time.time() > deadline - 5:
            break
        new_S = S | {e}
        # Check if valid: only need to check critical sets that contain e
        # A critical set contains e if e is one of {a,b,c,a+b,a+c,b+c}
        # For efficiency, just check all critical sets that contain e
        valid = True
        for cs in critical_sets:
            if e in cs and cs.issubset(frozenset(new_S)):
                valid = False
                break
        if valid:
            S = new_S
    
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    # Now do local search: try swaps
    def try_improve(S, deadline):
        """Try to improve S by adding elements or doing swaps."""
        S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any element not in S
            for x in range(1, 129):
                if x in S:
                    continue
                new_S = S | {x}
                # Check validity
                valid = True
                for cs in critical_sets:
                    if x in cs and cs.issubset(frozenset(new_S)):
                        valid = False
                        break
                if valid:
                    S = new_S
                    improved = True
                    break
            
            if not improved:
                # Try 1-for-1 swaps
                S_list = list(S)
                not_in_S = [x for x in range(1, 129) if x not in S]
                random.shuffle(S_list)
                random.shuffle(not_in_S)
                
                for x in not_in_S:
                    if time.time() > deadline:
                        break
                    for y in S_list:
                        new_S = (S - {y}) | {x}
                        if len(new_S) != len(S):
                            continue
                        # Check validity
                        valid = True
                        for cs in critical_sets:
                            if (x in cs or y in cs) and cs.issubset(frozenset(new_S)):
                                valid = False
                                break
                        if valid:
                            S = new_S
                            improved = True
                            break
                    if improved:
                        break
        
        return S
    
    # Run local search from the greedy result
    S_improved = try_improve(S, deadline - 5)
    if len(S_improved) > best_size:
        best_size = len(S_improved)
        best_S = S_improved
    
    # Also try starting from different initial sets
    # Start with high numbers
    S2 = set(range(65, 129))
    for e in range(1, 65):
        if time.time() > deadline - 5:
            break
        new_S = S2 | {e}
        valid = True
        for cs in critical_sets:
            if e in cs and cs.issubset(frozenset(new_S)):
                valid = False
                break
        if valid:
            S2 = new_S
    
    S2_improved = try_improve(S2, deadline - 5)
    if len(S2_improved) > best_size:
        best_size = len(S2_improved)
        best_S = S2_improved
    
    # Final verification
    if not is_valid_set(best_S):
        # Fallback to odds
        best_S = set(range(1, 129, 2))
    
    result = [int(x) for x in best_S]
    return result
