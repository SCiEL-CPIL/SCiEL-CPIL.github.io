"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    def check_feasible(selected_set, candidate):
        """Check if adding candidate to selected_set maintains the property."""
        # selected_set is a set for O(1) lookup
        # For every a < b <= c in (selected_set ∪ {candidate}), a must not divide b+c
        # We need to check all triples involving the new candidate
        # Cases: candidate is a, candidate is b, candidate is c
        # Also need to verify existing triples are still fine (they were before)
        
        # Since we're adding one element, we only need to check triples that include it
        # Let x = candidate
        # Case 1: x is the smallest (a = x), so x < b <= c, both in selected_set
        #   Need: for all b,c in selected_set with x < b <= c, x does not divide b+c
        # Case 2: x is the middle (b = x), so a < x <= c, a in selected_set, c in selected_set or c=x
        #   Sub-case 2a: a < x <= x (c=x): need a does not divide x+x=2x for all a in selected_set with a<x
        #   Sub-case 2b: a < x <= c, c in selected_set, c >= x: need a does not divide x+c
        # Case 3: x is the largest (c = x), so a < b <= x, a,b in selected_set, b <= x
        #   Need: for all a,b in selected_set with a < b <= x, a does not divide b+x
        
        # More systematically: for all triples (a,b,c) with a<b<=c from S∪{x}, 
        # where at least one is x, check a does not divide b+c.
        
        sel_list = sorted(selected_set)
        
        # Build list with candidate
        # Check all triples involving x
        # For efficiency, iterate over pairs and check
        
        # Case: x is a (smallest)
        for i, b in enumerate(sel_list):
            if b <= x:
                continue
            for c in sel_list[i:]:  # c >= b, and c > x since b > x
                if c <= x:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case: x is b (middle), a < x <= c
        for a in sel_list:
            if a >= x:
                break
            # c can be x or any element in sel_list with c >= x
            # c = x: check a does not divide 2x
            if (2 * x) % a == 0:
                return False
            # c in sel_list with c >= x
            for c in sel_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case: x is c (largest), a < b <= x
        for i, a in enumerate(sel_list):
            if a >= x:
                break
            for j in range(i+1, len(sel_list)):
                b = sel_list[j]
                if b > x:
                    break
                # a < b <= x, c = x
                if (b + x) % a == 0:
                    return False
        
        return True
    
    # Start with the known good upper interval [171, 256]
    base = set(range(171, n + 1))
    
    # Try to add elements from 1 to 170 in increasing order
    result_set = set(base)
    
    for candidate in range(1, 171):
        if time.time() > deadline:
            break
        if check_feasible(result_set, candidate):
            result_set.add(candidate)
    
    # Try to add elements from 1 to 170 in decreasing order as well (different order might help)
    # Actually, let's try a second pass with different ordering
    # But first, let's also try starting from scratch with greedy
    
    # Alternative: try greedy from 1 upward
    greedy_set = set()
    for candidate in range(1, n + 1):
        if time.time() > deadline:
            break
        if check_feasible(greedy_set, candidate):
            greedy_set.add(candidate)
    
    # Also try greedy from n downward
    greedy_desc = set()
    for candidate in range(n, 0, -1):
        if time.time() > deadline:
            break
        if check_feasible(greedy_desc, candidate):
            greedy_desc.add(candidate)
    
    # Pick the best
    candidates = [result_set, greedy_set, greedy_desc]
    best = max(candidates, key=len)
    
    # Try to improve best by attempting to add any remaining elements
    for candidate in range(1, n + 1):
        if time.time() > deadline:
            break
        if candidate not in best:
            if check_feasible(best, candidate):
                best.add(candidate)
    
    return list(best)
# EVOLVE-BLOCK-END
