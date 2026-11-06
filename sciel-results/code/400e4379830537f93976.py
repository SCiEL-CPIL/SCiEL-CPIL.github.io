"""Program that maximizes reciprocal sum of 3-AP-free subset of [1,256]."""
import sys
from fractions import Fraction

def construct():
    n = 256
    
    # Precompute: for each pair (a, c) with a < c and a+c even,
    # the midpoint b = (a+c)/2 is forbidden if both a and c are selected.
    # We'll build a conflict structure: for each x, which other elements
    # conflict with x (i.e., which pairs with x form a 3-AP).
    
    # For efficiency, precompute for each x the list of (a, c) pairs where
    # x is the midpoint: a + c = 2x, a < x < c.
    # When we select x, we cannot have both a and c selected for any such pair.
    # Equivalently, for each selected x, we add constraints that for each (a,c)
    # with a+c=2x, at most one of a,c can be selected.
    
    # Actually, a cleaner formulation: we're looking for a maximum weight
    # independent set where two elements conflict if they "block" each other.
    # But the conflict is not pairwise - it's a 3-way constraint.
    
    # Let me think differently. A set S is 3-AP-free if for no three distinct
    # elements a, b, c in S do we have a + c = 2b.
    
    # This is equivalent to: for every b in S, and every pair (a, c) with a+c=2b,
    # at most one of a, c is in S.
    
    # For a branch-and-bound approach, we can use the following:
    # Process numbers from 1 to n. At each step, decide whether to include or exclude x.
    # The constraint is: if x is included, then for each pair (a, c) with a+c=2x,
    # we cannot have both a and c in S.
    
    # Upper bound for pruning: sum of 1/i for all i from current position to n
    # that are not yet ruled out.
    
    # Let's use a more direct approach: greedy with local search, but with
    # a better greedy strategy.
    
    # Actually, let me try a different approach: since the reciprocal weights
    # decay as 1/x, the optimal solution will include many small numbers.
    # Let's use a greedy that goes from small to large, but with backtracking
    # for the first few elements where the decisions matter most.
    
    # Better idea: use the structure of 3-AP-free sets. 
    # A key observation: if we look at numbers in binary, 3-AP-free sets are
    # related to sums of distinct powers of 3 (Cantor set analogy), but that's
    # for density, not for reciprocal sum optimization.
    
    # Let me try a practical approach:
    # 1. Start with greedy (ascending order)
    # 2. Try local improvements: for each element, try swapping it out for a
    #    larger element that was previously skipped
    # 3. Iterate until no improvement
    
    def greedy_asc():
        selected = []
        for x in range(1, n + 1):
            conflict = False
            for a in selected:
                for b in selected:
                    if a == b:
                        continue
                    # Check if {a, b, x} contains a 3-AP
                    # Three numbers form a 3-AP iff one is the average of the other two
                    if 2 * a == b + x or 2 * b == a + x or 2 * x == a + b:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                selected.append(x)
        return selected
    
    def is_3ap_free(S):
        Sset = set(S)
        for a in S:
            for c in S:
                if a == c:
                    continue
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if a < b < c and b in Sset:
                        return False
        return True
    
    def reciprocal_sum(S):
        return sum(Fraction(1, x) for x in S)
    
    def reciprocal_sum_float(S):
        return sum(1.0 / x for x in S)
    
    # Start with greedy
    best = greedy_asc()
    best_val = reciprocal_sum_float(best)
    
    # Local search: try to improve by swapping elements
    # For each element in best, try removing it and adding a skipped element
    for iteration in range(50):  # limited iterations for time
        improved = False
        current = list(best)
        current_set = set(current)
        skipped = [x for x in range(1, n + 1) if x not in current_set]
        
        for i, x in enumerate(current):
            # Try removing x
            test_set = current_set - {x}
            # Try adding each skipped element
            for y in skipped:
                if y in test_set:
                    continue
                candidate = test_set | {y}
                if is_3ap_free(list(candidate)):
                    new_val = reciprocal_sum_float(list(candidate))
                    if new_val > best_val + 1e-12:
                        best = list(candidate)
                        best_val = new_val
                        current = list(best)
                        current_set = set(best)
                        skipped = [x for x in range(1, n + 1) if x not in current_set]
                        improved = True
                        break
        if not improved:
            break
    
    # Also try: for each pair of elements in best, try swapping one for a skipped one
    # This is more expensive, so limit it
    if iteration < 50:  # only if we haven't used all iterations
        for iteration2 in range(20):
            improved = False
            current = list(best)
            current_set = set(current)
            skipped = [x for x in range(1, n + 1) if x not in current_set]
            
            for i, x in enumerate(current):
                test_set = current_set - {x}
                for y in skipped:
                    if y in test_set:
                        continue
                    candidate = test_set | {y}
                    if is_3ap_free(list(candidate)):
                        new_val = reciprocal_sum_float(list(candidate))
                        if new_val > best_val + 1e-12:
                            best = list(candidate)
                            best_val = new_val
                            current = list(best)
                            current_set = set(best)
                            skipped = [x for x in range(1, n + 1) if x not in current_set]
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    return best
