"""Program: large 3-sum-free-pair-sum set in 1..128 via bit-parallel greedy independent set."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

N = 128

def construct():
    """
    Find a large subset S of {1..128} such that there do NOT exist three distinct
    a,b,c in S with a+b, a+c, b+c all in S.
    
    Approach:
    1. Precompute all forbidden triples (a,b,c) with a<b<c where all pair sums are in [1,128].
    2. Use randomized greedy with multiple strategies to find large valid sets.
    3. Use a conflict graph approach: when adding vertex x, check against all existing vertices.
    """
    
    # Precompute forbidden triples: (a,b,c) with a<b<c and a+b<=128, a+c<=128, b+c<=128
    # The triple is "dangerous" if all three pair sums could be in S
    # We'll use this to quickly check conflicts
    
    # Strategy: Greedy with random ordering, multiple restarts
    # Key insight: Large numbers are safer because their pair sums are more likely to exceed 128
    
    best = None
    best_len = 0
    
    rng = random.Random(42)
    
    # Precompute: for each pair (a,b), what is a+b?
    # And for checking if adding x creates a conflict with existing S:
    # For each pair (a,b) in S with a<b, check if (x+a in S) and (x+b in S) and (a+b in S or a+b==x)
    
    def can_add_fast(x, S_set):
        """Check if x can be added to S_set without creating a forbidden triple."""
        # For each pair (a,b) in S_set with a<b:
        #   If x+a in S_set and x+b in S_set and (a+b in S_set or a+b == x):
        #     Then {x,a,b} is a forbidden triple
        # Optimization: only need to check pairs where x+a<=128 and x+b<=128
        # Also, a+b must be <=128 or a+b==x
        
        # Collect all a in S_set such that x+a <= 128 and x+a in S_set
        # Then for each pair (a,b) from this filtered set, check if a+b in S_set or a+b==x
        
        # More efficient: iterate over pairs directly but with early termination
        elems = sorted(S_set)
        n = len(elems)
        
        for i in range(n):
            a = elems[i]
            xa = x + a
            if xa > 128:
                break  # Since elems is sorted, all subsequent a will also give xa>128
            if xa not in S_set:
                continue
            for j in range(i+1, n):
                b = elems[j]
                xb = x + b
                if xb > 128:
                    break
                if xb not in S_set:
                    continue
                ab = a + b
                if ab == x or ab in S_set:
                    return False
        return True
    
    def greedy_trial(ordering_type="random"):
        """Run one greedy trial with given ordering strategy."""
        if ordering_type == "random":
            perm = list(range(1, 129))
            rng.shuffle(perm)
        elif ordering_type == "descending":
            perm = list(range(128, 0, -1))
        elif ordering_type == "ascending":
            perm = list(range(1, 129))
        elif ordering_type == "large_first_random":
            # Start with large numbers, then randomize the rest
            large = list(range(128, 64, -1))
            small = list(range(1, 65))
            rng.shuffle(small)
            perm = large + small
        else:
            perm = list(range(1, 129))
            rng.shuffle(perm)
        
        S_set = set()
        for x in perm:
            if can_add_fast(x, S_set):
                S_set.add(x)
        
        return S_set
    
    # Try multiple strategies and restarts
    strategies = ["random", "descending", "large_first_random"]
    
    for strat in strategies:
        for trial in range(500):
            S = greedy_trial(strat)
            if len(S) > best_len:
                best_len = len(S)
                best = S
            
            if best_len >= 95:
                break
        if best_len >= 95:
            break
    
    # Final validation
    def is_valid(S):
        S_set = set(S)
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True
    
    if best is None or not is_valid(best):
        # Fallback: top half is always valid
        best = set(range(65, 129))
    
    return list(best)
# EVOLVE-BLOCK-END
