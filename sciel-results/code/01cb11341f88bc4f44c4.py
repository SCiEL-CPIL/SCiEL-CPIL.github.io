"""Rewritten program: search for optimal finite sets maximizing the sumset/difference-set exponent lower bound."""
import itertools
import math

# EVOLVE-BLOCK-START
def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize the score."""
    # The score is 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    # We want to maximize |U-U|/|U+U| relative to the spread of U.
    
    # Strategy: Use a set based on a well-known construction from additive combinatorics.
    # The set {0, 1, 3, 7, 15, ..., 2^k - 1} has good properties.
    # More generally, we can search computationally for small optimal sets.
    
    # Precomputed optimal small sets found by exhaustive search:
    # For n=1: {0} -> score = 1 (degenerate)
    # For n=2: {0, a} -> |U-U|=3, |U+U|=3, ratio=1, score=1
    # For n=3: best is {0, 1, 3} or {0, 1, 2}? 
    #   {0,1,3}: U+U={0,1,2,3,4,6} (6), U-U={-3,-2,-1,0,1,2,3} (7), ratio=7/6
    #   {0,1,2}: U+U={0,1,2,3,4} (5), U-U={-2,-1,0,1,2} (5), ratio=1
    #   So {0,1,3} is better.
    # For n=4: try {0,1,3,7}
    #   U+U: 0,1,2,3,4,6,7,8,10,14 -> let me compute properly
    #   Sums: 0+0=0, 0+1=1, 0+3=3, 0+7=7, 1+1=2, 1+3=4, 1+7=8, 3+3=6, 3+7=10, 7+7=14
    #   U+U = {0,1,2,3,4,6,7,8,10,14}, |U+U|=10
    #   U-U: all differences. Max diff=7, min=-7. 
    #   Differences: 0, ±1, ±3, ±7, ±2, ±4, ±6, ±8, ±10, ±14
    #   Let me list: from 0: 0,1,3,7; from 1: -1,0,2,6; from 3: -3,-2,0,4; from 7: -7,-6,-4,0
    #   All unique diffs: -7,-6,-4,-3,-2,-1,0,1,2,3,4,6,7 -> that's 13 elements
    #   Wait, I'm missing some. Let me be more careful.
    #   Pairs (a,b) with a,b in {0,1,3,7}:
    #   0-0=0, 0-1=-1, 0-3=-3, 0-7=-7
    #   1-0=1, 1-1=0, 1-3=-2, 1-7=-6
    #   3-0=3, 3-1=2, 3-3=0, 3-7=-4
    #   7-0=7, 7-1=6, 7-3=4, 7-7=0
    #   Unique: -7,-6,-4,-3,-2,-1,0,1,2,3,4,6,7 -> 13 elements
    #   |U-U|=13, |U+U|=10, ratio=13/10=1.3
    #   log(1.3)/log(15) = 0.262/2.708 ≈ 0.097, score ≈ 1.097
    #
    # Let me try to find better sets computationally for small n.
    
    def score_of(U):
        U = sorted(set(U))
        if not U or U[0] != 0:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        ratio = len(diffs) / len(sums)
        if ratio <= 0:
            return 0
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 0
        return 1 + math.log(ratio) / denom
    
    # Search for good sets up to a certain size
    best_set = [0, 1, 3]
    best_score = score_of(best_set)
    
    # Try sets of form {0, a1, a2, ..., ak} with small values
    # Use a greedy/local search approach
    
    def local_search(start_set, max_val=30, iterations=2000):
        current = list(start_set)
        current_score = score_of(current)
        for _ in range(iterations):
            improved = False
            # Try adding a new element
            for x in range(1, max_val + 1):
                if x not in current:
                    new_set = current + [x]
                    s = score_of(new_set)
                    if s > current_score + 1e-12:
                        current = new_set
                        current_score = s
                        improved = True
                        break
            if improved:
                continue
            # Try removing an element (keep 0)
            for i in range(1, len(current)):
                new_set = current[:i] + current[i+1:]
                if len(new_set) < 2:
                    continue
                s = score_of(new_set)
                if s > current_score + 1e-12:
                    current = new_set
                    current_score = s
                    improved = True
                    break
            if improved:
                continue
            # Try replacing an element
            for i in range(1, len(current)):
                old = current[i]
                for x in range(1, max_val + 1):
                    if x == old or x in current:
                        continue
                    new_set = current[:i] + [x] + current[i+1:]
                    s = score_of(new_set)
                    if s > current_score + 1e-12:
                        current = new_set
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return current, current_score
    
    # Run local search from several starting points
    starts = [
        [0, 1, 3],
        [0, 1, 2, 5],
        [0, 1, 4, 6],
        [0, 2, 3, 7],
        [0, 1, 3, 4],
        [0, 1, 2, 3, 7],
        [0, 1, 3, 7, 12],
    ]
    
    for start in starts:
        result_set, result_score = local_search(start, max_val=40, iterations=1000)
        if result_score > best_score:
            best_score = result_score
            best_set = result_set
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
