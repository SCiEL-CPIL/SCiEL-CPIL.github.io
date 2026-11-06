"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # We want to find a subset S of {1,...,24} such that for every x in S,
    # 1/x is NOT equal to the sum of reciprocals of any non-empty subset of S \ {x}.
    # Equivalently: for every x in S, 1/x is not in the set of all subset sums of {1/y : y in S, y != x}.
    
    # Strategy:
    # 1. Start with a greedy construction trying to add elements in some order.
    # 2. Use local search (add/remove/swap) to maximize cardinality.
    # 3. Use exact arithmetic with Fractions to avoid floating point errors.
    
    def is_valid(S):
        """Check if S is a valid reciprocal-subsum-free set."""
        if not S:
            return False
        S_set = set(S)
        for x in S_set:
            # Check if 1/x can be formed as sum of reciprocals of a subset of S_set - {x}
            others = [y for y in S_set if y != x]
            if not others:
                continue
            target = Fraction(1, x)
            # Check all non-empty subsets of others
            # For small sets this is fine, for larger we need pruning
            # Use DFS with pruning
            others_sorted = sorted(others, key=lambda v: Fraction(1, v), reverse=True)
            # Precompute reciprocal values
            reciprocals = [Fraction(1, v) for v in others_sorted]
            
            def dfs(idx, current_sum):
                if current_sum == target:
                    return True
                if idx >= len(reciprocals):
                    return False
                if current_sum > target:
                    return False
                # Pruning: even if we add all remaining, can we reach target?
                # This is tricky with fractions, skip for now
                # Option 1: skip
                if dfs(idx + 1, current_sum):
                    return True
                # Option 2: take
                new_sum = current_sum + reciprocals[idx]
                if new_sum <= target:
                    if dfs(idx + 1, new_sum):
                        return True
                return False
            
            if dfs(0, Fraction(0)):
                return False
        return True
    
    def can_add(S, x):
        """Check if x can be added to S while maintaining validity."""
        if x in S:
            return False
        S_new = S | {x}
        return is_valid(S_new)
    
    # Greedy construction: try to build a large set
    # Try different orderings
    best_set = set()
    
    # First, try the simple approach: start with larger numbers (smaller reciprocals)
    # and greedily add
    for start_order in [
        range(24, 0, -1),  # 24, 23, ..., 1
        range(1, 25),      # 1, 2, ..., 24
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        # Interleaved: large evens, then large odds, etc.
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    ]:
        if time.time() > deadline:
            break
        S = set()
        for x in start_order:
            if can_add(S, x):
                S.add(x)
        if len(S) > len(best_set):
            best_set = S.copy()
    
    # Local search: try to improve by adding, removing, or swapping elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding elements not in best_set
        for x in range(1, 25):
            if x not in best_set:
                if can_add(best_set, x):
                    best_set.add(x)
                    improved = True
                    break  # Restart the loop after a change
        
        if time.time() > deadline:
            break
        
        # Try removing one element and adding two others
        if not improved:
            for x in list(best_set):
                if time.time() > deadline:
                    break
                S_minus = best_set - {x}
                # Try to add any element not in best_set
                for y in range(1, 25):
                    if y not in best_set and can_add(S_minus, y):
                        S_plus_y = S_minus | {y}
                        # Try to add another element
                        for z in range(1, 25):
                            if z not in S_plus_y and can_add(S_plus_y, z):
                                S_new = S_plus_y | {z}
                                if len(S_new) > len(best_set):
                                    best_set = S_new
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    break
    
    # Final check
    if not best_set:
        # Fallback: return a known valid small set
        best_set = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
