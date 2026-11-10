# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

def construct():
    n = 192
    max_k = 3
    # We want to select a subset S of {1,...,n} such that for every a in S,
    # there do not exist distinct b,c,d in S\{a} with a | (b+c+d).
    # Maximize |S|.
    
    # Precompute for each triple (i,j,k) with i<j<k, the sum s=i+j+k.
    # For each possible divisor a (1..n), we can mark which triples have sums divisible by a.
    # But more efficiently: for each a, we want to know if there exists a triple of other selected elements whose sum is divisible by a.
    
    # Let's think differently. Use a backtracking search with pruning.
    # Order candidates from largest to smallest (heuristic: larger numbers are less likely to divide sums).
    # For each candidate a, we need to check: given current selection S, does adding a violate any constraint?
    #   - For a itself: are there 3 distinct elements in S whose sum is divisible by a?
    #   - For each existing element b in S: does adding a create a new triple involving a and two other elements in S\{b} such that b divides that triple's sum?
    #     Wait, the constraint for b is: no 3 distinct OTHER selected integers sum to a multiple of b.
    #     When we add a, the new triples that include a are: (a, x, y) where x,y in S, x!=y, and x,y != b.
    #     So for each b in S, we need to check if there exist x,y in S\{b,a} such that b | (a + x + y).
    #     Also, triples not involving a were already checked when they were formed.
    
    # This incremental check is O(|S|^2) per candidate. With |S| up to ~48 and n=192, this is manageable.
    
    # But we also need to ensure we find the MAXIMUM cardinality. Greedy might not be optimal.
    # Let's use a backtracking approach that tries to include elements, and if stuck, backtracks.
    # However, with n=192, pure backtracking is too slow.
    
    # Alternative: Use a greedy approach with multiple orderings and take the best.
    # Or: Use a local search / simulated annealing.
    
    # Given the 60-second limit, let's try a greedy approach with a good heuristic, 
    # then try to improve by removing elements and re-adding others.
    
    # Actually, let's try a branch-and-bound style search but with strong pruning.
    # Sort candidates in descending order.
    # At each step, try to add the next candidate. If it violates constraints, skip it.
    # This is the current approach. Let's enhance it with:
    # 1. Better ordering (maybe by "difficulty" - elements that are hard to place first)
    # 2. After greedy, try to improve by swapping.
    
    # Let's first implement the greedy with descending order, then try local improvements.
    
    def is_valid_addition(a, S):
        """Check if adding a to S maintains the constraint."""
        S_set = set(S)
        # Check constraint for a: no 3 distinct elements in S with sum divisible by a
        if a == 1:
            # 1 divides everything, so we need no 3 elements in S (always true if |S|<3, but if |S|>=3, we need to check)
            # Actually, 1 divides any sum, so if |S| >= 3, adding 1 is invalid.
            if len(S) >= 3:
                return False
            # If |S| < 3, no triple exists, so it's valid for a's constraint
            pass
        else:
            # Check all triples in S
            for triple in combinations(S, 3):
                if sum(triple) % a == 0:
                    return False
        
        # Check constraint for each b in S: adding a might create a new triple (a, x, y) with x,y in S\{b}
        # such that b | (a + x + y)
        for b in S:
            # We need to check if there exist x, y in S such that x != y, x != b, y != b, and b | (a + x + y)
            # Note: x and y must be distinct from each other and from b and a.
            # Since a is not in S yet, x,y are just from S\{b}.
            found = False
            S_without_b = [v for v in S if v != b]
            for x_idx in range(len(S_without_b)):
                for y_idx in range(x_idx + 1, len(S_without_b)):
                    x = S_without_b[x_idx]
                    y = S_without_b[y_idx]
                    if b == 1:
                        # 1 divides everything, so if there are any 2 elements in S\{b}, the triple (a,x,y) sum is divisible by 1
                        # But wait, we need 3 DISTINCT OTHER elements. The triple is (a, x, y).
                        # These are 3 distinct elements, all different from b (since x,y != b and a != b as a not in S).
                        # So if b=1 and |S\{b}| >= 2, then b divides the sum.
                        found = True
                        break
                    else:
                        if (a + x + y) % b == 0:
                            found = True
                            break
                if found:
                    break
            if found:
                return False
        
        return True
    
    # Greedy approach: try candidates in descending order
    S = []
    for a in range(n, 0, -1):
        if is_valid_addition(a, S):
            S.append(a)
    
    base_size = len(S)
    
    # Try to improve by local search: remove an element and try to add others
    # Or try different orderings
    
    # Let's try a few different orderings
    import random
    best_S = S[:]
    best_size = base_size
    
    # Try ascending order
    S_asc = []
    for a in range(1, n + 1):
        if is_valid_addition(a, S_asc):
            S_asc.append(a)
    if len(S_asc) > best_size:
        best_S = S_asc
        best_size = len(S_asc)
    
    # Try random orderings (limited number due to time)
    random.seed(42)
    for trial in range(50):
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_rand = []
        for a in order:
            if is_valid_addition(a, S_rand):
                S_rand.append(a)
        if len(S_rand) > best_size:
            best_S = S_rand
            best_size = len(S_rand)
    
    # Try to improve best_S by swapping: remove one element, try to add multiple others
    # Simple hill climbing: try removing each element and see if we can add more
    improved = True
    while improved:
        improved = False
        for i in range(len(best_S)):
            a = best_S[i]
            S_temp = best_S[:i] + best_S[i+1:]
            # Try to add elements not in S_temp
            added = False
            for b in range(n, 0, -1):
                if b not in S_temp and is_valid_addition(b, S_temp):
                    S_temp.append(b)
                    added = True
                    break
            if len(S_temp) > len(best_S):
                best_S = S_temp
                improved = True
                break
    
    return best_S
# EVOLVE-BLOCK-END
