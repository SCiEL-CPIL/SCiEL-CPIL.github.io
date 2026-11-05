import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 192
    
    # Build conflict graph: edge between i,j if they cannot coexist
    # i and j conflict if:
    #   - i divides j + a + b for some distinct a,b in [1..n] \ {i,j}
    #   - j divides i + a + b for some distinct a,b in [1..n] \ {i,j}
    # But we only care about conflicts among selected elements.
    # Actually, the condition is: no selected integer x divides sum of any 3 DISTINCT OTHER selected integers.
    # So for a set S, for each x in S, for each {a,b,c} subset of S\{x} with |{a,b,c}|=3, x must not divide a+b+c.
    
    # This is a 4-uniform hypergraph constraint, not just pairwise.
    # However, we can model it as: two elements i and j are "compatible" if there exists a valid set containing both.
    # That's hard to precompute. Instead, let's use a different approach.
    
    # Key insight: The constraint is that for any 4-tuple {x, a, b, c} of distinct elements in S,
    # x must not divide a+b+c.
    
    # Let's use a branch-and-bound with a greedy upper bound.
    
    # Precompute: for each pair (i, j), determine if they can coexist in some valid set.
    # Two elements i and j conflict if there exist distinct a, b in [1..n] \ {i,j} such that:
    #   i | (j + a + b) OR j | (i + a + b)
    # But this is necessary but not sufficient for conflict in a larger set.
    
    # Actually, let's think about it differently. The constraint is local to 4 elements.
    # For a set S to be valid, every 4-subset {x, a, b, c} must satisfy: x ∤ (a+b+c).
    
    # Let's use a maximum independent set approach on a conflict graph where
    # we define conflict more carefully.
    
    # Alternative: Use simulated annealing or local search on the set.
    
    # Let's try a greedy + local search approach with bitset optimization.
    
    # Represent sets as bitsets for fast operations.
    
    def popcount(x):
        return bin(x).count('1')
    
    # Precompute sums of triples for quick lookup
    # For each x, and each pair (a,b) with a<b, compute a+b
    # Then x conflicts with {a,b,c} if c is such that x | (a+b+c)
    
    # Let's use a different strategy: greedy construction with backtracking.
    
    best_size = 0
    best_set = []
    
    # Helper: check if adding x to set S (represented as list) is valid
    def can_add(x, S):
        if len(S) < 3:
            # Need to check: does any element in S divide x + sum of 2 others?
            # And does x divide sum of any 3 others? (but len(S)<3, so no triple in S)
            # Actually if len(S) < 3, there are no 3 distinct others in S, so x dividing a triple sum is not an issue.
            # But we need: for each a in S, does a divide x + b + c for distinct b,c in S\{a}?
            # If len(S) >= 2, we need to check.
            if len(S) >= 2:
                S_list = list(S)
                for i, a in enumerate(S_list):
                    others = [v for j, v in enumerate(S_list) if j != i]
                    for b, c in combinations(others, 2):
                        if (x + b + c) % a == 0:
                            return False
            return True
        
        S_list = list(S)
        # Check 1: x does not divide sum of any 3 distinct others in S
        for a, b, c in combinations(S_list, 3):
            if (a + b + c) % x == 0:
                return False
        
        # Check 2: No a in S divides x + b + c for distinct b, c in S \ {a}
        for i, a in enumerate(S_list):
            others = [v for j, v in enumerate(S_list) if j != i]
            if len(others) < 2:
                continue
            for b, c in combinations(others, 2):
                if (x + b + c) % a == 0:
                    return False
        return True
    
    # Greedy with multiple random orders
    def greedy(order):
        S = []
        S_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if can_add(x, S):
                S.append(x)
                S_set.add(x)
        return S
    
    # Local search: try to improve a set by swapping elements
    def local_search(S, max_iters=100):
        S = list(S)
        S_set = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try removing each element and adding a new one
            for i in range(len(S)):
                S_no_i = S[:i] + S[i+1:]
                S_no_i_set = S_set - {S[i]}
                # Try adding each candidate not in S_no_i
                for x in range(1, n+1):
                    if x in S_no_i_set:
                        continue
                    if can_add(x, S_no_i):
                        # Check if this improves: we want larger sets, but size is same after swap
                        # Actually swap keeps size same. We need to try to grow.
                        pass
            # Try growing: add any element that fits
            for x in range(1, n+1):
                if x in S_set:
                    continue
                if can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
        return S
    
    # Strategy 1: Greedy with different orderings
    orders = []
    # Descending
    orders.append(list(range(n, 0, -1)))
    # Ascending
    orders.append(list(range(1, n+1)))
    # Random
    for _ in range(20):
        o = list(range(1, n+1))
        random.shuffle(o)
        orders.append(o)
    # Biased large
    for _ in range(20):
        large = list(range(n, n//2, -1))
        small = list(range(n//2, 0, -1))
        random.shuffle(large)
        random.shuffle(small)
        orders.append(large + small)
    
    for order in orders:
        if time.time() > deadline - 2:
            break
        S = greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            if best_size >= 48:
                break
    
    # Strategy 2: Local search from best found
    if best_set and time.time() < deadline - 2:
        S_improved = local_search(best_set, max_iters=50)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved
    
    # Strategy 3: More random restarts with local search
    while time.time() < deadline - 2:
        order = list(range(1, n+1))
        random.shuffle(order)
        S = greedy(order)
        S = local_search(S, max_iters=30)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            if best_size >= 48:
                break
    
    return best_set
