"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    n = 256
    # Weights: reciprocal of each element
    weights = [Fraction(1, i) for i in range(1, n+1)]
    
    # Precompute for each element x, the set of pairs (a,b) such that a,b,x form a 3-AP
    # x is in a 3-AP if there exist a < b < x with b-a = x-b, i.e., a = 2b-x
    # Or a < x < b with x-a = b-x, i.e., b = 2x-a
    # Or x < a < b with a-x = b-a, i.e., b = 2a-x
    
    # For conflict checking: if we select x, we cannot have both a and b where {a,b,x} is a 3-AP
    # Precompute: for each x, list of (a,b) pairs that conflict with x
    # Actually, better: for each pair (a,b) with a<b, the third element c is determined:
    #   if 2b-a <= n: c = 2b-a (x = c, so selecting a,b,c together is bad)
    #   if 2a-b >= 1: c = 2a-b (x = c)
    #   if 2a+b ... wait, the middle term: if a<c<b and c=(a+b)/2
    
    # Let me think differently. A 3-AP is (a, a+d, a+2d) for d>=1.
    # For each such triple, we can't select all three.
    # 
    # For B&B: maintain selected set S. When considering adding x, check if adding x creates a 3-AP.
    # x creates a 3-AP with a,b in S if:
    #   x is the middle: a + b = 2x (i.e., a = 2x-b for some b in S, and a in S)
    #   x is the end: x = a+2d, b = a+d => x-b = b-a => a = 2b-x (b in S, a in S)
    #   x is the start: x, x+d, x+2d => if x+d and x+2d are in S
    #
    # More efficiently: when we have selected set S and want to add x:
    # Check if there exist a,b in S such that {a,b,x} forms a 3-AP.
    # This happens if:
    #   - 2x - a is in S for some a in S (x is middle, a is one end)
    #   - 2a - x is in S for some a in S (x is an end, a is the other end or middle)
    #
    # Actually simplest: for each a in S, check if 2a-x is in S (and != a, != x, >=1)
    # and if 2x-a is in S (and != a, != x, <=n)
    
    # For efficiency, use a set for selected elements
    # Process elements in order of decreasing weight (i.e., increasing value)
    # But we also need to be careful about which elements to branch on
    
    # Order: process from 1 to 256 (smaller = higher weight)
    # But for B&B, we should branch on the most impactful decisions
    
    # Let's use a different approach: recursive B&B with inclusion/exclusion
    # State: current selected set (as a set), current weight, index of next element to consider
    # For each element, decide include or exclude
    
    # Optimization: if including x would create a 3-AP, skip it
    # Upper bound: current weight + sum of weights of all remaining elements that could be added
    
    # Let's precompute suffix sums of weights for upper bounding
    # But we need to account for conflicts... a simple upper bound is:
    # current_weight + sum(weights[i] for i in remaining elements)
    # This is valid but loose. Better: for each remaining element, if it conflicts with selected, exclude it from bound
    
    # For a tighter bound, we can note that if x is in S, then for any pair (a,b) where {a,b,x} is 3-AP,
    # we can't have both a and b. But computing this exactly is complex.
    # Start with the simple bound and see if it's fast enough.
    
    # Convert to list for indexing
    elements = list(range(1, n+1))
    w = [Fraction(1, i) for i in elements]
    
    # Precompute conflict pairs: for each x, which pairs (i,j) in indices conflict
    # Actually, let's just check on the fly using a set
    
    best_weight = Fraction(0)
    best_set = set()
    
    # We'll do DFS with B&B
    # To make it efficient, process in order 1, 2, ..., 256
    # At each step, try to include the current element if possible, then exclude it
    
    selected = set()
    
    def upper_bound(idx, selected_set, current_weight):
        """Compute an upper bound on the total weight we can achieve from elements idx..n-1"""
        # Simple bound: current_weight + sum of all remaining weights
        # Tighter: for each remaining element, check if it can be added (no conflict with selected_set)
        # If it conflicts, it contributes 0 to the bound
        # This is still an upper bound because we're being optimistic about future conflicts
        ub = current_weight
        for j in range(idx, n):
            x = elements[j]
            # Check if x can be added to selected_set without creating a 3-AP
            # x conflicts if there exist a,b in selected_set such that {a,b,x} is a 3-AP
            can_add = True
            for a in selected_set:
                # Check if 2*a - x is in selected_set (and valid)
                c = 2*a - x
                if c != a and c != x and 1 <= c <= n and c in selected_set:
                    can_add = False
                    break
                # Check if 2*x - a is in selected_set (and valid)
                c = 2*x - a
                if c != a and c != x and 1 <= c <= n and c in selected_set:
                    can_add = False
                    break
            if can_add:
                ub += w[j]
        return ub
    
    def dfs(idx, selected_set, current_weight):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
        
        if idx == n:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(selected_set)
            return
        
        x = elements[idx]
        
        # Check if we can include x
        can_include = True
        for a in selected_set:
            c = 2*a - x
            if c != a and c != x and 1 <= c <= n and c in selected_set:
                can_include = False
                break
            c = 2*x - a
            if c != a and c != x and 1 <= c <= n and c in selected_set:
                can_include = False
                break
        
        # Upper bound if we skip x (and all subsequent)
        # Try including x first (higher weight for smaller x)
        if can_include:
            new_weight = current_weight + w[idx]
            # Upper bound check: even if we add all remaining compatible elements, can we beat best?
            # Quick check: new_weight + sum of remaining weights
            if new_weight + sum(w[idx+1:]) > best_weight:
                dfs(idx+1, selected_set | {x}, new_weight)
        
        # Try excluding x
        # Upper bound: current_weight + sum of remaining weights
        if current_weight + sum(w[idx+1:]) > best_weight:
            dfs(idx+1, selected_set, current_weight)
    
    # The above is too slow due to O(|S|) conflict checks and O(n) upper bound computation
    # Let me optimize significantly
    
    # Better approach: 
    # 1. Use a boolean array for selected
    # 2. Precompute for each x, the list of (a,b) pairs that would form a 3-AP with x
    # 3. Use bitsets for fast conflict checking
    
    # Actually, let's use a completely different strategy:
    # Greedy with local search / simulated annealing
    
    # Start with the greedy solution (which is the seed)
    # Then try to improve by local search
    
    # Greedy: add elements in order 1,2,...,256 if no 3-AP is formed
    greedy_set = set()
    for x in range(1, n+1):
        can_add = True
        for a in greedy_set:
            c = 2*a - x
            if c != a and c != x and 1 <= c <= n and c in greedy_set:
                can_add = False
                break
            c = 2*x - a
            if c != a and c != x and 1 <= c <= n and c in greedy_set:
                can_add = False
                break
        if can_add:
            greedy_set.add(x)
    
    greedy_weight = sum(Fraction(1, x) for x in greedy_set)
    
    # Now do local search: try to swap out one element for another to improve weight
    # Or try to add elements that were previously excluded
    # Or remove an element to allow adding multiple others
    
    # Let's do a more systematic B&B but with better pruning
    
    # Key optimization: process in order, use a list for selected, and precompute conflicts
    
    # For each element x, precompute which other elements it conflicts with in terms of 3-APs
    # x conflicts with pair (a,b) if {a,b,x} is a 3-AP
    # For fixed x, the conflicting pairs are:
    #   (2x-d, x+d) for d in 1..min(x-1, n-x)  [x is middle]
    #   (x-d, x+d) ... wait, that's the same
    #   Actually: a,b,x form 3-AP iff 2*middle = a+b
    #   If x is middle: a+b=2x, so pairs (x-d, x+d) for d=1..min(x-1, n-x)
    #   If x is left end: x, x+d, x+2d => pairs (x+d, x+2d) for d=1..(n-x)//2
    #   If x is right end: x-2d, x-d, x => pairs (x-2d, x-d) for d=1..x//2
    
    # So for each x, precompute the set of pairs that conflict with x
    # But for B&B, we need: given selected set S, can we add x?
    # x can be added iff no pair (a,b) in S conflicts with x
    # i.e., for all pairs in the conflict list of x, not both are in S
    
    # Let's precompute conflict_pairs[x] = list of (a,b) where a<b and {a,b,x} is a 3-AP
    conflict_pairs = [[] for _ in range(n+1)]
    for x in range(1, n+1):
        pairs = set()
        # x is middle: (x-d, x+d)
        for d in range(1, min(x-1, n-x)+1):
            a, b = x-d, x+d
            pairs.add((a, b))
        # x is left end: (x+d, x+2d)
        for d in range(1, (n-x)//2 + 1):
            a, b = x+d, x+2*d
            pairs.add((a, b))
        # x is right end: (x-2d, x-d)
        for d in range(1, x//2 + 1):
            a, b = x-2*d, x-d
            pairs.add((a, b))
        conflict_pairs[x] = list(pairs)
    
    # Now B&B with this precomputation
    # State: selected as a set, idx, current_weight
    # To check if x can be added: for each (a,b) in conflict_pairs[x], check if a in S and b in S
    
    # For upper bound, use: current_weight + sum of weights of remaining elements that don't conflict with S
    # This is O(n * avg_conflicts) per bound computation
    
    # Let's implement with careful optimization
    import sys
    sys.setrecursionlimit(10000)
    
    best_weight = greedy_weight
    best_set = set(greedy_set)
    
    # Try to improve via B&B
    # Process elements in order 1 to 256
    # Use iterative deepening or just DFS with good pruning
    
    def can_add_x(x, S):
        """Check if x can be added to set S without creating a 3-AP"""
        for a, b in conflict_pairs[x]:
            if a in S and b in S:
                return False
        return True
    
    def compute_upper_bound(idx, S, current_weight):
        """Upper bound on achievable weight from elements idx..n-1"""
        ub = current_weight
        for j in range(idx, n):
            x = elements[j]
            if can_add_x(x, S):
                ub += w[j]
        return ub
    
    # The above upper bound is O(n * conflicts) which might be too slow
    # Let's use a simpler bound: current_weight + sum(w[idx:])
    # And rely on the greedy solution being close to optimal
    
    # Actually, let's try a different approach: 
    # Start from the greedy solution and try local improvements
    # 1. Try to add any excluded element (if it doesn't conflict)
    # 2. Try to remove one element and add multiple others
    # 3. Try pairwise swaps
    
    # First, let's see what the greedy solution looks like
    # Then do local search
    
    # Local search: try to improve by considering each element not in the set
    # and seeing if removing one element from the set allows adding it (net gain)
    
    improved = True
    while improved:
        improved = False
        current_set = set(best_set)
        current_weight = best_weight
        
        # Try adding each excluded element
        for x in range(1, n+1):
            if x in current_set:
                continue
            if can_add_x(x, current_set):
                new_weight = current_weight + w[x-1]
                if new_weight > best_weight:
                    best_weight = new_weight
                    best_set = current_set | {x}
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try removing one element and adding another
        for x in range(1, n+1):
            if x not in current_set:
                continue
            # Try removing x
            new_set = current_set - {x}
            # Try adding any element not in new_set
            for y in range(1, n+1):
                if y in new_set or y == x:
                    continue
                if can_add_x(y, new_set):
                    new_weight = current_weight - w[x-1] + w[y-1]
                    if new_weight > best_weight:
                        best_weight = new_weight
                        best_set = new_set | {y}
                        improved = True
                        break
            if improved:
                break
    
    # Try removing one element and adding two others
    for x in range(1, n+1):
        if x not in best_set:
            continue
        new_set = best_set - {x}
        # Try adding pairs of elements
        candidates = [y for y in range(1, n+1) if y not in new_set and y != x]
        for i, y in enumerate(candidates):
            if can_add_x(y, new_set):
                set_with_y = new_set | {y}
                for z in candidates[i+1:]:
                    if z == y:
                        continue
                    if can_add_x(z, set_with_y):
                        new_weight = best_weight - w[x-1] + w[y-1] + w[z-1]
                        if new_weight > best_weight:
                            best_weight = new_weight
                            best_set = set_with_y | {z}
                            break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return list(best_set)
# EVOLVE-BLOCK-END
