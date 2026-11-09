"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    Strategy: Analyze the prime factor structure of 255 = 3*5*17.
    Build conflict graph (edges where gcd=1) on candidates divisible by 3, 5, or 17.
    Use exact maximum independent set via branch-and-bound to find optimal mixed set.
    Fall back to multiples-of-3 baseline if search is too slow.
    """
    start_time = time.time()
    deadline = 80  # Safety margin before 90s limit
    
    # Build candidate set: numbers in [1,255] that share a factor with 255 = 3*5*17
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build conflict graph: conflict[i][j] = True if gcd(candidates[i], candidates[j]) == 1
    # We want maximum independent set in conflict graph = maximum clique in compatibility graph
    conflict = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) == 1:
                conflict[i][j] = True
                conflict[j][i] = True
    
    # Convert to neighbor sets for independent set (compatible neighbors)
    compat = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and not conflict[i][j]:
                compat[i].add(j)
    
    # Baseline: all multiples of 3
    mult3 = [i for i in range(1, 256) if i % 3 == 0]
    best_set = set(mult3)
    best_size = len(best_set)
    
    # Exact branch-and-bound for maximum independent set containing 255
    # In conflict graph, independent set = set with no conflicts = pairwise gcd > 1
    # We need 255 in the set, so we look for max independent set in conflict graph
    # restricted to vertices compatible with 255 (all candidates are, since they share factor with 255)
    
    # Use branch and bound with greedy coloring upper bound
    # Order vertices by degree in conflict graph (higher degree first for better bounds)
    degrees = [sum(conflict[i][j] for j in range(n) if j != i) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    # Map original index to position in order
    pos_in_order = {v: p for p, v in enumerate(order)}
    
    # Greedy coloring of conflict graph for upper bound
    # Color such that adjacent vertices (in conflict graph) get different colors
    # Max independent set size <= max color class size... actually we use it differently
    # Upper bound: greedy coloring gives that the remaining set can have at most (max_color) independent set
    
    def greedy_color(vertices):
        """Color the induced subgraph on vertices in conflict graph.
        Returns (coloring, max_color) where coloring[v] = color, max_color = number of colors used.
        This gives an upper bound on the independent set size in the subgraph."""
        coloring = {}
        color_of = {}
        next_color = 0
        for v in vertices:
            # Try to assign lowest available color
            used_colors = set()
            for u in vertices:
                if u != v and u in coloring and conflict[v][u]:
                    used_colors.add(coloring[u])
            c = 0
            while c in used_colors:
                c += 1
            coloring[v] = c
            next_color = max(next_color, c + 1)
        return coloring, next_color
    
    best_independent = set([idx_255])
    
    def branch_bound(remaining, current_set, upper_bound):
        nonlocal best_independent, best_size
        if time.time() - start_time > deadline:
            return
        
        if not remaining:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_independent = set(current_set)
            return
        
        if len(current_set) + upper_bound <= best_size:
            return
        
        # Pick a vertex to branch on
        v = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include v
        # All vertices in remaining that conflict with v must be excluded
        new_remaining = [u for u in rest if not conflict[v][u]]
        new_current = current_set + [v]
        # Recompute upper bound for new_remaining
        _, ub = greedy_color(new_remaining)
        branch_bound(new_remaining, new_current, ub)
        
        # Option 2: exclude v
        new_remaining2 = rest
        _, ub2 = greedy_color(new_remaining2)
        branch_bound(new_remaining2, current_set, ub2)
    
    # Initialize: must include 255, so start with 255 included
    # Remaining candidates are all others compatible with 255 (which is all of them)
    remaining = [v for v in order if v != idx_255]
    _, initial_ub = greedy_color(remaining)
    branch_bound(remaining, [idx_255], initial_ub)
    
    # Convert best_independent to values
    if best_independent:
        result = sorted([candidates[i] for i in best_independent])
    else:
        result = sorted(mult3)
    
    return result

# EVOLVE-BLOCK-END
