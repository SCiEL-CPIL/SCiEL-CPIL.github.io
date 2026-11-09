"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [2, 256] such that lcm(a, b) > 256 for all distinct pairs a, b.
    Maximize the exact reciprocal sum (sum of 1/x for x in the subset).
    
    Uses exact maximum-weight independent set via branch-and-bound with greedy coloring bounds.
    """
    from math import gcd
    from fractions import Fraction

    n = 256
    numbers = list(range(2, n + 1))
    num_count = n - 1  # 255 numbers
    
    # Precompute weights
    weights = [Fraction(1, x) for x in numbers]
    
    # Precompute conflict: conflict[i] = set of j where lcm(numbers[i], numbers[j]) <= 256
    conflict = [set() for _ in range(num_count)]
    for i in range(num_count):
        a = numbers[i]
        for j in range(i + 1, num_count):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Build compatibility graph: compat[i] = set of j where i != j and j not in conflict[i]
    compat = [set() for _ in range(num_count)]
    for i in range(num_count):
        for j in range(num_count):
            if i != j and j not in conflict[i]:
                compat[i].add(j)
    
    # Maximum weight clique in compat graph using Bron-Kerbosch with pivot and coloring
    
    best_weight = Fraction(0, 1)
    best_clique = []
    
    def max_weight_clique_branch_and_bound(candidates, excluded, current_weight, current_clique):
        """
        Branch and bound for maximum weight clique.
        candidates: set of vertices that can be added
        excluded: set of vertices that cannot be added (already considered)
        """
        nonlocal best_weight, best_clique
        
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_clique = list(current_clique)
            return
        
        # Greedy coloring of candidates in the compat graph
        # Order by descending weight for better bounds
        ordered = sorted(candidates, key=lambda v: -weights[v])
        
        # Assign colors: two vertices get same color if they are NOT adjacent in compat graph
        color_of = {}
        color_classes = []
        
        for v in ordered:
            assigned = None
            for c in range(len(color_classes)):
                ok = True
                for u in color_classes[c]:
                    if u in compat[v]:
                        ok = False
                        break
                if ok:
                    assigned = c
                    break
            if assigned is None:
                assigned = len(color_classes)
                color_classes.append(set())
            color_classes[assigned].add(v)
            color_of[v] = assigned
        
        # Compute suffix bound
        suffix_bound = [Fraction(0, 1)] * (len(ordered) + 1)
        color_max = {}
        for i in range(len(ordered) - 1, -1, -1):
            v = ordered[i]
            c = color_of[v]
            if c not in color_max or weights[v] > color_max[c]:
                color_max[c] = weights[v]
            suffix_bound[i] = sum(color_max.values())
        
        # Pivot selection
        all_v = candidates | excluded
        pivot = None
        max_deg = -1
        for u in all_v:
            deg = len(candidates & compat[u])
            if deg > max_deg:
                max_deg = deg
                pivot = u
        
        if pivot is None:
            return
        
        if pivot in candidates:
            to_process = candidates - compat[pivot]
        else:
            to_process = candidates
        
        for i, v in enumerate(ordered):
            if v not in to_process:
                continue
            
            if current_weight + suffix_bound[i] <= best_weight:
                break
            
            new_candidates = candidates & compat[v]
            new_excluded = excluded & compat[v]
            current_clique.append(v)
            max_weight_clique_branch_and_bound(new_candidates, new_excluded, 
                                               current_weight + weights[v], current_clique)
            current_clique.pop()
            
            candidates.discard(v)
            excluded.add(v)
    
    all_vertices = set(range(num_count))
    max_weight_clique_branch_and_bound(all_vertices, set(), Fraction(0, 1), [])
    
    result = [numbers[i] for i in best_clique]
    if not result:
        result = [2]
    
    return result
# EVOLVE-BLOCK-END
