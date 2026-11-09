"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    # Build compatibility graph: edge (i,j) exists iff lcm(i,j) > 256
    n = 255  # numbers 2..256
    # Precompute gcd table for speed
    # Use compatibility: compatible if lcm(a,b) > 256, i.e., a*b/gcd(a,b) > 256
    
    # weights = 1/x for x in 2..256
    xs = list(range(2, 257))
    weights = [Fraction(1, x) for x in xs]
    
    # Build adjacency as bitmasks (Python ints)
    # adj[i] is a bitmask of all j compatible with i
    adj = [0] * n
    for i in range(n):
        ai = xs[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            aj = xs[j]
            # lcm > 256
            g = math.gcd(ai, aj)
            if ai // g * aj > 256:
                mask |= (1 << j)
        adj[i] = mask
    
    # Greedy coloring for upper bound (weight-based)
    # We use a weighted coloring: sort vertices, assign colors
    # For maximum weight clique, use branch and bound with coloring
    
    # Bron-Kerbosch with pivoting + greedy coloring bounds
    # Since we want MAXIMUM WEIGHT clique, we use a weight-aware coloring
    
    # Precompute: order vertices by weight descending for better bounds
    # Actually for clique we work with the graph directly.
    
    # Weighted coloring: assign color numbers such that vertices of same color are not adjacent
    # Upper bound for a candidate set R + P: sum of top weights per color class
    
    def greedy_coloring(vertices, adj_list):
        """Assign colors to vertices (indices) such that adjacent vertices get different colors.
        Returns color list where color[v] is the color index (0-based).
        Also returns for each color the sum of weights of vertices with that color (for bound)."""
        # Sort by weight descending for tighter bounds
        order = sorted(vertices, key=lambda v: weights[v], reverse=True)
        color = {}
        # color_classes: list of sets
        color_classes = []
        for v in order:
            # Find smallest color not used by neighbors
            neighbor_colors = set()
            for c_idx, cls in enumerate(color_classes):
                if any((adj_list[v] >> u) & 1 for u in cls):
                    neighbor_colors.add(c_idx)
            # Assign smallest available
            c = 0
            while c in neighbor_colors:
                c += 1
            if c == len(color_classes):
                color_classes.append(set())
            color_classes[c].add(v)
            color[v] = c
        # Bound: for a set of vertices, the max weight clique <= sum over colors of max weight in that color
        # Actually we need: for any clique, it can have at most one vertex per color.
        # So upper bound = sum over colors of (max weight among vertices of that color in the candidate set)
        return color, color_classes
    
    # For efficiency, precompute colors for the full graph once
    # But the bound needs to be computed for subsets P.
    # Standard approach: at each node, color P greedily, then bound = sum of max weights per color in P.
    
    # To speed up, we precompute a global coloring and use it as a bound (valid since any subset's 
    # coloring can reuse the global one - a clique in subset is still independent in the coloring).
    # Actually a global coloring gives a valid bound for any subset: clique in subset <= sum of top weights 
    # per color restricted to subset. But computing "top weight per color in subset" each node is costly.
    
    # Alternative: use the standard maximum weight clique B&B with greedy coloring at each node.
    # With n=255 this might be slow. Let's use a hybrid: 
    #   - Get a good initial solution via greedy/heuristic.
    #   - Use B&B with coloring bounds.
    
    # First, get a good lower bound via greedy heuristic
    def greedy_heuristic():
        # Greedy: repeatedly pick vertex with max weight that is compatible with all chosen
        chosen = []
        remaining = set(range(n))
        while remaining:
            # Pick max weight vertex compatible with all chosen
            best = None
            best_w = Fraction(0)
            for v in remaining:
                if all((adj[v] >> u) & 1 for u in chosen):
                    if weights[v] > best_w:
                        best_w = weights[v]
                        best = v
            if best is None:
                break
            chosen.append(best)
            remaining.discard(best)
        return chosen, sum(weights[v] for v in chosen)
    
    best_clique, best_weight = greedy_heuristic()
    
    # Also try: greedy by density, and local search
    # Local search: try to add/remove to improve
    def local_search(clique):
        clique = set(clique)
        improved = True
        while improved:
            improved = False
            # Try adding
            for v in range(n):
                if v in clique:
                    continue
                if all((adj[v] >> u) & 1 for u in clique):
                    clique.add(v)
                    improved = True
            # Try swapping: remove one, add two
            # Skip for now
        return list(clique), sum(weights[v] for v in clique)
    
    best_clique, best_weight = local_search(best_clique)
    
    # Now B&B with coloring bounds
    # We'll do a recursive search. To handle time, we check deadline.
    
    # Precompute a global ordering for branching: sort by weight desc
    # Branch on vertices in order of weight descending
    
    # For coloring bound at each node, we do greedy coloring on P.
    # To make it fast, we limit: if |P| is large, the bound is loose but we prune by time.
    
    # Use a stack-based (iterative) B&B to avoid recursion depth issues
    # State: (R, P, bound)
    # We process nodes in order of bound (best-first) to find good solutions early.
    
    # Actually, let's do depth-first with good branching order and coloring bounds.
    
    # Precompute: for each vertex, list of neighbors (for coloring speed)
    # adj is already bitmask.
    
    # Greedy coloring function optimized:
    def color_and_bound(P):
        """P is a set (or list) of vertex indices. Returns upper bound on max weight clique in P."""
        if not P:
            return Fraction(0)
        # Greedy coloring: process vertices in order of weight descending
        verts = sorted(P, key=lambda v: weights[v], reverse=True)
        # color_classes: list of sets
        color_classes = []
        # For speed, track for each color the set of vertices
        for v in verts:
            # Check which colors conflict
            conflict = [False] * len(color_classes)
            for c_idx, cls in enumerate(color_classes):
                # Check if v adjacent to any in cls
                # Use bitmask: adj[v] & mask_of_cls
                # Precompute mask for cls? That's costly. Instead check each.
                # For small cls this is fine.
                found = False
                for u in cls:
                    if (adj[v] >> u) & 1:
                        found = True
                        break
                if found:
                    conflict[c_idx] = True
            # Find smallest non-conflicting color
            c = 0
            while c < len(color_classes) and conflict[c]:
                c += 1
            if c == len(color_classes):
                color_classes.append({v})
            else:
                color_classes[c].add(v)
        # Bound: sum over colors of max weight in that color
        bound = Fraction(0)
        for cls in color_classes:
            mw = max(weights[v] for v in cls)
            bound += mw
        return bound
    
    # B&B search
    # We branch: pick a vertex v from P, recurse with R∪{v}, P∩N(v), and P\{v}
    # To order: process the "include v" branch first if v has high weight.
    
    # Use recursion with a time check.
    # To avoid deep recursion, we use an explicit stack but DFS order.
    
    # Let's do a cleaner recursive DFS with coloring bound.
    # Sort P by weight descending at each node for better bounds.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def search(R, P):
        nonlocal best_clique, best_weight
        if time.time() > DEADLINE:
            return
        # If P empty, update best
        if not P:
            w = sum(weights[v] for v in R)
            if w > best_weight:
                best_weight = w
                best_clique = list(R)
            return
        # Bound
        bound = color_and_bound(P)
        cur = sum(weights[v] for v in R)
        if cur + bound <= best_weight:
            return
        # Also, if cur + sum of all weights in P <= best_weight, prune
        # (bound already covers this since bound <= sum of weights in P)
        
        # Pick a pivot vertex to branch on
        # Choose vertex with max weight in P for good branching
        verts = sorted(P, key=lambda v: weights[v], reverse=True)
        v = verts[0]
        
        # Branch 1: include v
        new_R = R + [v]
        # P' = P ∩ N(v)
        P_incl = set()
        for u in P:
            if (adj[v] >> u) & 1:
                P_incl.add(u)
        search(new_R, P_incl)
        
        # Branch 2: exclude v
        P_excl = set(P)
        P_excl.discard(v)
        search(R, P_excl)
    
    # Start search
    # Initial P = all vertices
    # To get a good initial bound, we already have greedy solution.
    # Start with empty R, P = all vertices.
    # But this could be huge. Let's seed with the greedy clique as a starting R?
    # No, standard B&B starts from empty. But we can use the greedy solution to prune.
    
    # Actually, to make it efficient, let's do a "branch and bound" that starts by 
    # considering the greedy solution's vertices as a guide.
    
    # Simpler: just run the search from empty R with P = all vertices.
    # The coloring bound should prune a lot.
    
    # But n=255, the initial P has 255 vertices. color_and_bound on 255 vertices 
    # each node could be slow. Let's optimize: precompute a global coloring.
    
    # Global coloring: color the whole graph once.
    # Then for any subset P, the bound = sum over colors of (max weight in P ∩ color_class).
    # This is a valid bound and much faster to compute if we precompute color classes.
    
    # Let's do global coloring.
    def global_coloring():
        order = sorted(range(n), key=lambda v: weights[v], reverse=True)
        color_classes = []
        for v in order:
            conflict = [False] * len(color_classes)
            for c_idx, cls in enumerate(color_classes):
                found = False
                for u in cls:
                    if (adj[v] >> u) & 1:
                        found = True
                        break
                if found:
                    conflict[c_idx] = True
            c = 0
            while c < len(color_classes) and conflict[c]:
                c += 1
            if c == len(color_classes):
                color_classes.append({v})
            else:
                color_classes[c].add(v)
        return color_classes
    
    color_classes = global_coloring()
    
    # For bound: sum over color classes of max weight in P ∩ class
    # To compute fast, for each color class, we need max weight among P ∩ class.
    # Precompute for each color class a sorted list of (weight, vertex) so we can 
    # quickly find max in P.
    # Actually, since weights are fixed, for each color class, sort vertices by weight desc.
    # Then max weight in P ∩ class = first vertex in sorted order that's in P.
    
    # Precompute sorted lists per color class
    color_sorted = []
    for cls in color_classes:
        sorted_cls = sorted(cls, key=lambda v: weights[v], reverse=True)
        color_sorted.append(sorted_cls)
    
    def bound_from_coloring(P_set):
        bound = Fraction(0)
        for sorted_cls in color_sorted:
            for v in sorted_cls:
                if v in P_set:
                    bound += weights[v]
                    break
        return bound
    
    # Now B&B with this faster bound.
    def search2(R, P_set):
        nonlocal best_clique, best_weight
        if time.time() > DEADLINE:
            return
        if not P_set:
            w = sum(weights[v] for v in R)
            if w > best_weight:
                best_weight = w
                best_clique = list(R)
            return
        cur = sum(weights[v] for v in R)
        bound = bound_from_coloring(P_set)
        if cur + bound <= best_weight:
            return
        # Pick pivot: max weight in P_set
        # Find max weight vertex in P_set
        best_v = None
        best_w = Fraction(0)
        for v in P_set:
            if weights[v] > best_w:
                best_w = weights[v]
                best_v = v
        if best_v is None:
            return
        v = best_v
        # Branch include
        new_R = R + [v]
        P_incl = set()
        for u in P_set:
            if (adj[v] >> u) & 1:
                P_incl.add(u)
        search2(new_R, P_incl)
        # Branch exclude
        P_excl = set(P_set)
        P_excl.discard(v)
        search2(R, P_excl)
    
    # Start search
    all_vertices = set(range(n))
    search2([], all_vertices)
    
    # Final local search on best clique
    best_clique, best_weight = local_search(best_clique)
    
    # Verify and return
    # Convert back to actual numbers
    result = [xs[v] for v in best_clique]
    # Verify
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            if math.lcm(a, b) <= 256:
                # Should not happen, but fallback to greedy
                return xs[:]  # fallback
    return result
# EVOLVE-BLOCK-END
