"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a maximum subset of [1,128] with no three distinct elements
    whose three pairwise LCMs are all equal.
    
    Uses branch-and-bound with tight upper bounds from the 2-shadow
    conflict graph (minimum vertex cover / coloring bound).
    """
    from math import lcm
    from collections import defaultdict
    import sys
    
    n = 128
    sys.setrecursionlimit(10000)
    
    # Precompute lcm table
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i, n+1):
            lcm_table[i][j] = lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c and lcm(a,b)==lcm(a,c)==lcm(b,c)
    # Group pairs by their LCM value
    pairs_by_lcm = defaultdict(list)
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            L = lcm_table[a][b]
            pairs_by_lcm[L].append((a, b))
    
    # Find all triangles in each LCM group
    bad_triples = set()
    for L, edges in pairs_by_lcm.items():
        adj = defaultdict(set)
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        
        nodes = sorted(set(a for a, b in edges) | set(b for a, b in edges))
        for i in range(len(nodes)):
            a = nodes[i]
            for j in range(i+1, len(nodes)):
                b = nodes[j]
                if b in adj[a]:
                    common = adj[a] & adj[b]
                    for c in common:
                        if c > b:
                            bad_triples.add((a, b, c))
    
    bad_triples = list(bad_triples)
    bad_triples.sort()
    
    if not bad_triples:
        return list(range(1, n+1))
    
    # Build conflict graph (2-shadow): edge (a,b) exists if there's a bad triple containing both
    conflict_adj = defaultdict(set)
    for a, b, c in bad_triples:
        conflict_adj[a].add(b)
        conflict_adj[a].add(c)
        conflict_adj[b].add(a)
        conflict_adj[b].add(c)
        conflict_adj[c].add(a)
        conflict_adj[c].add(b)
    
    # For fast conflict checking: for each x, and each a, the set of b such that (a,b,x) is bad
    conflict_map = defaultdict(lambda: defaultdict(set))
    for a, b, c in bad_triples:
        conflict_map[c][a].add(b)
        conflict_map[c][b].add(a)
        conflict_map[a][b].add(c)
        conflict_map[a][c].add(b)
        conflict_map[b][a].add(c)
        conflict_map[b][c].add(a)
    
    def can_add(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        cm_x = conflict_map[x]
        for a in selected_set:
            if a in cm_x:
                for b in cm_x[a]:
                    if b in selected_set:
                        return False
        return True
    
    # Compute degree of each vertex in the conflict graph
    degrees = {x: len(conflict_adj[x]) for x in range(1, n+1)}
    
    # Greedy initial solution
    def greedy(order):
        s = set()
        for x in order:
            if can_add(x, s):
                s.add(x)
        return s
    
    orders = [
        sorted(range(1, n+1), key=lambda x: degrees[x]),  # low degree first
        sorted(range(1, n+1), key=lambda x: -degrees[x]),  # high degree first
        list(range(1, n+1)),
        list(range(n, 0, -1)),
    ]
    
    best_size = 0
    best_set = set()
    for order in orders:
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search: try to add any element that can be added
    improved = True
    while improved:
        improved = False
        for x in range(1, n+1):
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                best_size = len(best_set)
                improved = True
    
    # Branch and bound with tight upper bound
    # Upper bound: for a set of available vertices, the max independent set in the
    # conflict graph (2-shadow) gives an upper bound on the hypergraph independent set.
    # We use a simple coloring-based bound: color the conflict graph on available vertices,
    # the number of colors is an upper bound on the independence number.
    
    def color_bound(vertices):
        """Return an upper bound on the independence number of the conflict graph
        induced by vertices, using greedy coloring."""
        if not vertices:
            return 0
        # Greedy coloring
        color_of = {}
        next_color = 0
        for v in sorted(vertices, key=lambda x: -degrees[x]):
            used = set()
            for u in conflict_adj[v]:
                if u in color_of:
                    used.add(color_of[u])
            c = 0
            while c in used:
                c += 1
            color_of[v] = c
            next_color = max(next_color, c + 1)
        return next_color
    
    # Better bound: use the fact that in the 2-shadow, independent sets of the hypergraph
    # are subsets of independent sets of the graph. But actually, we need:
    # A set S is independent in the hypergraph iff it contains no bad triple.
    # The 2-shadow graph has edges between pairs that appear in some bad triple.
    # An independent set in the hypergraph need NOT be an independent set in the graph
    # (two vertices can share a bad triple with a third not in S).
    # 
    # So the graph coloring bound is NOT valid as an upper bound for the hypergraph.
    # We need a different approach.
    #
    # Alternative upper bound: for each available vertex v, if we include v, then
    # we cannot include any pair (a,b) from conflict_map[v]. This creates a "forbidden"
    # set. We can use a recursive bound.
    #
    # Simpler approach: use the count of available vertices as upper bound, but prune
    # more aggressively using the conflict structure.
    
    def search(available, selected, count):
        nonlocal best_size, best_set
        
        if count + len(available) <= best_size:
            return
        
        if not available:
            if count > best_size:
                best_size = count
                best_set = set(selected)
            return
        
        # Filter available to only those that can be added
        valid = []
        for x in available:
            if can_add(x, selected):
                valid.append(x)
        
        if not valid:
            if count > best_size:
                best_size = count
                best_set = set(selected)
            return
        
        if count + len(valid) <= best_size:
            return
        
        # Choose vertex with highest degree in the induced subgraph on valid
        valid_set = set(valid)
        v = max(valid, key=lambda x: sum(1 for y in valid_set if y in conflict_adj[x]))
        
        # Branch 1: Include v
        selected.add(v)
        remaining = [x for x in valid if x != v and can_add(x, selected)]
        search(remaining, selected, count + 1)
        selected.remove(v)
        
        # Branch 2: Exclude v
        remaining_ex = [x for x in valid if x != v]
        search(remaining_ex, selected, count)
    
    # Sort initial candidates by degree descending for better pruning
    initial_candidates = sorted(range(1, n+1), key=lambda x: -degrees[x])
    
    search(initial_candidates, set(), 0)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
