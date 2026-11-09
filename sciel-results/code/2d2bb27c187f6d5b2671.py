def construct():
    import time
    import math

    deadline = time.time() + 55
    LIMIT = 512
    N = LIMIT

    # Build adjacency list
    # adj[i] is a set of nodes j such that lcm(i, j) <= 512
    adj = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i // math.gcd(i, j)) * j <= LIMIT:
                adj[i].add(j)
                adj[j].add(i)
    
    # Tomita's Max Clique Algorithm
    best_clique = []
    
    def color_sort(P):
        """
        Greedily color the subgraph induced by P.
        Returns a list of (vertex, color) sorted by color descending, then by vertex.
        The color of a vertex is an upper bound on the clique size in the subgraph 
        induced by that vertex and subsequent vertices in the sorted list.
        """
        if not P:
            return []
        
        P_list = list(P)
        # Sort by degree in P descending (Largest First)
        P_list.sort(key=lambda v: -len(adj[v] & P))
        
        colored = []
        colors = {}
        current_color = 0
        remaining = set(P_list)
        
        # Greedy coloring: assign colors 1, 2, ...
        # We process vertices in P_list order.
        # Actually, standard Tomita coloring processes vertices in a specific order.
        # Let's use the standard "color sort" where we assign colors to vertices 
        # such that vertices with the same color are independent.
        
        # Simpler implementation:
        # We want to partition P into independent sets $I_1, I_2, \dots$
        # such that any clique is contained in $\bigcup I_i$ and has at most 1 vertex from each $I_i$?
        # No, a clique can have at most 1 vertex from each independent set.
        # So if we color with $k$ colors, the max clique size is at most $k$.
        
        # Greedy Coloring Algorithm:
        # Iterate through vertices in P_list.
        # Assign the smallest possible color.
        
        # However, for Tomita, we usually order vertices by color.
        # Let's just do a simple greedy coloring and return the list of (v, color)
        # sorted by color.
        
        color_of = {}
        for v in P_list:
            # Find smallest color not used by neighbors in P_list
            used_colors = set()
            for u in adj[v]:
                if u in color_of:
                    used_colors.add(color_of[u])
            
            c = 1
            while c in used_colors:
                c += 1
            color_of[v] = c
        
        # Create list of (v, color)
        items = [(v, color_of[v]) for v in P_list]
        # Sort by color descending, then by degree descending (optional tie-breaker)
        items.sort(key=lambda x: (-x[1], -len(adj[x[0]] & P)))
        
        return items

    def expand(C, P):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if not P:
            if len(C) > len(best_clique):
                best_clique = list(C)
            return
        
        if len(C) + len(P) <= len(best_clique):
            return
            
        # Color sort P
        P_sorted = color_sort(P)
        
        # Iterate in reverse (highest color first)
        for i in range(len(P_sorted) - 1, -1, -1):
            v, c = P_sorted[i]
            
            # Bound: len(C) + c <= len(best_clique)
            if len(C) + c <= len(best_clique):
                return # All subsequent vertices have color <= c, so no need to continue
            
            # Branch on v
            C.add(v)
            new_P = P & adj[v]
            expand(C, new_P)
            C.discard(v)
            
            # Remove v from P for the next iteration (exclude v)
            P.discard(v)
            
            # Update P_sorted? 
            # The standard Tomita algorithm re-colors or adjusts.
            # For simplicity and correctness, we can just recurse with the modified P.
            # But we are in a loop. We need to handle the "exclude" branch.
            # The loop structure above effectively handles "include v" then "exclude v" 
            # by continuing the loop with P reduced.
            # However, the coloring `c` was calculated for the original P.
            # As we remove vertices, the upper bound `c` for subsequent vertices might be loose 
            # but still valid (it's an upper bound for the original set, so it's an upper bound for the subset).
            # So the bound check `len(C) + c <= len(best_clique)` is still safe.
            
            # We must be careful: `P` is being modified.
            # The `P_sorted` list contains vertices that might have been removed from P?
            # No, we only remove `v` from P. The other vertices in `P_sorted[i+1:]` are still in P.
            # Wait, `new_P` was `P & adj[v]`. This is a subset.
            # The exclude branch keeps `P` as is (minus v).
            # So the loop continues with `P` having `v` removed.
            # The next vertex `v_next` is from `P_sorted`. It is in `P`.
            # The bound `c_next` is the color assigned to `v_next` in the initial coloring of the original P.
            # Since the subgraph induced by the remaining vertices is a subgraph of the original,
            # the max clique size is bounded by the original max color.
            # So the pruning is valid.

    # Initial call
    # Start with empty clique, all nodes as candidates
    # To get a good initial bound, we can run a greedy heuristic first
    def greedy_clique():
        C = set()
        P = set(range(1, N + 1))
        while P:
            # Pick vertex with max degree in P
            v = max(P, key=lambda x: len(adj[x] & P))
            C.add(v)
            P = P & adj[v]
        return C

    init_clique = greedy_clique()
    best_clique = list(init_clique)
    
    # Run Tomita
    expand(set(), set(range(1, N + 1)))
    
    return best_clique
