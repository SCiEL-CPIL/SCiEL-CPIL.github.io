def construct():
    import time
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        """Solve minimum clique cover for a split graph."""
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # K = vertices 0..k-1, C = vertices k..n-1
        # For each c in C, compute its neighborhood in K
        c_neighbors = []
        for c in range(k, n):
            nbrs = frozenset(adj[c])  # all neighbors are in K
            c_neighbors.append(nbrs)
        
        # Group C-vertices by their neighborhood
        from collections import defaultdict
        groups = defaultdict(list)
        for i, c in enumerate(range(k, n)):
            nbrs = c_neighbors[i]
            groups[nbrs].append(c)
        
        # We need to cover:
        # 1. All K-edges: edges within K
        # 2. All cross edges: edges between K and C
        # 
        # Strategy: For each group of C-vertices with neighborhood S,
        # we form cliques that cover cross edges and K-edges within S.
        #
        # For a group with m C-vertices and neighborhood S (|S|=d):
        # - Total cross edges to cover: m * d
        # - K-edges within S: C(d,2)
        #
        # Optimal: Use cliques of size 3 {c, s1, s2} where s1,s2 in S
        # Each such clique covers 2 cross edges and 1 K-edge.
        # 
        # But we need to be careful: each K-edge can only be in ONE clique.
        # And each cross edge (c,s) must be in exactly one clique.
        #
        # Approach:
        # - For each C-vertex c with neighborhood S, we need to partition S into pairs (and possibly one singleton)
        # - Each pair {s1,s2} forms clique {c, s1, s2}
        # - If |S| is odd, one K-vertex is left, forming clique {c, s}
        #
        # But this doesn't account for sharing K-edges across C-vertices.
        # 
        # Better approach: Think of it as we need to assign each cross edge (c,s) to a clique.
        # A clique containing c and a subset T of S covers cross edges (c,t) for t in T
        # and K-edges within T.
        #
        # The constraint is that each K-edge (s1,s2) can appear in at most one clique.
        #
        # Optimal solution for split graphs:
        # For C-vertices with the same neighborhood S:
        # - We can form cliques that each contain exactly one C-vertex from the group
        #   and a subset of S
        # - To minimize cliques, we want each clique to cover as many cross edges as possible
        # - But we're limited by K-edge usage
        #
        # Let me think differently. The minimum number of cliques:
        # For each C-vertex c with |N(c)| = d:
        #   We need at least ceil(d/2) cliques containing c (each covers at most 2 cross edges from c)
        #   But if d is odd, one clique has just 1 cross edge
        #
        # Total cliques = sum over all c of ceil(|N(c)| / 2) ... but this double-counts K-edges
        #
        # Actually, the constraint is that K-edges are shared. Let me use a different approach.
        #
        # For the given parameters (p=0.7, k=10/20/30), most C-vertices will have large neighborhoods.
        # Let's use a greedy approach that's near-optimal.
        
        # Better algorithm:
        # 1. Sort C-vertices by neighborhood size (descending)
        # 2. For each C-vertex, assign its cross edges to cliques
        # 3. Track which K-edges are already used
        
        # Actually, let's use the known optimal structure:
        # For split graphs, the minimum clique cover can be found by:
        # - For each C-vertex, the number of cliques it participates in is ceil(d/2) where d=|N(c)|
        # - But we need to ensure K-edges aren't reused
        #
        # Let me just implement a clean solution:
        # For each distinct neighborhood S with m C-vertices:
        #   We need to cover m*|S| cross edges and C(|S|,2) K-edges (within S)
        #   Using cliques of form {c, s1, s2} (size 3) or {c, s} (size 2)
        #
        # Key insight: K-edges within S can be covered by cliques from ANY C-vertex with neighborhood S
        # So we should first use cliques to cover K-edges within S, then handle remaining cross edges.
        
        cliques = []
        
        # Track which K-edges are covered
        k_edges_covered = set()
        all_k_edges = [(u, v) for u in range(k) for v in range(u+1, k)]
        
        # For each group, process C-vertices
        # Strategy: For each group with neighborhood S:
        #   Phase 1: Cover K-edges within S using cliques with C-vertices
        #   Phase 2: Cover remaining cross edges
        
        # Let's process groups by size of S (larger first for efficiency)
        group_items = list(groups.items())
        group_items.sort(key=lambda x: len(x[0]), reverse=True)
        
        for S, c_list in group_items:
            S_list = sorted(S)
            d = len(S_list)
            m = len(c_list)
            
            if d == 0:
                # No cross edges, no K-edges to cover from this group
                continue
            
            if d == 1:
                # Each C-vertex has exactly 1 neighbor in K
                s = S_list[0]
                # All C-vertices can be in one clique with s? No, because they're not connected to each other
                # Each C-vertex needs its own clique: {c, s}
                for c in c_list:
                    cliques.append([c, s])
                continue
            
            # d >= 2
            # Cover K-edges within S first
            # Number of K-edges within S: C(d,2)
            # Each clique {c, s1, s2} covers 1 K-edge and 2 cross edges
            # We have m C-vertices available
            
            # Phase 1: Use C-vertices to cover K-edges within S
            # We need C(d,2) K-edges covered, each by one clique
            # Each clique uses 1 C-vertex and covers 2 cross edges
            # If m >= C(d,2), we can cover all K-edges with dedicated cliques
            # Otherwise, some K-edges must be covered by K-only cliques (size 2)
            
            num_k_edges = d * (d - 1) // 2
            num_c_available = m
            
            # Generate all K-edge pairs within S
            s_pairs = [(S_list[i], S_list[j]) for i in range(d) for j in range(i+1, d)]
            
            # For each K-edge, try to cover it with a C-vertex clique
            # Each C-vertex can be used multiple times (in different cliques)
            # Wait, no - a C-vertex can be in multiple cliques!
            # A C-vertex c can be in cliques {c, s1, s2}, {c, s3, s4}, etc.
            # as long as the K-edges (s1,s2), (s3,s4) are distinct.
            
            # So each C-vertex c with neighborhood S can participate in up to floor(d/2) 
            # size-3 cliques (pairing up its neighbors) or ceil(d/2) if we allow one size-2.
            
            # Optimal for one C-vertex: partition S into floor(d/2) pairs + (d mod 2) singletons
            # This uses ceil(d/2) cliques per C-vertex.
            
            # But we can share: if two C-vertices c1, c2 both have neighborhood S,
            # we can have clique {c1, s1, s2} and {c2, s3, s4} using different K-edges.
            
            # Total cross edges to cover: m * d
            # Total K-edges to cover: C(d,2) [but these might overlap with other groups! No, K-edges are global]
            
            # Wait, K-edges are global - a K-edge (s1,s2) can only be in ONE clique total.
            # So if multiple groups have s1,s2 in their neighborhood, only one clique can use that K-edge.
            
            # This makes it complex. Let me use a simpler approach that's still good.
            
            # Simple approach: For each C-vertex, create cliques covering its cross edges
            # by pairing up its neighbors. Track used K-edges.
            
            for c in c_list:
                # Get neighbors not yet "used up" for K-edges with this c
                # We need to cover all (c, s) for s in S
                # Pair up elements of S
                remaining = list(S_list)
                while remaining:
                    if len(remaining) >= 2:
                        s1, s2 = remaining[0], remaining[1]
                        remaining = remaining[2:]
                        k_edge = (min(s1,s2), max(s1,s2))
                        if k_edge not in k_edges_covered:
                            cliques.append([c, s1, s2])
                            k_edges_covered.add(k_edge)
                        else:
                            # K-edge already covered, use two separate cliques
                            cliques.append([c, s1])
                            cliques.append([c, s2])
                    else:
                        s = remaining[0]
                        remaining = []
                        cliques.append([c, s])
            
            # After processing all C-vertices in this group, cover any remaining K-edges within S
            # that weren't covered
            for s1, s2 in s_pairs:
                k_edge = (min(s1,s2), max(s1,s2))
                if k_edge not in k_edges_covered:
                    cliques.append([s1, s2])
                    k_edges_covered.add(k_edge)
        
        # Cover any remaining K-edges not covered by any group
        for u, v in all_k_edges:
            if (u, v) not in k_edges_covered:
                cliques.append([u, v])
                k_edges_covered.add((u, v))
        
        # Verify: every clique must have at least 2 vertices and be a valid clique
        # Every edge must be in exactly one clique
        # Let's verify and fix if needed
        
        # Check that all vertices in cliques are distinct and cliques are valid
        cleaned_cliques = []
        for cl in cliques:
            cl = list(set(cl))  # ensure distinct
            if len(cl) >= 2:
                # Verify it's a clique
                valid = True
                for i in range(len(cl)):
                    for j in range(i+1, len(cl)):
                        if cl[j] not in adj[cl[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    cleaned_cliques.append(sorted(cl))
        
        return cleaned_cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE - 5:
            # Not enough time, use fallback
            break
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)
    
    # If we ran out of time, fill with fallback
    while len(results) < 3:
        idx = len(results)
        spec = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))[idx]
        # Fallback: just use edge list as cliques
        edges = instance(*spec)
        results.append([list(e) for e in edges])
    
    return results
