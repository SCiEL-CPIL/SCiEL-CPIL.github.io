"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    from random import Random
    from itertools import combinations
    from collections import defaultdict

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split_clique_partition(n, k, p, seed):
        # Generate the graph
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        all_edges = edges + cross_edges
        
        # Build adjacency structure for the clique part (0..k-1)
        # In a split graph, any clique is either:
        # 1. A subset of the clique part (0..k-1)
        # 2. A subset of the clique part plus some independent set vertices that are connected to all in the subset
        
        # Let's identify for each subset of the clique part, which independent vertices can join it.
        # However, enumerating all subsets of k=30 is 2^30 ~ 1e9, which is too large.
        
        # Alternative approach:
        # Every edge is either (clique, clique) or (clique, independent).
        # (clique, clique) edges must be covered by cliques entirely within the clique part.
        # (clique, independent) edges must be covered by cliques that include that independent vertex.
        
        # Let C be the set of clique vertices (0..k-1).
        # Let I be the set of independent vertices (k..n-1).
        
        # For each independent vertex v in I, let N(v) be its neighbors in C.
        # Any clique containing v must be a subset of N(v) ∪ {v}.
        # Since I is an independent set, a clique can contain at most one vertex from I.
        
        # So the cliques are of two types:
        # Type 1: Subsets of C (cover clique-clique edges)
        # Type 2: {v} ∪ S where v in I and S is a clique in C, S ⊆ N(v) (cover clique-independent edges incident to v)
        
        # We need to cover all edges exactly once.
        # This is a set cover/partition problem.
        
        # Let's use a greedy approach first, then try to improve.
        # Actually, let's think about the structure more carefully.
        
        # Edges within C: must be covered by Type 1 cliques.
        # Edges between C and I: each edge (u,v) with u in C, v in I must be covered by a Type 2 clique containing v and u.
        
        # Since each Type 2 clique contains exactly one vertex from I, the edges incident to different I vertices are covered by disjoint sets of Type 2 cliques.
        # So we can solve for each v in I independently: cover all edges (u,v) for u in N(v) using cliques of the form {v} ∪ S where S is a clique in C, S ⊆ N(v).
        # But we also need to cover the edges within C. These can be covered by Type 1 cliques OR by Type 2 cliques (the S part).
        
        # Wait, if we use a Type 2 clique {v} ∪ S, it covers edges within S (which are clique-clique edges) AND edges from S to v.
        # So the clique-clique edges can be covered by either Type 1 or Type 2 cliques.
        
        # Let's formulate this as:
        # We have a set of potential cliques. We need to select a subset such that every edge is covered exactly once.
        # This is an exact cover problem, which is NP-hard, but the structure might help.
        
        # Let's try a simpler heuristic:
        # 1. For each v in I, the edges incident to v must be covered. Let's create cliques for v.
        # 2. The remaining edges (within C) must be covered by Type 1 cliques.
        
        # But this might not be optimal because using Type 2 cliques to cover some C-C edges might reduce the total number.
        
        # Let's try a different approach:
        # Since k is at most 30, we can enumerate all maximal cliques in the C part? No, C is a complete graph, so the only maximal clique is C itself.
        
        # Actually, in the C part (which is a complete graph), any subset is a clique.
        # So Type 1 cliques are just subsets of C.
        # Type 2 cliques are {v} ∪ S where S ⊆ N(v) and S is any subset (since C is complete, any S ⊆ C is a clique).
        
        # So the problem reduces to:
        # Partition the edges of the complete graph on C, plus the cross edges, into cliques.
        # Each cross edge (u,v) must be in a clique containing v.
        
        # Let's use a greedy algorithm:
        # 1. For each v in I, group the neighbors N(v) into cliques. Since any subset of C is a clique, we can just take S = N(v) as one clique {v} ∪ N(v). But this might not be optimal if N(v) is large.
        # Actually, to minimize the number of cliques, we want to make the S parts as large as possible, because they also cover C-C edges.
        
        # Let's try:
        # For each v in I, create one clique {v} ∪ N(v). This covers all edges incident to v, and all C-C edges within N(v).
        # Then, cover the remaining C-C edges (those not in any N(v)) with Type 1 cliques.
        
        # But this might not be optimal. Let's think...
        
        # Actually, the optimal solution likely involves:
        # - For each v in I, the clique {v} ∪ N(v) is a natural choice, but maybe we can split N(v) into multiple cliques to better cover C-C edges.
        
        # Let's try a simple approach first and see how many cliques we get.
        
        # Approach:
        # 1. For each v in I, add clique {v} ∪ N(v) if N(v) is non-empty.
        # 2. Find all C-C edges not covered by these cliques.
        # 3. Cover the remaining C-C edges with cliques. Since C is a complete graph, we can cover the remaining edges with a clique partition of the remaining subgraph.
        
        # But the remaining C-C edges form a graph on C. We need to partition this graph into cliques.
        # The minimum number of cliques to cover a graph is the chromatic number of the complement graph.
        
        # Let's implement this.
        
        # Build the graph of C-C edges
        # C = {0, ..., k-1}
        # An edge (u, w) in C is covered by a Type 2 clique for v if u, w ∈ N(v).
        
        covered_cc = set()
        cliques = []
        
        # For each v in I
        for v in range(k, n):
            N_v = [u for u in range(k) if (u, v) in set(cross_edges) or (v, u) in set(cross_edges)]
            # Since cross_edges are (u, v) with u < k <= v, we have (u, v)
            N_v = [u for u in range(k) if (u, v) in set(cross_edges)]
            if N_v:
                cliques.append([v] + N_v)
                for u, w in combinations(N_v, 2):
                    covered_cc.add((min(u, w), max(u, w)))
        
        # Find remaining C-C edges
        all_cc_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        remaining_cc = all_cc_edges - covered_cc
        
        # Cover remaining_cc with cliques
        # Build the graph of remaining_cc
        # We need to find a clique partition of this graph.
        # The minimum clique partition is the chromatic number of the complement.
        
        if remaining_cc:
            # Build adjacency for the remaining graph
            adj = defaultdict(set)
            for u, v in remaining_cc:
                adj[u].add(v)
                adj[v].add(u)
            
            # Get the vertices involved
            vertices = list(set(u for u, v in remaining_cc) | set(v for u, v in remaining_cc))
            
            # Find a clique partition using greedy coloring of the complement
            # Complement graph: edge (u, v) if (u, v) NOT in remaining_cc and u != v
            comp_adj = defaultdict(set)
            for u in vertices:
                for w in vertices:
                    if u != w and (u, w) not in remaining_cc and (w, u) not in remaining_cc:
                        comp_adj[u].add(w)
            
            # Greedy coloring
            color = {}
            for v in sorted(vertices, key=lambda x: -len(comp_adj[x])):
                used_colors = set(color.get(w, -1) for w in comp_adj[v] if w in color)
                c = 0
                while c in used_colors:
                    c += 1
                color[v] = c
            
            # Group by color
            from collections import defaultdict
            color_classes = defaultdict(list)
            for v, c in color.items():
                color_classes[c].append(v)
            
            for c, vs in color_classes.items():
                if len(vs) >= 2:
                    cliques.append(sorted(vs))
        
        # Filter cliques with at least 2 vertices
        cliques = [c for c in cliques if len(c) >= 2]
        
        # Verify and return
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split_clique_partition(n, k, p, seed)
        # Convert to lists of ints
        cliques = [[int(v) for v in clique] for clique in cliques]
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
