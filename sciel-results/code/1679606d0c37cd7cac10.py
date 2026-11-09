"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from collections import defaultdict
    
    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Identify K and I
        K = set(range(k))
        I = set(range(k, n))
        
        # Build adjacency for I to K
        neighbors_in_K = {v: set() for v in I}
        for (u, v) in edges:
            if u in K and v in I:
                neighbors_in_K[v].add(u)
            elif v in K and u in I:
                neighbors_in_K[u].add(v)
        
        # Initial assignment: S_v = N(v)
        S = {v: set(neighbors_in_K[v]) for v in I}
        
        # Track which K-edges are covered by which I-cliques
        # edge_covered_by[(u, w)] = set of v such that u, w in S_v
        edge_covered_by = defaultdict(set)
        
        def update_coverage(v, added, removed):
            # Remove v from coverage of edges involving removed nodes
            # This is complex to do incrementally correctly without full rebuild for small n
            pass
            
        # For simplicity and correctness, we rebuild coverage map in each iteration of conflict resolution
        # n is small (<=90), so this is fast enough.
        
        orphans = [] # list of (v, u) pairs that need size-2 cliques
        
        # Iterative conflict resolution
        # We want to ensure no K-edge is covered by >1 S_v
        changed = True
        while changed:
            changed = False
            # Rebuild edge_covered_by
            edge_covered_by = defaultdict(set)
            for v in I:
                S_v = S[v]
                if len(S_v) < 2:
                    continue
                # Iterate over pairs in S_v
                S_list = list(S_v)
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        u, w = S_list[i], S_list[j]
                        if u > w:
                            u, w = w, u
                        edge_covered_by[(u, w)].add(v)
            
            # Find conflicts
            conflicts = []
            for e, vs in edge_covered_by.items():
                if len(vs) > 1:
                    conflicts.append((e, list(vs)))
            
            if not conflicts:
                break
            
            # Resolve one conflict
            # Pick the first conflict
            e, vs = conflicts[0]
            u, w = e
            # Pick one v to break
            v_to_break = vs[0]
            
            # Remove w from S[v_to_break]
            # We need to remove one of the nodes from the pair to break the edge (u,w) in S_v
            # Removing w is arbitrary
            S[v_to_break].remove(w)
            
            # The edge (v_to_break, w) is now orphaned (if it was covered by S_v_to_break)
            # It was covered because w was in S. Now it's not.
            # We must cover (v_to_break, w) with a new clique.
            orphans.append((v_to_break, w))
            
            changed = True
            
        # Construct final cliques
        cliques = []
        
        # 1. I-cliques
        for v in I:
            S_v = S[v]
            if len(S_v) >= 1:
                clique = [v] + list(S_v)
                cliques.append(clique)
        
        # 2. Orphan cliques
        for (v, u) in orphans:
            cliques.append([v, u])
        
        # 3. K-only cliques for remaining edges
        # Determine which K-edges are already covered by I-cliques
        covered_K_edges = set()
        for v in I:
            S_v = S[v]
            if len(S_v) < 2:
                continue
            S_list = list(S_v)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u, w = S_list[i], S_list[j]
                    if u > w:
                        u, w = w, u
                    covered_K_edges.add((u, w))
        
        # Get all K-edges
        all_K_edges = set()
        for u in range(k):
            for w in range(u+1, k):
                all_K_edges.add((u, w))
        
        remaining_K_edges = all_K_edges - covered_K_edges
        
        # Partition remaining_K_edges into cliques
        # Greedy Maximal Clique
        remaining = set(remaining_K_edges)
        while remaining:
            # Find a maximal clique in the subgraph induced by remaining edges
            # Since it's a subgraph of a complete graph, any set of vertices forming a clique in the original
            # is a clique here. We just need to find a set of edges that form a clique.
            # A set of edges forms a clique if they share vertices in a specific way?
            # No, a clique in the graph of edges? No.
            # The cliques must be sets of vertices.
            # So we need to find a subset of vertices V_sub such that all edges in V_sub are in 'remaining'.
            # And we want to maximize the number of edges covered? Or just find any?
            # To minimize count, we want large cliques.
            
            # Simple heuristic: Pick an edge, extend to maximal clique.
            edge = next(iter(remaining))
            u, w = edge
            # Try to extend {u, w} to a larger clique
            # Candidates are vertices connected to both u and w in the remaining graph?
            # Actually, in the remaining graph, we only have edges in 'remaining'.
            # So we need a set of vertices {u, w, ...} such that all pairs are in 'remaining'.
            
            current_clique_vertices = {u, w}
            # Find all vertices x such that (u,x) and (w,x) are in remaining
            potential = set()
            for x in range(k):
                if x == u or x == w:
                    continue
                e1 = (min(u,x), max(u,x))
                e2 = (min(w,x), max(w,x))
                if e1 in remaining and e2 in remaining:
                    potential.add(x)
            
            # Greedily add vertices from potential
            # To keep it simple and fast, we just add all potential vertices that form a clique with each other?
            # That's complex. Let's just add one by one or all if they form a clique.
            # For small k, we can just check if potential + {u,w} is a clique.
            # If not, we can just use {u,w}.
            
            # Let's try to find a maximal clique in the "line graph" sense?
            # No, simpler:
            # Just use the edge {u, w} as a clique of size 2 if extension is hard.
            # But we can do better.
            
            # Let's try to add vertices from potential one by one.
            # A vertex x can be added if it is connected to ALL current clique vertices in 'remaining'.
            new_clique_vertices = set(current_clique_vertices)
            added_any = True
            while added_any:
                added_any = False
                for x in list(potential):
                    if x in new_clique_vertices:
                        continue
                    # Check if x is connected to all in new_clique_vertices
                    is_connected = True
                    for y in new_clique_vertices:
                        e = (min(x,y), max(x,y))
                        if e not in remaining:
                            is_connected = False
                            break
                    if is_connected:
                        new_clique_vertices.add(x)
                        added_any = True
                        # Remove x from potential to avoid re-checking?
                        # potential = potential - {x}
            
            # Add this clique
            clique_verts = list(new_clique_vertices)
            cliques.append(clique_verts)
            
            # Remove all edges within this clique from 'remaining'
            for i in range(len(clique_verts)):
                for j in range(i+1, len(clique_verts)):
                    u, w = clique_verts[i], clique_verts[j]
                    if u > w:
                        u, w = w, u
                    remaining.discard((u, w))
        
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        res = solve_instance(*spec)
        # Ensure types are int and list
        clean_res = []
        for c in res:
            clean_res.append([int(v) for v in c])
        results.append(clean_res)
        
    return results
# EVOLVE-BLOCK-END
