import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # Clique on first k
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Identify C and I
        C = set(range(k))
        I = set(range(k, n))
        
        # Build cross adjacency: w -> set of neighbors in C
        cross_adj = {w: set() for w in I}
        internal_edges = set()
        
        for u, v in edges:
            if u in C and v in C:
                internal_edges.add((min(u, v), max(u, v)))
            elif u in C and v in I:
                cross_adj[v].add(u)
            elif u in I and v in C:
                cross_adj[u].add(v)
        
        # Determine coverable internal edges
        # e = (u, v) is coverable if exists w in I such that u, v in cross_adj[w]
        coverable_edges = set()
        # Map edge to list of w's that cover it
        edge_to_w = {}
        
        for w in I:
            neighbors = cross_adj[w]
            if len(neighbors) < 2:
                continue
            # Generate all pairs in neighbors
            neighbors_list = list(neighbors)
            for i in range(len(neighbors_list)):
                for j in range(i + 1, len(neighbors_list)):
                    u = neighbors_list[i]
                    v = neighbors_list[j]
                    e = (min(u, v), max(u, v))
                    if e not in coverable_edges:
                        coverable_edges.add(e)
                        edge_to_w[e] = []
                    edge_to_w[e].append(w)
        
        # Find maximum matching in G' = (C, coverable_edges)
        # Greedy approach: sort edges, pick disjoint ones
        # To maximize, we can try to pick edges that are "critical" first?
        # Or just random shuffle and greedy.
        # Given density, simple greedy is likely optimal.
        
        edges_to_match = list(coverable_edges)
        # Sort by degree in G'? Or just random.
        # Let's sort by the number of w's covering them (least constrained first?)
        # Or most constrained?
        # Actually, any maximum matching works.
        # Let's use a simple greedy with a few random restarts to ensure we find a large matching.
        
        best_matching = []
        rng = Random(seed + 1000)
        
        for _ in range(10): # 10 restarts
            rng.shuffle(edges_to_match)
            matching = []
            used_vertices = set()
            for e in edges_to_match:
                u, v = e
                if u not in used_vertices and v not in used_vertices:
                    matching.append(e)
                    used_vertices.add(u)
                    used_vertices.add(v)
            if len(matching) > len(best_matching):
                best_matching = matching
                if len(best_matching) == k // 2: # Max possible
                    break
        
        # Assign w to each edge in best_matching
        cliques = []
        covered_internal = set()
        covered_cross = set()
        
        for e in best_matching:
            u, v = e
            # Pick a w
            w_list = edge_to_w.get(e, [])
            if w_list:
                # Pick the first one
                w = w_list[0]
                cliques.append([u, v, w])
                covered_internal.add(e)
                covered_cross.add((min(u, w), max(u, w)))
                covered_cross.add((min(v, w), max(v, w)))
        
        # Cover remaining internal edges
        for e in internal_edges:
            if e not in covered_internal:
                u, v = e
                cliques.append([u, v])
                covered_internal.add(e)
        
        # Cover remaining cross edges
        for w in I:
            for u in cross_adj[w]:
                e_cross = (min(u, w), max(u, w))
                if e_cross not in covered_cross:
                    cliques.append([u, w])
                    covered_cross.add(e_cross)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
    
    # Ensure types
    final_results = []
    for cliques in results:
        clean_cliques = []
        for c in cliques:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                clean_cliques.append([int(x) for x in c])
        final_results.append(clean_cliques)
    
    return final_results
