"""Clique partition of split graphs via structural decomposition + greedy set cover."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges, time_budget):
        """
        Split graph: C = {0..k-1} (clique), I = {k..n-1} (independent set).
        Edges: C-C (all present), C-I (probabilistic), I-I (none).
        
        Strategy:
        1. For each i in I, create clique {i} ∪ N(i) where N(i) = C-neighbors of i.
           This covers all C-I edges incident to i, plus C-C edges among N(i).
        2. The remaining uncovered edges are C-C edges not covered in step 1.
           Cover these with minimum number of cliques (subsets of C).
           This is a set cover problem on the C-vertex graph with remaining edges.
        """
        # Build adjacency
        edge_set = set()
        for e in edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        # Step 1: Cover all C-I edges
        # For each independent set vertex i (k..n-1), form clique {i} ∪ N_C(i)
        # This covers all edges (c, i) for c in N_C(i)
        # It also covers C-C edges among N_C(i)
        
        # Track which C-C edges are covered
        cc_edges = set()
        for (u, v) in edge_set:
            if u < k and v < k:
                cc_edges.add((u, v))
        
        covered_cc = set()
        
        for i in range(k, n):
            neighbors_in_c = sorted([c for c in range(k) if c in adj[i]])
            if not neighbors_in_c:
                continue
            # Clique: {i} ∪ neighbors_in_c
            clique = [i] + neighbors_in_c
            if len(clique) >= 2:
                cliques.append(clique)
            # Mark C-C edges among neighbors_in_c as covered
            nc_set = set(neighbors_in_c)
            for a in range(len(neighbors_in_c)):
                for b in range(a + 1, len(neighbors_in_c)):
                    u, v = neighbors_in_c[a], neighbors_in_c[b]
                    covered_cc.add((min(u, v), max(u, v)))
        
        # Step 2: Cover remaining C-C edges
        remaining_cc = cc_edges - covered_cc
        
        if remaining_cc:
            # Build adjacency on C vertices for remaining edges
            rem_adj = [set() for _ in range(k)]
            for (u, v) in remaining_cc:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Greedy set cover: repeatedly pick a clique (subset of C) that covers
            # the most remaining edges, add it, remove those edges.
            # Candidate cliques: for each vertex, its neighborhood + itself forms
            # a maximal clique candidate. But we can also try various subsets.
            
            # Better approach: since all C vertices form a clique, any subset is a clique.
            # The problem reduces to: partition the edge set of a graph on k vertices
            # into minimum number of complete subgraphs (cliques).
            # This is the "clique partition number" or "clique edge cover number".
            
            # Greedy: repeatedly find a large clique in the remaining edge graph.
            rem_edges = set(remaining_cc)
            
            # Use randomized greedy with restarts
            best_solution = None
            best_count = float('inf')
            
            rng2 = Random(42)
            
            # Time budget for set cover
            sc_deadline = time.time() + min(time_budget, 30.0)
            
            num_restarts = 50
            for restart in range(num_restarts):
                if time.time() > sc_deadline:
                    break
                
                cur_edges = set(rem_edges)
                cur_cliques = []
                
                while cur_edges:
                    # Find a large clique in the remaining edge graph
                    # Use greedy: pick vertex with max degree, extend
                    rem_deg = [0] * k
                    rem_adj_local = [set() for _ in range(k)]
                    for (u, v) in cur_edges:
                        rem_deg[u] += 1
                        rem_deg[v] += 1
                        rem_adj_local[u].add(v)
                        rem_adj_local[v].add(u)
                    
                    # Pick starting vertex (randomize among top-degree for diversity)
                    max_deg = max(rem_deg)
                    top_vertices = [v for v in range(k) if rem_deg[v] == max_deg]
                    start_v = rng2.choice(top_vertices)
                    
                    # Greedily build clique from start_v
                    current_clique = [start_v]
                    available = set(rem_adj_local[start_v])
                    
                    # Sort candidates by degree for greedy
                    candidates = sorted(available, key=lambda x: -rem_deg[x])
                    
                    for cand in candidates:
                        if cand not in available:
                            continue
                        can_add = all(cand in rem_adj_local[vc] for vc in current_clique)
                        if can_add:
                            current_clique.append(cand)
                            available = {x for x in available if x in rem_adj_local[cand]}
                    
                    if len(current_clique) < 2:
                        u, v = next(iter(cur_edges))
                        current_clique = [u, v]
                    
                    cur_cliques.append(current_clique)
                    
                    # Remove covered edges
                    cs = set(current_clique)
                    cur_edges = {(u, v) for (u, v) in cur_edges if not (u in cs and v in cs)}
                
                if len(cur_cliques) < best_count:
                    best_count = len(cur_cliques)
                    best_solution = [list(c) for c in cur_cliques]
            
            if best_solution:
                cliques.extend(best_solution)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        elapsed = time.time() - start_time
        remaining_time = deadline - elapsed - 10.0  # leave 10s buffer
        
        if remaining_time < 5:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges, remaining_time)
        
        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
