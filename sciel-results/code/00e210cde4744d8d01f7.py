import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # All internal edges of K
        all_k_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_k_edges.add((u, v))
        
        # Available internal edges
        available_edges = set(all_k_edges)
        
        cliques = []
        
        # Sort vertices in I by degree (size of N(v)) descending
        vertices_i = list(range(k, n))
        vertices_i.sort(key=lambda v: len(neighbors[v]), reverse=True)
        
        # Precompute edge sets for potential U_v? 
        # We will compute on the fly or check availability.
        
        for v in vertices_i:
            N_v = neighbors[v]
            if not N_v:
                continue
                
            # Find max subset U of N_v such that all edges in U are in available_edges
            # This is Max Clique in graph induced by N_v on available_edges
            
            # Simple greedy for Max Clique:
            # 1. Start with all nodes in N_v
            # 2. Iteratively pick a node, keep only its neighbors that are connected to all previous picks?
            # Actually, we just need a clique in the graph G_avail where edges are available_edges.
            
            # Let's use a simple heuristic: 
            # Sort N_v by "availability" (how many edges to other N_v nodes are available)
            # Or just try to build a clique.
            
            # Given k is small (<=30), we can use a simple backtracking or greedy.
            # Greedy:
            # Start with empty clique C.
            # Candidates = N_v
            # While Candidates:
            #   Pick node u in Candidates with max degree in G_avail[Candidates]
            #   Add u to C
            #   Candidates = Candidates ∩ N_Gavail(u)
            
            candidates = set(N_v)
            clique_U = set()
            
            while candidates:
                # Find node with max degree within candidates
                best_u = None
                max_deg = -1
                for u in candidates:
                    deg = 0
                    for w in candidates:
                        if u == w: continue
                        if (min(u, w), max(u, w)) in available_edges:
                            deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        best_u = u
                        if deg == len(candidates) - 1: # Complete subgraph found?
                            break
                
                if best_u is None:
                    break
                
                clique_U.add(best_u)
                
                # Update candidates: must be connected to best_u in G_avail
                new_candidates = set()
                for w in candidates:
                    if w == best_u: continue
                    if (min(best_u, w), max(best_u, w)) in available_edges:
                        new_candidates.add(w)
                candidates = new_candidates
                
            # If we found a clique U of size >= 2, use it
            if len(clique_U) >= 2:
                # Check if it's worth it? Gain is |U|-1. Cost is 1 clique.
                # Always worth it if |U| >= 2.
                
                # Add clique
                cliques.append([v] + list(clique_U))
                
                # Remove covered edges from available
                for u, w in combinations(clique_U, 2):
                    e = (min(u, w), max(u, w))
                    if e in available_edges:
                        available_edges.remove(e)
            
            # For remaining neighbors of v (not in clique_U), add 2-cliques
            for u in N_v:
                if u not in clique_U:
                    cliques.append([u, v])
        
        # Cover remaining available edges with cliques
        # Greedy: pick max clique in remaining graph, remove, repeat
        while available_edges:
            # Find max clique in graph defined by available_edges
            # Nodes are 0..k-1
            # Use similar greedy as above
            candidates = set(range(k))
            # Actually, we should only consider nodes incident to available edges
            incident_nodes = set()
            for u, w in available_edges:
                incident_nodes.add(u)
                incident_nodes.add(w)
            candidates = incident_nodes
            
            clique_K = set()
            while candidates:
                best_u = None
                max_deg = -1
                for u in candidates:
                    deg = 0
                    for w in candidates:
                        if u == w: continue
                        if (min(u, w), max(u, w)) in available_edges:
                            deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        best_u = u
                        if deg == len(candidates) - 1:
                            break
                if best_u is None:
                    break
                clique_K.add(best_u)
                new_candidates = set()
                for w in candidates:
                    if w == best_u: continue
                    if (min(best_u, w), max(best_u, w)) in available_edges:
                        new_candidates.add(w)
                candidates = new_candidates
            
            if len(clique_K) >= 2:
                cliques.append(list(clique_K))
                for u, w in combinations(clique_K, 2):
                    e = (min(u, w), max(u, w))
                    if e in available_edges:
                        available_edges.remove(e)
            else:
                # If greedy failed to find clique of size >= 2, pick any edge
                # This shouldn't happen if available_edges is not empty
                u, w = next(iter(available_edges))
                cliques.append([u, w])
                available_edges.remove((u, w))
                
        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            # Fallback to simple solution if timeout
            def simple_solve(n, k, p, seed):
                neighbors = instance(n, k, p, seed)
                cliques = [list(range(k))]
                for v in range(k, n):
                    for u in neighbors[v]:
                        cliques.append([u, v])
                return [[int(x) for x in c] for c in cliques]
            results.append(simple_solve(*spec))
        else:
            cliques = solve_instance(*spec)
            results.append(cliques)
            
    return results
