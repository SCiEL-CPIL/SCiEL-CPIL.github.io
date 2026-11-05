import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute for fast operations
    # We will use a deterministic construction based on difference sets / Paley-like structures
    # then refine with simulated annealing that uses a more effective neighborhood: 
    # removing edges that are part of many K2,3 violations and adding edges that reduce penalty.
    
    # However, the current program already does SA. 
    # Let's try a different approach: 
    # Start with a highly structured graph (e.g., a union of cliques or a specific design) 
    # that is known to have high edge density and low common neighbors, then optimize.
    
    # Alternative strategy: Use a "penalty-based" local search where we explicitly track 
    # the number of K2,3 subgraphs and try to eliminate them while adding edges.
    
    # But let's look at the inspirations. Inspiration 3 mentions Paley graphs and cyclic difference sets.
    # Let's try constructing a graph based on a (20, 9, 4) difference set if it exists, or similar.
    # Actually, ex(20, K2,3) is related to bipartite Ramsey numbers.
    
    # Let's try a completely different heuristic: 
    # 1. Start with a random graph.
    # 2. Use a "repair" phase that specifically removes edges involved in K2,3s.
    # 3. Use a "fill" phase that adds edges that don't create K2,3s.
    # 4. Alternate these.
    
    # This is essentially what the current program does with SA + greedy fill.
    
    # Let's try: 
    # 1. Initialize with a graph that is a disjoint union of K5s (4 * K5). 
    #    Edges: 4 * 10 = 40. 
    #    Common neighbors: within a K5, any two vertices have 3 common neighbors. 
    #    So this is NOT K2,3-free. 
    #    We need graphs where no two vertices have 3 common neighbors.
    
    # A (10, 3, 1) design? No.
    # Let's use a bipartite graph approach? No, we want max edges, so likely non-bipartite.
    
    # Let's try a specific known structure: 
    # The complement of a linear hypergraph?
    
    # Let's stick to the SA but change the initial state to be a "greedy maximal" graph 
    # built from a specific ordering, and use a more aggressive cooling schedule.
    
    # Actually, let's try a "memetic" algorithm: 
    # Multiple SA runs, but also try swapping vertices (relabeling) to break symmetry?
    # No, relabeling doesn't change the graph structure, just the labels.
    
    # Let's try: 
    # 1. Generate several random graphs.
    # 2. For each, run a fast local search (hill climbing) that only accepts moves that 
    #    either increase edges (if valid) or decrease penalty (if invalid).
    # 3. Keep the best.
    
    # The current program does this with SA. 
    # Let's try a different neighborhood: 
    # Instead of flipping a single edge, try "2-flips" (swap edges) or "rotate" moves.
    
    # Or, simpler: 
    # Use a deterministic greedy algorithm with a clever ordering.
    # Order vertices by degree?
    
    # Let's try: 
    # 1. Start with empty graph.
    # 2. Add edges in a specific order (e.g., random permutation of all pairs).
    # 3. When considering adding edge (u,v), check if it creates a K2,3.
    #    If not, add it.
    # 4. Repeat with different random orders.
    
    # This is a simple greedy heuristic. It might be faster than SA and allow more restarts.
    
    def greedy_build(order):
        adj = [[0]*n for _ in range(n)]
        cn = [[0]*n for _ in range(n)]
        
        for u, v in order:
            # Check if adding (u,v) is valid
            # We need to ensure that for all w, if v~w then cn[u][w] < 2, and if u~w then cn[v][w] < 2.
            valid = True
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[v][w] and cn[u][w] >= 2:
                    valid = False
                    break
                if adj[u][w] and cn[v][w] >= 2:
                    valid = False
                    break
            
            if valid:
                adj[u][v] = 1
                adj[v][u] = 1
                # Update cn
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[v][w]:
                        cn[u][w] += 1
                        cn[w][u] += 1
                    if adj[u][w]:
                        cn[v][w] += 1
                        cn[w][v] += 1
        
        return adj, cn

    best_edges = []
    best_m = 0
    
    # Run many random greedy builds
    num_restarts = 200  # Can afford many if fast
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        order = all_pairs[:]
        random.shuffle(order)
        
        adj, cn = greedy_build(order)
        
        m = sum(adj[u][v] for u in range(n) for v in range(u+1, n))
        
        if m > best_m:
            best_m = m
            best_edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u][v]]
            
            # Optional: do a local search polish on this best graph
            if time.time() < deadline - 5:
                # Polish: try to add any missing edge
                improved = True
                while improved and time.time() < deadline - 2:
                    improved = False
                    for u, v in all_pairs:
                        if adj[u][v]:
                            continue
                        # Check if can add
                        valid = True
                        for w in range(n):
                            if w == u or w == v:
                                continue
                            if adj[v][w] and cn[u][w] >= 2:
                                valid = False
                                break
                            if adj[u][w] and cn[v][w] >= 2:
                                valid = False
                                break
                        if valid:
                            adj[u][v] = 1
                            adj[v][u] = 1
                            for w in range(n):
                                if w == u or w == v:
                                    continue
                                if adj[v][w]:
                                    cn[u][w] += 1
                                    cn[w][u] += 1
                                if adj[u][w]:
                                    cn[v][w] += 1
                                    cn[w][v] += 1
                            improved = True
                            m += 1
                            best_edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u][v]]
                            best_m = m
                            
        # Also, try to improve the current best with SA? 
        # Maybe just stick to greedy + polish for now to see if it's competitive.
        
    return best_edges
