import time
import random
from itertools import combinations

def construct():
    """Construct a dense triangle-free graph on 16 vertices that is not 3-colorable."""
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    n = 16
    best_edges = []
    best_count = 0
    
    # 1. Define the Grötzsch graph (4-chromatic, triangle-free, 11 vertices)
    # Vertices 0-4 form a 5-cycle
    # Vertices 5-9 are connected to the cycle in a specific Mycielski pattern
    # Vertex 10 is connected to 5-9
    
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (4, 8), (4, 9), (0, 9),
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)
    ]
    
    # Initialize adjacency for the base graph
    base_adj = [set() for _ in range(n)]
    for u, v in groetzsch_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
        
    # The current best is the Grötzsch graph plus isolated vertices 11-15
    # Edges: 20
    best_edges = [list(e) for e in groetzsch_edges]
    best_count = len(groetzsch_edges)
    
    # 2. Local Search to add more edges
    # We want to add edges (u, v) such that N(u) and N(v) are disjoint.
    # Since we start with a 4-chromatic graph, any supergraph is 4-chromatic.
    # So we only need to check triangle-free property.
    
    def try_add_edges(adj, edges, deadline):
        # Create a list of all possible edges not currently present
        current_edge_set = set(edges)
        candidate_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in current_edge_set:
                    candidate_edges.append((u, v))
        
        # We will try to add edges greedily.
        # To avoid local maxima, we shuffle the candidates.
        # We might do multiple passes or restarts.
        
        max_iter = 50 # Number of restarts/tries
        local_best_edges = edges[:]
        local_best_count = len(edges)
        
        for restart in range(max_iter):
            if time.time() > deadline:
                break
            
            # Copy state
            adj_copy = [set(s) for s in adj]
            edges_copy = local_best_edges[:] # Start from current best? Or from base?
            # Starting from base every time might be safer to explore different regions,
            # but starting from current best allows incremental improvement.
            # Let's start from the current global best to improve it.
            
            # Actually, a pure greedy from base might get stuck.
            # Let's try a greedy from base, then try to improve the result.
            # Or just run greedy from base multiple times with different random orders.
            
            adj_init = [set() for _ in range(n)]
            for u, v in groetzsch_edges:
                adj_init[u].add(v)
                adj_init[v].add(u)
            
            cand = candidate_edges[:]
            random.shuffle(cand)
            
            count = len(groetzsch_edges)
            for u, v in cand:
                if time.time() > deadline:
                    break
                # Check if adding (u, v) creates a triangle
                # Triangle exists if there is a w such that w in N(u) and w in N(v)
                if not (adj_init[u] & adj_init[v]):
                    # Safe to add
                    adj_init[u].add(v)
                    adj_init[v].add(u)
                    edges_copy.append((u, v))
                    count += 1
            
            if count > local_best_count:
                local_best_count = count
                local_best_edges = edges_copy[:]
        
        return local_best_edges, local_best_count

    # Run the local search
    # We can run this multiple times if time permits, but the inner loop does restarts.
    # Let's just run the function which does 50 restarts internally.
    
    # To allow for more exploration, we can run this block multiple times if time remains.
    while time.time() < deadline - 10:
        # The try_add_edges function does 50 restarts.
        # Let's increase restarts in the function or call it multiple times.
        # Let's modify try_add_edges to accept a restart count.
        pass
    
    # Let's implement a more flexible search directly here.
    
    all_possible_edges = list(combinations(range(n), 2))
    base_edge_set = set(groetzsch_edges)
    
    for restart in range(200): # Up to 200 restarts
        if time.time() > deadline:
            break
            
        adj = [set() for _ in range(n)]
        for u, v in groetzsch_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        current_edges = list(groetzsch_edges)
        
        # Candidate edges: all edges not in base
        candidates = [e for e in all_possible_edges if e not in base_edge_set]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            if not (adj[u] & adj[v]):
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append((u, v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
            
    return best_edges
