import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # 1. Precompute forbidden differences (non-zero squares)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # 2. Build adjacency as bitmasks
    # adj[u] is a bitmask where bit v is 1 if (u,v) is an edge
    adj = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u != v:
                d = (u - v) % m
                if d in forbidden:
                    mask |= (1 << v)
        adj[u] = mask
        
    # 3. Maximum Independent Set using Branch and Bound with Coloring
    
    # Greedy coloring of the subgraph induced by P
    # Returns the number of colors used
    def greedy_color(P):
        colors = 0
        U = P
        while U:
            colors += 1
            # Find a maximal independent set in U
            MIS = 0
            # We pick vertices greedily. 
            # Order doesn't matter much for the bound validity, but affects the tightness.
            # Picking the vertex with the smallest index is fast.
            while U:
                # Get lowest set bit
                v_bit = U & (-U)
                v = v_bit.bit_length() - 1
                MIS |= v_bit
                # Remove v and its neighbors from U
                # Note: adj[v] contains neighbors in the full graph.
                # We only care about neighbors in U.
                U &= ~(adj[v] | v_bit)
        return colors

    best_size = 0
    best_set_mask = 0
    
    # Recursive search
    # P: bitmask of available vertices
    # C: current size of independent set
    def search(P, C, current_set):
        nonlocal best_size, best_set_mask
        
        if not P:
            if C > best_size:
                best_size = C
                best_set_mask = current_set
            return
        
        # Bound 1: Cardinality
        if C + P.bit_count() <= best_size:
            return
            
        # Bound 2: Coloring
        # Calculate upper bound on MIS size in G[P]
        # This is the expensive part.
        colors = greedy_color(P)
        if C + colors <= best_size:
            return
            
        # Branching
        # Choose a vertex to branch on.
        # Heuristic: Choose a vertex with high degree in the remaining graph P?
        # For simplicity and speed, let's pick the lowest set bit.
        # A better heuristic is to pick a vertex that maximizes the reduction of P in the "include" branch.
        # Let's stick to lowest bit for now, as N is small.
        
        v_bit = P & (-P)
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        # New available vertices: P minus v and its neighbors
        P_include = P & ~(adj[v] | v_bit)
        search(P_include, C + 1, current_set | v_bit)
        
        # Branch 2: Exclude v
        P_exclude = P & ~v_bit
        search(P_exclude, C, current_set)
        
    # Initial P is all vertices
    initial_P = (1 << m) - 1
    start_time = time.time()
    
    # We might need to be careful with recursion depth, but 205 is fine.
    # However, Python recursion limit is usually 1000, so 205 is safe.
    
    # To improve performance, we can order vertices.
    # But the bitmask approach handles the order implicitly by the branching choice.
    
    # Let's run the search.
    # If it takes too long, we might need to add a time check, but B&B with coloring is usually very fast for N=205.
    
    search(initial_P, 0, 0)
    
    # Extract result
    result = []
    for i in range(m):
        if (best_set_mask >> i) & 1:
            result.append(i)
            
    return np.array(result, dtype=np.int64)
