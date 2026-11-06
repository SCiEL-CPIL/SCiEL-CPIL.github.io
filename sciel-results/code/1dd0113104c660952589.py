import random
import time

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = None
    best_score = float('inf')
    
    # Helper to count C4s in a given edge list
    def count_c4(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Count common neighbors of i and j
                common = adj[i] & adj[j]
                c4_count += bin(common).count('1')
        return c4_count // 2 # Each C4 is counted twice (once for each pair of opposite vertices)
        
    # Helper to build a C4-free graph greedily
    def build_c4_free(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        adj = [0] * n
        edges = []
        remaining_edges = all_edges[:]
        random.shuffle(remaining_edges)
        
        while len(edges) < 90: # We only need up to 90 for the C4-free base
            # Find an edge with 0 common neighbors
            best_edge = None
            for u, v in remaining_edges:
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    best_edge = (u, v)
                    break
            if best_edge is None:
                break
            
            u, v = best_edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append(best_edge)
            remaining_edges.remove(best_edge)
            
        return edges, adj

    # Run multiple random restarts to find a good C4-free graph
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        edges_0, adj_0 = build_c4_free(i)
        
        if len(edges_0) < 90:
            # If we didn't reach 90, try to add more edges even if they create C4s
            # to get to 91 total, minimizing C4s
            # But first, let's see if we can get a better C4-free graph
            pass
        
        # We have a base graph edges_0 (size <= 90)
        # We need to add edges to reach 91
        # If len(edges_0) == 90, we add 1 edge.
        # If len(edges_0) < 90, we add (91 - len(edges_0)) edges.
        
        current_edges = edges_0[:]
        current_adj = adj_0[:]
        
        # Generate candidate edges not in current_edges
        existing_set = set(current_edges)
        candidates = [e for e in all_edges if e not in existing_set]
        
        # Add edges one by one, choosing the one with min new C4s
        while len(current_edges) < target_edges:
            if time.time() > deadline:
                break
                
            best_cand = None
            min_common = float('inf')
            
            for u, v in candidates:
                # Check if already added
                if current_adj[u] & (1 << v):
                    continue
                common = bin(current_adj[u] & current_adj[v]).count('1')
                if common < min_common:
                    min_common = common
                    best_cand = (u, v)
            
            if best_cand is None:
                break
                
            u, v = best_cand
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            current_edges.append(best_cand)
            candidates.remove(best_cand)
            
        # Evaluate this solution
        if len(current_edges) == target_edges:
            score = count_c4(current_edges)
            if score < best_score:
                best_score = score
                best_edges = current_edges[:]
                
    # If we didn't find a valid solution (should not happen), fallback
    if best_edges is None:
        # Fallback to lexicographic
        best_edges = all_edges[:target_edges]
        
    return best_edges
