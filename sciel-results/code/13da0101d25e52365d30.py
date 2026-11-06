import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    def popcount(x):
        return bin(x).count('1')

    def get_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    def verify_full(adj, n):
        """
        Verify that no subset of size >= 6 induces a regular subgraph.
        """
        vertices = list(range(n))
        for k in range(6, n + 1):
            for comb in itertools.combinations(vertices, k):
                mask = 0
                for u in comb:
                    mask |= (1 << u)
                
                first_deg = -1
                is_regular = True
                for u in comb:
                    deg = popcount(adj[u] & mask)
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_regular = False
                        break
                
                if is_regular:
                    return False
        return True

    def find_valid_graph(n, time_budget):
        """
        Find a valid graph on n vertices using constraint propagation style local search.
        """
        deadline = start_time + time_budget
        vertices = list(range(n))
        # Precompute all 6-combinations for speed
        combs_6 = list(itertools.combinations(vertices, 6))
        
        # Precompute masks for each 6-combination
        masks_6 = []
        for comb in combs_6:
            mask = 0
            for u in comb:
                mask |= (1 << u)
            masks_6.append(mask)
        
        # Precompute edges within each 6-combination for flipping logic
        # Actually, we can just iterate over pairs in the comb when needed
        
        best_adj = None
        best_violation_count = float('inf')
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
                
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Initial check
            if verify_full(adj, n):
                return adj
            
            # Local Search
            for step in range(2000):
                if time.time() > deadline:
                    break
                
                # Find all regular 6-subsets
                regular_subsets = []
                for idx, comb in enumerate(combs_6):
                    mask = masks_6[idx]
                    first_deg = -1
                    is_regular = True
                    for u in comb:
                        deg = popcount(adj[u] & mask)
                        if first_deg == -1:
                            first_deg = deg
                        elif deg != first_deg:
                            is_regular = False
                            break
                    if is_regular:
                        regular_subsets.append(comb)
                
                if not regular_subsets:
                    # No 6-regular subsets. Check larger ones.
                    if verify_full(adj, n):
                        return adj
                    # If there are larger regular subsets but no 6-regular, 
                    # we are stuck in a local minimum for 6-subsets.
                    # Randomly flip an edge to escape?
                    # Or just restart?
                    # Let's try flipping a random edge
                    u, v = random.sample(vertices, 2)
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    continue
                
                # Count edge frequencies in regular subsets
                edge_counts = {}
                for comb in regular_subsets:
                    for u, v in itertools.combinations(comb, 2):
                        key = (min(u, v), max(u, v))
                        edge_counts[key] = edge_counts.get(key, 0) + 1
                
                # Pick the edge with the highest count
                if edge_counts:
                    # To avoid getting stuck in a cycle, add some randomness
                    # Sort by count, pick from top 10%
                    sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
                    top_k = max(1, len(sorted_edges) // 10)
                    chosen_edge = random.choice(sorted_edges[:top_k])[0]
                    u, v = chosen_edge
                    
                    # Flip the edge
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    # Should not happen if regular_subsets is not empty
                    break
            
            # Track best state (though we return immediately if valid)
            # We don't really need best_adj if we restart, but good for debugging
            pass

        return None

    # Try to find a valid graph on 21 vertices first
    # Give it most of the time
    adj21 = find_valid_graph(21, 100.0)
    if adj21 is not None:
        return {'n': 21, 'edges': get_edges_from_adj(adj21, 21)}
    
    # Try to find a valid graph on 20 vertices
    adj20 = find_valid_graph(20, 40.0)
    if adj20 is not None:
        return {'n': 20, 'edges': get_edges_from_adj(adj20, 20)}
    
    # Fallback: Return a small valid graph
    n = 5
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': n, 'edges': get_edges_from_adj(adj, n)}
