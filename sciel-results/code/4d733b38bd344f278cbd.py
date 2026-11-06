import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    def is_regular_induced(subset_mask, adj_mask, n):
        """Check if induced subgraph on vertices in subset_mask is regular.
        subset_mask: bitmask of vertices in subset
        adj_mask: list where adj_mask[i] is bitmask of neighbors of vertex i
        """
        k = bin(subset_mask).count('1')
        if k < 6:
            return False
        # Get degrees of each vertex in the subset within the induced subgraph
        degrees = []
        m = subset_mask
        while m:
            v = (m & -m).bit_length() - 1  # lowest set bit
            m &= m - 1
            deg = bin(adj_mask[v] & subset_mask).count('1')
            degrees.append(deg)
        # Check if all degrees are the same
        d0 = degrees[0]
        for d in degrees[1:]:
            if d != d0:
                return False
        return True
    
    def check_graph_valid(n, adj_mask):
        """Check if no subset of size >= 6 induces a regular subgraph."""
        all_mask = (1 << n) - 1
        # Enumerate all subsets of size >= 6
        # Use Gosper's hack or iterate all masks and filter by popcount
        for subset in range(1 << n):
            k = bin(subset).count('1')
            if k < 6:
                continue
            if k > n:
                break
            if is_regular_induced(subset, adj_mask, n):
                return False
        return True
    
    def build_adj_mask(edges, n):
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask
    
    # Try from largest n down
    for n in range(22, 4, -1):
        if time.time() > deadline - 5.0:
            break
        max_attempts = 5000 if n <= 15 else 2000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5.0:
                break
            # Generate random graph
            edges = []
            adj_mask = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        edges.append([i, j])
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            
            # Check validity
            if check_graph_valid(n, adj_mask):
                return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph
    return {'n': 5, 'edges': []}
