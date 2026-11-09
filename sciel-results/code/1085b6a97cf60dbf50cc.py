import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100 seconds limit to be safe
    
    # Try to find a valid graph for n from 22 down to 1
    for n in range(22, 0, -1):
        # Try a few random graphs for this n
        # The number of tries can be adaptive based on time
        max_tries = 50 if n <= 10 else 20
        
        for attempt in range(max_tries):
            if time.time() > deadline:
                break
            
            # Generate a random graph
            # Use p=0.5
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        edges.append([u, v])
            
            # Verify
            if is_valid_graph(n, edges):
                return {'n': n, 'edges': edges}
        
        # If we exhausted tries for this n, move to smaller n
        # But we should probably try harder for larger n if time permits
        # For now, just move on.

    # Fallback: return a small valid graph
    # n=1 is always valid
    return {'n': 1, 'edges': []}

def is_valid_graph(n, edges):
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Iterate through all subsets
    # We can iterate from 0 to 2^n - 1
    # But we only care about subsets with size >= 6
    # We can use Gosper's hack to iterate subsets of size k, or just filter
    # Filtering is simpler:
    for mask in range(1, 1 << n):
        # Quick check: if popcount < 6, skip
        # bit_count is fast in Python 3.10+
        if mask.bit_count() < 6:
            continue
        
        # Check if induced subgraph on 'mask' is regular
        # Get the list of vertices in the mask
        # We can iterate through bits
        # To be efficient, we can check degrees
        
        # Find the first vertex in mask to get the target degree
        # But we need to check all vertices have the same degree
        
        # Let's extract vertices
        # This is expensive if done naively for every mask.
        # Optimization: 
        # If we find a bad subgraph, we return False immediately.
        # Most random graphs will have independent sets or cliques of size 6, which are bad.
        # So we expect early exit.
        
        # To speed up, we can check for independent sets and cliques first?
        # An independent set has degree 0 for all.
        # A clique has degree k-1 for all.
        
        # Let's just do the general check.
        # Get vertices
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        # Check degrees
        k = len(verts)
        if k < 6:
            continue
            
        target_deg = -1
        is_reg = True
        for v in verts:
            # Degree of v in mask
            deg = (adj[v] & mask).bit_count()
            if target_deg == -1:
                target_deg = deg
            elif deg != target_deg:
                is_reg = False
                break
        
        if is_reg:
            return False
            
    return True
