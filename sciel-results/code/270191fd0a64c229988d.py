import random
import time
import math

def construct():
    n = 16
    m = 65
    
    # Initialize with K_{8,8} + edge (0,1)
    # Vertices 0-7 in A, 8-15 in B
    adj = [0] * n
    degrees = [0] * n
    
    def add_edge(u, v):
        if adj[u] & (1 << v):
            return
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        degrees[u] += 1
        degrees[v] += 1

    def remove_edge(u, v):
        if not (adj[u] & (1 << v)):
            return
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        degrees[u] -= 1
        degrees[v] -= 1

    # Build K_{8,8}
    for u in range(8):
        for v in range(8, 16):
            add_edge(u, v)
            
    # Add one edge inside A
    add_edge(0, 1)
    
    # Verify edge count
    edge_count = sum(degrees) // 2
    # Should be 64 + 1 = 65
    
    def calculate_score():
        max_sum = 0
        # Iterate over all pairs u < v
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    # Find common neighbors w
                    # w must be connected to both u and v
                    common = adj[u] & adj[v]
                    # Exclude u and v themselves (though they are not neighbors of themselves)
                    # But we need to ensure w is distinct from u and v
                    # adj[u] does not have bit u set, so common does not have u or v?
                    # Wait, adj[u] has bits for neighbors. u is not a neighbor of u.
                    # So common contains w such that w~u and w~v.
                    # We just iterate over set bits in common.
                    common &= ~(1 << u) & ~(1 << v) # Just in case, though not strictly needed if self-loops are absent
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        common ^= lsb
                        s = degrees[u] + degrees[v] + degrees[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    current_score = calculate_score()
    best_score = current_score
    best_adj = [x for x in adj]
    best_degrees = [x for x in degrees]
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    T = 10.0
    cooling = 0.999
    
    # Precompute all possible pairs for faster random selection?
    # There are 120 pairs. 65 edges, 55 non-edges.
    
    while time.time() < deadline:
        # Pick a random edge to remove
        # Simple method: pick random u, v, check if edge exists
        # To be faster, maintain a list of edges? 
        # With n=16, iterating is fast enough.
        
        # Random edge (u, v)
        while True:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and (adj[u] & (1 << v)):
                break
        
        # Random non-edge (x, y)
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x != y and not (adj[x] & (1 << y)):
                break
        
        # Ensure we don't swap an edge with itself (impossible as one is edge, one is non-edge)
        # Ensure we don't create duplicate edges (handled by non-edge check)
        
        # Apply swap
        remove_edge(u, v)
        add_edge(x, y)
        
        new_score = calculate_score()
        
        delta = new_score - current_score
        
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = [a for a in adj]
                best_degrees = [d for d in degrees]
        else:
            # Revert
            remove_edge(x, y)
            add_edge(u, v)
            
        T *= cooling
        
    # Reconstruct edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
