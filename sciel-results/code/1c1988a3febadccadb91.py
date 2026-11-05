import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 20
    max_edges = n * (n - 1) // 2
    
    # Precompute all pairs
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_add(adj, u, v):
        # adj is list of int masks
        # Check common neighbors of u, v
        c = adj[u] & adj[v]
        if c.bit_count() > 2:
            return False
        
        # Check neighbors of v
        mask_v = adj[v]
        mask_u = adj[u]
        
        # For all z in adj[v], CN(u, z) must be <= 1
        z = 0
        while mask_v:
            if mask_v & 1:
                # z is neighbor of v
                if (adj[u] & adj[z]).bit_count() > 1:
                    return False
            mask_v >>= 1
            z += 1
            
        # For all z in adj[u], CN(v, z) must be <= 1
        z = 0
        while mask_u:
            if mask_u & 1:
                # z is neighbor of u
                if (adj[v] & adj[z]).bit_count() > 1:
                    return False
            mask_u >>= 1
            z += 1
            
        return True

    def get_edges_from_adj(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            for v in range(u + 1, n):
                if mask & (1 << v):
                    edges.append([u, v])
        return edges

    best_edge_count = 0
    best_adj = [0] * n

    while time.time() < deadline:
        # Initialize with random valid graph
        adj = [0] * n
        # Greedy random initialization
        random.shuffle(pairs)
        for u, v in pairs:
            if is_valid_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        current_count = sum(a.bit_count() for a in adj) // 2
        
        # Local Search / Simulated Annealing
        T = 1.0
        no_improve = 0
        
        # We can iterate many times
        for step in range(10000):
            if time.time() > deadline:
                break
                
            # Try to add a random non-edge
            u, v = random.choice(pairs)
            if not (adj[u] & (1 << v)):
                if is_valid_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_count += 1
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # If edge exists, maybe try to remove it?
                # Or just count as no improve
                no_improve += 1
            
            # If stuck for a while, remove a random edge
            if no_improve > 50:
                # Find an existing edge to remove
                # Pick random u, then random neighbor v
                u = random.randint(0, n - 1)
                mask = adj[u]
                if mask:
                    # Pick random set bit
                    z = 0
                    temp_mask = mask
                    while temp_mask:
                        if temp_mask & 1:
                            if random.random() < 0.5: # Probability of removing
                                v = z
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                current_count -= 1
                                break
                        temp_mask >>= 1
                        z += 1
                no_improve = 0
                T *= 0.9 # Cool down? Or just reset counter
            
            if current_count > best_edge_count:
                best_edge_count = current_count
                best_adj = adj[:]
        
        # If we found a better graph, we might want to restart from it?
        # Or just continue with new random starts.
        
    return get_edges_from_adj(best_adj)
