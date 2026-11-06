import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin before 120s limit
    
    n = 40
    
    # Bitset adjacency for fast operations
    adj = [0] * n
    
    def would_create_c6(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        """
        Nu = adj[u]
        Nv = adj[v]
        
        # Iterate w in Nu
        w_mask = Nu
        while w_mask:
            w_bit = w_mask & -w_mask
            w = w_bit.bit_length() - 1
            w_mask ^= w_bit
            
            # Iterate x in Nv
            x_mask = Nv
            while x_mask:
                x_bit = x_mask & -x_mask
                x = x_bit.bit_length() - 1
                x_mask ^= x_bit
                
                if w == x:
                    continue
                
                # Check path w-z-y-x
                # z in N(w), y in N(z) \cap N(x)
                # We need y != w, y != x, z != w, z != x, z != y
                z_mask = adj[w]
                while z_mask:
                    z_bit = z_mask & -z_bit
                    z = z_bit.bit_length() - 1
                    z_mask ^= z_bit
                    
                    if z == x:
                        continue
                    
                    # Check if N(z) \cap N(x) has a valid y
                    common_y = adj[z] & adj[x]
                    # Exclude w from common_y
                    # If common_y has any bit other than w, we have a C6
                    if (common_y & ~(1 << w)) != 0:
                        return True
        
        return False

    # Initial graph: 8 disjoint K5s
    def init_k5s():
        adj = [0] * n
        for k in range(8):
            base = k * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    u = base + i
                    v = base + j
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    best_edges = []
    best_count = 0
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        adj = init_k5s()
        
        # Collect current edges
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        
        # Candidate edges: all non-edges
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    candidates.append((u, v))
        
        # Shuffle candidates
        random.shuffle(candidates)
        
        # Greedy addition
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            # Check if still a non-edge
            if not ((adj[u] >> v) & 1):
                # Check if adding this edge would create a C6
                if not would_create_c6(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        # Optional: Local search / Edge swapping if time permits
        # For now, just restarts
        
    return best_edges
