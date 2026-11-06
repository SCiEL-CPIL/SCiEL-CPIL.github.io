import random
import time
import math
import numpy as np

def construct():
    n = 16
    target_edges = 65
    
    # Start with K_{8,8} + 1 edge
    # Bitmask representation for adjacency
    adj = [0] * n
    
    # Part A: 0-7, Part B: 8-15
    mask_B = ((1 << 8) - 1) << 8
    for i in range(8):
        adj[i] = mask_B
        for j in range(8, 16):
            adj[j] |= 1 << i
            
    # Add edge (0,1)
    adj[0] |= 1 << 1
    adj[1] |= 1 << 0
    
    # Helper to get edge list from bitmask adjacency
    def get_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask ^= lsb
        return edges
    
    # Helper to compute score (max triangle degree sum) using bitmasks
    def compute_score(adj):
        deg = [bin(a).count('1') for a in adj]
        max_sum = 0
        for u in range(n):
            mask_u = adj[u]
            for v in range(u + 1, n):
                if not (mask_u >> v) & 1:
                    continue
                # Common neighbors of u and v
                common = mask_u & adj[v]
                # Only consider w > v to count each triangle once (via smallest edge)
                # Mask out bits <= v
                mask_w = common & ~((1 << (v + 1)) - 1)
                while mask_w:
                    lsb = mask_w & -mask_w
                    w = lsb.bit_length() - 1
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
                    mask_w ^= lsb
        return max_sum

    current_score = compute_score(adj)
    best_adj = adj[:]
    best_score = current_score
    
    start_time = time.time()
    timeout = 115.0
    T = 1.0
    cooling = 0.9999
    
    # List of all possible edges (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Current edge set for quick lookup and random choice
    current_edge_set = set()
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                current_edge_set.add((u, v))
            mask ^= lsb
            
    # Non-edges
    non_edges = [e for e in all_edges if e not in current_edge_set]
    
    while time.time() - start_time < timeout:
        if not current_edge_set or not non_edges:
            break
        
        # Pick a random edge to remove
        remove_edge = random.choice(list(current_edge_set))
        u1, v1 = remove_edge
        
        # Pick a random non-edge to add
        add_edge = random.choice(non_edges)
        u2, v2 = add_edge
        
        # Apply move
        # Remove (u1, v1)
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        current_edge_set.discard((u1, v1))
        non_edges.append((u1, v1))
        
        # Add (u2, v2)
        adj[u2] |= 1 << v2
        adj[v2] |= 1 << u2
        current_edge_set.add((u2, v2))
        non_edges.remove((u2, v2))
        
        # Compute new score
        new_score = compute_score(adj)
        
        # Acceptance criteria
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
        else:
            delta = new_score - current_score
            if delta > 0:
                if delta / T > 700:
                    prob = 0.0
                else:
                    prob = math.exp(-delta / T)
                
                if random.random() >= prob:
                    # Revert
                    adj[u1] |= 1 << v1
                    adj[v1] |= 1 << u1
                    current_edge_set.add((u1, v1))
                    non_edges.remove((u1, v1))
                    
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    current_edge_set.discard((u2, v2))
                    non_edges.append((u2, v2))
            # If delta == 0, accept (do nothing)
            
        T *= cooling
    
    # Return best result
    best_edges = get_edges(best_adj)
    return {'n': n, 'edges': best_edges}
