import time
import random

def is_3_colorable(adj):
    n = 16
    colors = [-1] * n
    
    # Order vertices by degree (highest first) for better pruning
    vertices = list(range(n))
    vertices.sort(key=lambda x: bin(adj[x]).count('1'), reverse=True)
    
    def backtrack(idx):
        if idx == len(vertices):
            return True
        
        v = vertices[idx]
        
        # Determine available colors
        used_colors = 0
        # Iterate over neighbors of v
        # adj[v] is a bitmask
        mask = adj[v]
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            u = lsb.bit_length() - 1
            if colors[u] != -1:
                used_colors |= (1 << colors[u])
            mask ^= lsb
        
        for c in range(3):
            if not (used_colors & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        
        return False
    
    return backtrack(0)

def construct():
    deadline = time.time() + 140.0
    best_edge_count = 0
    best_adj = None
    
    def create_grotzsch():
        adj = [0] * 16
        # Cycle 0-4
        adj[0] |= (1 << 1) | (1 << 4)
        adj[1] |= (1 << 0) | (1 << 2)
        adj[2] |= (1 << 1) | (1 << 3)
        adj[3] |= (1 << 2) | (1 << 4)
        adj[4] |= (1 << 3) | (1 << 0)
        
        # Mycielski step
        adj[5] |= (1 << 1) | (1 << 4)
        adj[1] |= (1 << 5)
        adj[4] |= (1 << 5)
        
        adj[6] |= (1 << 0) | (1 << 2)
        adj[0] |= (1 << 6)
        adj[2] |= (1 << 6)
        
        adj[7] |= (1 << 1) | (1 << 3)
        adj[1] |= (1 << 7)
        adj[3] |= (1 << 7)
        
        adj[8] |= (1 << 2) | (1 << 4)
        adj[2] |= (1 << 8)
        adj[4] |= (1 << 8)
        
        adj[9] |= (1 << 0) | (1 << 3)
        adj[0] |= (1 << 9)
        adj[3] |= (1 << 9)
        
        adj[10] |= (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)
        adj[5] |= (1 << 10)
        adj[6] |= (1 << 10)
        adj[7] |= (1 << 10)
        adj[8] |= (1 << 10)
        adj[9] |= (1 << 10)
        
        return adj

    while time.time() < deadline:
        adj = create_grotzsch()
        
        possible_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
        random.shuffle(possible_edges)
        
        current_edge_count = 20
        
        for u, v in possible_edges:
            if time.time() > deadline:
                break
            
            if adj[u] & (1 << v):
                continue
            
            if adj[u] & adj[v]:
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not is_3_colorable(adj):
                current_edge_count += 1
            else:
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
        
        if current_edge_count > best_edge_count:
            best_edge_count = current_edge_count
            best_adj = adj[:]

    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
