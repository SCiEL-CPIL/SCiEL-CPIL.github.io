import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety
    
    best_edges = []
    best_count = 0
    
    def try_greedy(seed):
        random.seed(seed)
        M = [0] * n
        neighbors = [[] for _ in range(n)]
        edge_list = []
        
        # Generate all edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check time
            if time.time() > deadline:
                break
            
            # Skip if already connected
            if (M[u] >> v) & 1:
                continue
                
            # Check C6
            # Compute S_v3: set of vertices at distance 3 from v
            S_v3 = 0
            for d in neighbors[v]:
                for c in neighbors[d]:
                    S_v3 |= M[c]
            
            # Check if any neighbor a of u has a neighbor b in S_v3
            conflict = False
            for a in neighbors[u]:
                if M[a] & S_v3:
                    conflict = True
                    break
            
            if not conflict:
                # Add edge
                M[u] |= (1 << v)
                M[v] |= (1 << u)
                neighbors[u].append(v)
                neighbors[v].append(u)
                edge_list.append([u, v])
        
        return edge_list

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        seed = restarts
        edges = try_greedy(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        restarts += 1
        
    return best_edges
