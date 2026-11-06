import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = 110.0
    
    best_edges = 0
    best_adj = None
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def has_k33(adj):
        # Returns (False, None) if valid, (True, (u, v)) if violation found
        # We need to find a triple U with >= 3 common neighbors outside U
        for u1 in range(n):
            for u2 in range(u1+1, n):
                for u3 in range(u2+1, n):
                    # Common neighbors of u1, u2, u3
                    # Intersection of adj[u1], adj[u2], adj[u3]
                    # Exclude u1, u2, u3 themselves
                    common = []
                    for v in range(n):
                        if v == u1 or v == u2 or v == u3:
                            continue
                        if adj[u1][v] and adj[u2][v] and adj[u3][v]:
                            common.append(v)
                    if len(common) >= 3:
                        # We have a K3,3
                        # The edges involved are (u1, common[0]), etc.
                        # We can return one such edge to remove
                        # Let's return (u1, common[0])
                        return True, (u1, common[0])
        return False, None

    def is_valid(adj):
        valid, _ = has_k33(adj)
        return not valid

    # Local search
    def improve(adj):
        # Try to add edges
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle missing edges
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]:
                        missing.append((i, j))
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline:
                    break
                # Try adding
                adj[u][v] = 1
                adj[v][u] = 1
                if is_valid(adj):
                    changed = True
                else:
                    adj[u][v] = 0
                    adj[v][u] = 0
        return adj

    # Main loop
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Repair
        while time.time() < deadline:
            valid, edge = has_k33(adj)
            if not valid:
                break
            # Remove edge
            u, v = edge
            adj[u][v] = 0
            adj[v][u] = 0
            
        # Improve
        adj = improve(adj)
        
        # Count edges
        count = sum(sum(row) for row in adj) // 2
        if count > best_edges:
            best_edges = count
            best_adj = [row[:] for row in adj]
            
    if best_adj is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return get_edges(best_adj)
