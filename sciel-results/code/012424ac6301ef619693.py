import time
import random

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initialize with K_{8,8} + (0,1)
    adj = [[0]*n for _ in range(n)]
    edges = set()
    
    # Add K_{8,8}
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
            edges.add((u, v))
            
    # Add (0,1)
    adj[0][1] = 1
    adj[1][0] = 1
    edges.add((0, 1))
    
    # Calculate initial degrees
    deg = [sum(adj[i]) for i in range(n)]
    
    def calc_score(adj, deg):
        max_sum = 0
        # Iterate over all edges
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    # Find common neighbors
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_score = calc_score(adj, deg)
    best_score = current_score
    best_adj = [row[:] for row in adj]
    best_edges = set(edges)
    
    # Local Search
    while time.time() < deadline:
        # Pick a random edge to remove
        u1 = random.randint(0, n-1)
        v1 = random.randint(0, n-1)
        if u1 == v1 or not adj[u1][v1]:
            continue
            
        # Pick a random non-edge to add
        added = False
        for _ in range(5): # Try a few times to find a valid non-edge
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2:
                continue
            if not adj[u2][v2]:
                # Ensure we are not adding the same edge we just removed
                if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                    continue
                # Apply move
                adj[u1][v1] = 0
                adj[v1][u1] = 0
                edges.remove((min(u1,v1), max(u1,v1)))
                
                adj[u2][v2] = 1
                adj[v2][u2] = 1
                edges.add((min(u2,v2), max(u2,v2)))
                
                # Update degrees
                deg[u1] -= 1
                deg[v1] -= 1
                deg[u2] += 1
                deg[v2] += 1
                
                new_score = calc_score(adj, deg)
                
                # Acceptance Criteria
                # Accept if better, or equal (to explore), or worse with low prob?
                # Greedy + Random Walk
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = [row[:] for row in adj]
                        best_edges = set(edges)
                elif new_score == current_score:
                    # Accept with 50% probability to escape local minima plateaus
                    if random.random() < 0.5:
                        current_score = new_score
                    else:
                        # Undo
                        adj[u1][v1] = 1
                        adj[v1][u1] = 1
                        edges.add((min(u1,v1), max(u1,v1)))
                        
                        adj[u2][v2] = 0
                        adj[v2][u2] = 0
                        edges.remove((min(u2,v2), max(u2,v2)))
                        
                        deg[u1] += 1
                        deg[v1] += 1
                        deg[u2] -= 1
                        deg[v2] -= 1
                else:
                    # Undo
                    adj[u1][v1] = 1
                    adj[v1][u1] = 1
                    edges.add((min(u1,v1), max(u1,v1)))
                    
                    adj[u2][v2] = 0
                    adj[v2][u2] = 0
                    edges.remove((min(u2,v2), max(u2,v2)))
                    
                    deg[u1] += 1
                    deg[v1] += 1
                    deg[u2] -= 1
                    deg[v2] -= 1
                added = True
                break
        
    # Reconstruct edge list from best_adj
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                final_edges.append([u, v])
                
    return {'n': n, 'edges': final_edges}
