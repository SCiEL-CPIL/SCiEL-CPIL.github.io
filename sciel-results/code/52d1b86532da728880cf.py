import numpy as np
import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_edges = []
    best_adj = np.zeros((n, n), dtype=int)
    best_edge_count = 0
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to check if adding (u, v) creates a C6
    def creates_c6(u, v, adj):
        # Check if there is a path of length 5 from u to v
        # BFS from u, stop at depth 5
        visited = [[False] * n for _ in range(6)] # visited[depth][node]
        queue = [(u, 0)]
        visited[0][u] = True
        
        while queue:
            node, depth = queue.pop(0)
            if depth == 5:
                continue
            for neighbor in range(n):
                if adj[node, neighbor] == 1:
                    if not visited[depth + 1][neighbor]:
                        visited[depth + 1][neighbor] = True
                        if neighbor == v and depth + 1 == 5:
                            return True
                        queue.append((neighbor, depth + 1))
        return False

    # Optimization: Use numpy for adjacency, but BFS in Python might be slow.
    # Let's try to optimize the check.
    # A path of length 5 exists if (A^5)[u, v] > 0.
    # Computing A^5 every time is $O(n^3 \log 5)$ or similar, which is fine for n=40.
    # But we can just do BFS.
    
    # Let's use a faster BFS with lists
    def has_path_len_5(u, v, adj):
        # adj is a numpy array or list of lists
        # We want to know if there is a path of length exactly 5 from u to v.
        # Note: A path of length 5 implies a C6 when adding (u,v).
        # It also implies shorter paths might exist, but we specifically need length 5.
        # Actually, if there is a path of length 5, adding (u,v) creates a cycle of length 6.
        # If there is a path of length 3, adding (u,v) creates a C4.
        # The task says "no six-cycle". It does NOT say no C4.
        # So we ONLY need to check for C6.
        # A C6 is formed if there is a path of length 5 between u and v.
        
        # BFS to find shortest path? No, we need *any* path of length 5.
        # If the shortest path is 1, 2, 3, or 4, it doesn't prevent a length 5 path from existing.
        # Example: u-a-v (len 2) and u-b-c-d-e-v (len 5).
        # So we must check for existence of a walk/path of length 5.
        # Since it's a simple graph, a "path" usually means simple path.
        # A C6 is a simple cycle.
        # So we need a simple path of length 5 from u to v.
        
        # This is harder to check than just reachability.
        # However, for sparse graphs, BFS visiting states (node, depth) might work if we don't prune too early.
        # But we can't prune if we've visited a node at a shallower depth, because we might need to take a longer route.
        # So we need to track visited states (node, depth).
        # Max states: 40 * 6 = 240. This is very small.
        
        visited = set()
        queue = [(u, 0)]
        visited.add((u, 0))
        
        while queue:
            node, depth = queue.pop(0)
            if depth == 5:
                if node == v:
                    return True
                continue
            for neighbor in range(n):
                if adj[node, neighbor] == 1:
                    state = (neighbor, depth + 1)
                    if state not in visited:
                        visited.add(state)
                        queue.append(state)
        return False

    # Run greedy iterations
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = np.zeros((n, n), dtype=int)
        current_edges = []
        
        for u, v in all_edges:
            if creates_c6(u, v, adj):
                continue
            # Add edge
            adj[u, v] = 1
            adj[v, u] = 1
            current_edges.append([u, v])
            
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_edges = current_edges
            best_adj = adj.copy()
            
        # If we reached 105, we can stop early? 
        # The problem says "best known... 105". 
        # If we find 105, it's likely optimal or close.
        if best_edge_count >= 105:
            break
            
    return best_edges
