import random
import time

def construct():
    n = 16
    # Start with the Grötzsch graph
    # Vertices 0..10 are Grötzsch, 11..15 are isolated
    # Grötzsch edges:
    # Cycle 0-1-2-3-4-0
    # Vertex 5 connected to 0,1; 6 to 1,2; 7 to 2,3; 8 to 3,4; 9 to 4,0
    # Vertex 10 connected to 5,6,7,8,9
    edges = set()
    cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in cycle:
        edges.add((u, v))
        edges.add((v, u))
    for i in range(5):
        u = i
        w = (i + 1) % 5
        edges.add((u, 5 + w))
        edges.add((5 + w, u))
        edges.add((w, 5 + u))
        edges.add((5 + u, w))
    for u in range(5, 10):
        edges.add((u, 10))
        edges.add((10, u))
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True

    def is_3_colorable(adj):
        # Backtracking with bitmasks
        color = [-1] * n
        # Order vertices by degree descending for efficiency
        order = sorted(range(n), key=lambda x: sum(adj[x]), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Available colors
            avail = 7  # 111 in binary
            for u in range(n):
                if adj[v][u] and color[u] != -1:
                    avail &= ~(1 << color[u])
            if avail == 0:
                return False
            while avail:
                c = (avail & -avail).bit_length() - 1
                avail &= avail - 1
                color[v] = c
                if backtrack(idx + 1):
                    return True
            color[v] = -1
            return False
        
        return backtrack(0)

    # Initial check
    if not is_triangle_free(adj) or is_3_colorable(adj):
        # Should not happen for Grötzsch
        return [[u, v] for u, v in edges if u < v]

    best_edges = len(edges) // 2
    best_adj = [row[:] for row in adj]

    start_time = time.time()
    deadline = 110.0  # Safety margin

    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Randomize the order of edge additions
        current_adj = [row[:] for row in best_adj]
        current_edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if current_adj[u][v]:
                    current_edges.add((u, v))
        
        # Greedy addition
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Randomize the order of non-edges
            non_edges = [(u, v) for u in range(n) for v in range(u+1, n) if not current_adj[u][v]]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > deadline:
                    break
                # Check if adding (u,v) keeps it triangle-free
                triangle = False
                for w in range(n):
                    if w != u and w != v and current_adj[u][w] and current_adj[v][w]:
                        triangle = True
                        break
                if triangle:
                    continue
                
                # Check if adding (u,v) keeps it not 3-colorable
                # Temporarily add edge
                current_adj[u][v] = True
                current_adj[v][u] = True
                
                if is_3_colorable(current_adj):
                    # Revert
                    current_adj[u][v] = False
                    current_adj[v][u] = False
                else:
                    # Accept
                    current_edges.add((u, v))
                    changed = True
        
        if len(current_edges) > best_edges:
            best_edges = len(current_edges)
            best_adj = [row[:] for row in current_adj]
            # Update best_adj from current_edges
            for u in range(n):
                for v in range(n):
                    best_adj[u][v] = False
            for u, v in current_edges:
                best_adj[u][v] = True
                best_adj[v][u] = True

    # Return the best graph found
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                result.append([u, v])
    return result
