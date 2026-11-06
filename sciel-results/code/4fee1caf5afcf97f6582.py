"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    base_edges = [
        [0,1],[1,2],[2,3],[3,4],[4,0],
        [1,5],[4,5],[2,6],[0,6],[3,7],[1,7],[4,8],[2,8],[0,9],[3,9],
        [5,10],[6,10],[7,10],[8,10],[9,10]
    ]
    
    def build_adj(edges, n):
        adj = [[False]*n for _ in range(n)]
        for u,v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def is_triangle_free(adj, n):
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True
    
    def is_3_colorable(adj, n):
        color = [-1]*n
        def backtrack(vertex):
            if vertex == n:
                return True
            for c in range(3):
                ok = True
                for w in range(n):
                    if adj[vertex][w] and color[w] == c:
                        ok = False
                        break
                if ok:
                    color[vertex] = c
                    if backtrack(vertex+1):
                        return True
                    color[vertex] = -1
            return False
        return backtrack(0)
    
    best_edges = base_edges[:]
    best_count = len(base_edges)
    adj = build_adj(base_edges, n)
    
    for vertex in range(11, 16):
        if time.time() > deadline - 10:
            break
        
        valid_neighbors = []
        for v in range(vertex):
            if not adj[vertex][v]:
                creates_tri = False
                for w in range(n):
                    if w != vertex and w != v and adj[vertex][w] and adj[v][w]:
                        creates_tri = True
                        break
                if not creates_tri:
                    valid_neighbors.append(v)
        
        # Try to add as many edges as possible while keeping non-3-colorable
        # Start with all, remove if needed
        test_adj = [row[:] for row in adj]
        for v in valid_neighbors:
            test_adj[vertex][v] = True
            test_adj[v][vertex] = True
        
        if is_triangle_free(test_adj, n) and not is_3_colorable(test_adj, n):
            adj = test_adj
            for v in valid_neighbors:
                best_edges.append([vertex, v])
        else:
            # Find maximum subset
            best_subset = []
            # Try removing 1 edge
            found = False
            for skip in range(len(valid_neighbors)):
                if time.time() > deadline - 10:
                    break
                t_adj = [row[:] for row in adj]
                for v in valid_neighbors:
                    if v != valid_neighbors[skip]:
                        t_adj[vertex][v] = True
                        t_adj[v][vertex] = True
                if is_triangle_free(t_adj, n) and not is_3_colorable(t_adj, n):
                    adj = t_adj
                    for v in valid_neighbors:
                        if v != valid_neighbors[skip]:
                            best_edges.append([vertex, v])
                    found = True
                    break
            if not found:
                # Remove 2 edges
                for i in range(len(valid_neighbors)):
                    for j in range(i+1, len(valid_neighbors)):
                        if time.time() > deadline - 10:
                            break
                        t_adj = [row[:] for row in adj]
                        for v in valid_neighbors:
                            if v != valid_neighbors[i] and v != valid_neighbors[j]:
                                t_adj[vertex][v] = True
                                t_adj[v][vertex] = True
                        if is_triangle_free(t_adj, n) and not is_3_colorable(t_adj, n):
                            adj = t_adj
                            for v in valid_neighbors:
                                if v != valid_neighbors[i] and v != valid_neighbors[j]:
                                    best_edges.append([vertex, v])
                            found = True
                            break
                    if found:
                        break
            if not found:
                # Add no edges for this vertex
                pass
    
    best_count = len(best_edges)
    return best_edges
# EVOLVE-BLOCK-END
