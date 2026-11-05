import itertools

def construct():
    # Construct Andrásfai graph And(9) on 27 vertices
    n = 27
    edges = []
    # Edge if (j-i) mod 27 is in {1, 4, 7, 10, 13, 16, 19, 22, 25}
    # These are residues congruent to 1 mod 3
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    
    # Helper to check triangle-free
    def is_triangle_free(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True

    # Helper to calculate independence number (Branch and Bound)
    def alpha(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        best = 0
        
        def dfs(candidates, current_size):
            nonlocal best
            if not candidates:
                if current_size > best:
                    best = current_size
                return
            
            # Pruning
            if current_size + len(candidates) <= best:
                return
            
            v = candidates[0]
            rest = candidates[1:]
            
            # Include v
            new_cands = [u for u in rest if not adj[v][u]]
            dfs(new_cands, current_size + 1)
            
            # Exclude v
            dfs(rest, current_size)

        dfs(list(range(n)), 0)
        return best

    current_n = n
    current_edges = edges
    
    # Deterministic expansion
    # We try to add vertices one by one.
    # To keep it fast, we limit the depth of the alpha check or use a heuristic.
    # Since n is small, exact alpha is okay for a few steps.
    
    for _ in range(15):
        # Find a max independent set in current graph
        adj = [[False]*current_n for _ in range(current_n)]
        for u, v in current_edges:
            adj[u][v] = True
            adj[v][u] = True
            
        max_set = []
        def find_max_is(candidates, current_set):
            nonlocal max_set
            if not candidates:
                if len(current_set) > len(max_set):
                    max_set = current_set.copy()
                return
            if len(current_set) + len(candidates) <= len(max_set):
                return
            v = candidates[0]
            rest = candidates[1:]
            new_cands = [u for u in rest if not adj[v][u]]
            find_max_is(new_cands, current_set + [v])
            find_max_is(rest, current_set)

        find_max_is(list(range(current_n)), [])
        I = max_set
        
        new_n = current_n + 1
        new_vertex = current_n
        new_edges = current_edges.copy()
        for v in I:
            new_edges.append([new_vertex, v])
        
        # Check validity
        if is_triangle_free(new_n, new_edges):
            a = alpha(new_n, new_edges)
            if a < 10:
                current_n = new_n
                current_edges = new_edges
            else:
                break
        else:
            break
            
    return {'n': current_n, 'edges': current_edges}
