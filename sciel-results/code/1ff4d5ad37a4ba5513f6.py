import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def solve(n):
        if n < 2:
            return [0] * (n*(n-1)//2)
        
        # adj[c][i] is a set of neighbors of i in color c
        adj = [[set() for _ in range(n)] for _ in range(4)]
        colors = [0] * (n*(n-1)//2)
        
        # Edge list in lexicographic order
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        def has_c6(u, v, c):
            # Check if there is a simple path of length 5 from u to v in G_c
            visited = {u: True}
            def dfs(node, depth):
                if depth == 5:
                    return node == v
                for nb in adj[c][node]:
                    if nb not in visited:
                        visited[nb] = True
                        if dfs(nb, depth + 1):
                            return True
                        del visited[nb]
                return False
            return dfs(u, 0)

        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == len(edges):
                return colors[:]
            
            u, v = edges[idx]
            # Heuristic: Try colors that are already used by neighbors? 
            # Or just 0,1,2,3.
            for c in range(4):
                if has_c6(u, v, c):
                    continue
                
                colors[idx] = c
                adj[c][u].add(v)
                adj[c][v].add(u)
                
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                
                adj[c][u].remove(v)
                adj[c][v].remove(u)
                colors[idx] = 0
                
            return None
            
        if n >= 5:
            # Symmetry breaking
            # Fix (0,1)=0, (0,2)=1, (0,3)=2, (0,4)=3
            for i, c in enumerate([0, 1, 2, 3]):
                u, v = edges[i]
                colors[i] = c
                adj[c][u].add(v)
                adj[c][v].add(u)
            return backtrack(4)
        else:
            return backtrack(0)

    best_colors = None
    best_n = 9
    
    # Try to find a solution for n=17, then 16, etc.
    for n in range(17, 8, -1):
        if time.time() > deadline - 5:
            break
        colors = solve(n)
        if colors is not None:
            best_n = n
            best_colors = colors
            break
            
    if best_colors is None:
        n = 9
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        best_n = n
        
    return {'n': best_n, 'colors': best_colors}
