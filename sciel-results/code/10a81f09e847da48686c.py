import time

def construct():
    start_time = time.time()
    
    for n in range(28, 25, -1):
        if time.time() - start_time > 110:
            break
        
        color = [[-1]*n for _ in range(n)]
        adj = [ [set() for _ in range(5)] for _ in range(n) ]
        
        def is_valid_color(u, v, c):
            # u < v
            # Check if adding (u,v) with color c creates C4.
            # C4 exists if >= 2 common neighbors in color c.
            # Common neighbors w must satisfy:
            # 1. w != u, w != v
            # 2. (u, w) is c
            # 3. (v, w) is c
            # Since we are building v, (v, w) is only known for w < u (already assigned in this step) 
            # or w < v in previous steps?
            # Wait. 
            # When we are at vertex v, we assign (0, v), (1, v), ..., (v-1, v).
            # So for a given u, (w, v) is assigned for all w < u.
            # For w >= u, (w, v) is NOT assigned yet.
            # So any common neighbor w MUST be < u.
            # So we only check w in adj[v][c] (which contains w < u).
            
            count = 0
            # adj[v][c] contains w such that (w, v) is c. All these w are < u.
            for w in adj[v][c]:
                if w in adj[u][c]:
                    count += 1
                    if count >= 2:
                        return False
            return True

        def solve(v):
            if time.time() - start_time > 105:
                return False
            if v == n:
                return True
            return solve_edge(v, 0)

        def solve_edge(v, u):
            if time.time() - start_time > 105:
                return False
            if u == v:
                return solve(v + 1)
            
            # Heuristic: Try colors that are less used?
            # Or just 0..4.
            for c in range(5):
                if is_valid_color(u, v, c):
                    color[u][v] = c
                    color[v][u] = c
                    adj[u][c].add(v)
                    adj[v][c].add(u)
                    
                    if solve_edge(v, u + 1):
                        return True
                    
                    color[u][v] = -1
                    color[v][u] = -1
                    adj[u][c].remove(v)
                    adj[v][c].remove(u)
            return False

        if solve(0):
            res_colors = []
            for i in range(n):
                for j in range(i+1, n):
                    res_colors.append(color[i][j])
            return {'n': n, 'colors': res_colors}
        
        continue
        
    return {'n': 1, 'colors': []}
