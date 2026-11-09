import time
import random
import numpy as np

def _has_mono_k222(adj, n):
    """Check if adjacency matrix contains a monochromatic K2,2,2."""
    for c_color in (0, 1):
        for a in range(n):
            for b in range(a+1, n):
                if adj[a][b] != c_color:
                    continue
                S = []
                for v in range(n):
                    if v != a and v != b and adj[a][v] == c_color and adj[b][v] == c_color:
                        S.append(v)
                if len(S) < 4:
                    continue
                m = len(S)
                edges = []
                for i in range(m):
                    for j in range(i+1, m):
                        if adj[S[i]][S[j]] == c_color:
                            edges.append((i, j))
                for i in range(len(edges)):
                    a1, b1 = edges[i]
                    for j in range(i+1, len(edges)):
                        a2, b2 = edges[j]
                        if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                            return True
    return False

def _paley_colors(n):
    """Generate Paley coloring for prime n."""
    squares = set()
    for x in range(1, n):
        squares.add((x*x) % n)
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            diff = (j - i) % n
            colors.append(0 if diff in squares else 1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    random.seed(42)
    
    n_target = 30
    n29 = 29
    num_edges = n_target * (n_target - 1) // 2
    
    edge_to_ij = [(i, j) for i in range(n_target) for j in range(i+1, n_target)]
    
    squares29 = set()
    for x in range(1, n29):
        squares29.add((x*x) % n29)
    
    best_colors = _paley_colors(n29)
    
    max_restarts = 30
    max_steps = 600
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        adj = [[0]*n_target for _ in range(n_target)]
        for i in range(n29):
            for j in range(i+1, n29):
                diff = (j - i) % n29
                c = 0 if diff in squares29 else 1
                adj[i][j] = c
                adj[j][i] = c
        for i in range(n29):
            c = random.randint(0, 1)
            adj[29][i] = c
            adj[i][29] = c
        
        if not _has_mono_k222(adj, n_target):
            colors = [adj[i][j] for i in range(n_target) for j in range(i+1, n_target)]
            return {'n': n_target, 'colors': colors}
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not _has_mono_k222(adj, n_target):
                colors = [adj[i][j] for i in range(n_target) for j in range(i+1, n_target)]
                return {'n': n_target, 'colors': colors}
            
            if step < max_steps // 3:
                i = random.randint(0, n29-1)
                j = 29
            else:
                eidx = random.randint(0, num_edges - 1)
                i, j = edge_to_ij[eidx]
            
            adj[i][j] = 1 - adj[i][j]
            adj[j][i] = adj[i][j]
    
    return {'n': 29, 'colors': best_colors}
