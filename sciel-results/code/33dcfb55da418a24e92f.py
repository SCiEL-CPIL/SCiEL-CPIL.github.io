import numpy as np

def construct():
    n = 256
    # Build conflict graph: edge between a,b if (a*b) % (a+b) == 0
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Compute eigenvector centrality via power iteration
    vec = np.ones(n, dtype=np.float64)
    vec /= np.linalg.norm(vec)
    for _ in range(300):
        new_vec = adj.astype(np.float64) @ vec
        norm = np.linalg.norm(new_vec)
        if norm < 1e-12:
            break
        new_vec /= norm
        if np.allclose(new_vec, vec, atol=1e-10):
            break
        vec = new_vec
    
    # Sort all indices by eigenvector centrality (highest first)
    order = list(range(n))
    order.sort(key=lambda i: -vec[i])
    
    # Greedy construction: add vertices in order of decreasing eigenvector centrality
    # if they don't conflict with already selected vertices
    selected = set()
    for i in order:
        can_add = True
        for j in selected:
            if adj[i, j] == 1:
                can_add = False
                break
        if can_add:
            selected.add(i)
    
    # Convert to list of values
    result = sorted([i + 1 for i in selected])
    return result
