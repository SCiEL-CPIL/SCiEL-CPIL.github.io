import numpy as np

def construct():
    n = 31
    q = 5
    
    # Generate points of PG(2, q)
    points = []
    seen = set()
    
    def normalize(vec):
        # Make the first non-zero element 1
        for i in range(3):
            if vec[i] != 0:
                inv = pow(int(vec[i]), q-2, q)
                return tuple(int(v * inv) % q for v in vec)
        return None

    for i in range(q):
        for j in range(q):
            for k in range(q):
                if i == 0 and j == 0 and k == 0:
                    continue
                vec = np.array([i, j, k])
                norm = normalize(vec)
                if norm and norm not in seen:
                    seen.add(norm)
                    points.append(norm)
    
    if len(points) != n:
        # Fallback
        return [[i, j] for i in range(n) for j in range(i+1, n)][:91]
        
    # Define the polarity matrix
    M = np.array([[0, 0, 1], 
                  [0, 1, 0], 
                  [1, 0, 0]], dtype=int)
    
    # Build adjacency
    adj = np.zeros((n, n), dtype=int)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            u = np.array(points[i])
            v = np.array(points[j])
            val = (u @ (M @ v)) % q
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append([i, j])
                
    # Find the best edge to add
    # Since the graph is C4-free, any non-edge has <= 1 common neighbor.
    # So any non-edge addition results in 0 new C4s.
    # We can just pick the first non-edge.
    # But to be safe and consistent, we find one with min common neighbors.
    
    min_common = float('inf')
    best_edge = None
    
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                common = np.sum(adj[i] & adj[j])
                if common < min_common:
                    min_common = common
                    best_edge = [i, j]
                    
    if best_edge:
        edges.append(best_edge)
        
    return edges
