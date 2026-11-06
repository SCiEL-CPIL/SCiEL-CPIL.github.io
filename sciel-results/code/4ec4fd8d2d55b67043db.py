import numpy as np

def construct():
    n = 16
    target_edges = 65
    
    # Helper to calculate the score (max degree sum over triangles)
    def calc_score(adj):
        degs = np.sum(adj, axis=1)
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    common = np.where(adj[u] & adj[v] > 0)[0]
                    if len(common) > 0:
                        sums = degs[u] + degs[v] + degs[common]
                        local_max = np.max(sums)
                        if local_max > max_sum:
                            max_sum = local_max
        return max_sum

    # Initialize with K_{8,8} + 1 edge
    adj = np.zeros((n, n), dtype=int)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    current_score = calc_score(adj)
    best_score = current_score
    best_adj = adj.copy()
    
    rng = np.random.default_rng(42)
    
    T = 5.0
    cooling = 0.9995
    iterations = 200000
    
    for i in range(iterations):
        # Decide move type
        use_targeted = (rng.random() < 0.5 and current_score > 0)
        
        u_rem = -1
        v_rem = -1
        
        if use_targeted:
            # Find worst triangle edge
            degs = np.sum(adj, axis=1)
            max_sum = 0
            worst_u, worst_v = -1, -1
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        common = np.where(adj[u] & adj[v] > 0)[0]
                        if len(common) > 0:
                            sums = degs[u] + degs[v] + degs[common]
                            idx = np.argmax(sums)
                            s = sums[idx]
                            if s > max_sum:
                                max_sum = s
                                worst_u, worst_v = u, v
            if worst_u != -1:
                u_rem, v_rem = worst_u, worst_v
        
        if u_rem == -1:
            # Random edge removal
            u = rng.integers(0, n)
            v = rng.integers(0, n)
            while u == v or adj[u, v] == 0:
                u = rng.integers(0, n)
                v = rng.integers(0, n)
            u_rem, v_rem = u, v
        
        # Random non-edge addition
        x = rng.integers(0, n)
        y = rng.integers(0, n)
        while x == y or adj[x, y] == 1:
            x = rng.integers(0, n)
            y = rng.integers(0, n)
            
        if (u_rem == x and v_rem == y) or (u_rem == y and v_rem == x):
            continue

        # Apply
        adj[u_rem, v_rem] = 0
        adj[v_rem, u_rem] = 0
        adj[x, y] = 1
        adj[y, x] = 1
        
        new_score = calc_score(adj)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj.copy()
        else:
            adj[u_rem, v_rem] = 1
            adj[v_rem, u_rem] = 1
            adj[x, y] = 0
            adj[y, x] = 0
            
        T *= cooling

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}
