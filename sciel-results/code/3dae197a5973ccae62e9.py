"""C4-free graph with high minimum degree (R(C4,K1,39) lower bound)."""
import time
import random

# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 135.0
    
    def try_build(n, seed):
        rng = random.Random(seed)
        adj = [0] * n
        deg = [0] * n
        # common[i][j] = number of common neighbors of i and j
        common = [[0] * n for _ in range(n)]
        
        # List of all potential edges
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        rng.shuffle(edges)
        
        for u, v in edges:
            if common[u][v] < 2:
                # Add edge (u, v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                
                # Update common neighbors
                # For each neighbor x of u (excluding v), common[x][v] increases by 1
                # For each neighbor y of v (excluding u), common[u][y] increases by 1
                
                nu = adj[u] & ~(1 << v) # Neighbors of u excluding v
                nv = adj[v] & ~(1 << u) # Neighbors of v excluding u
                
                # Iterate over bits in nu
                x_mask = nu
                while x_mask:
                    lsb = x_mask & (-x_mask)
                    x = lsb.bit_length() - 1
                    common[x][v] += 1
                    common[v][x] += 1
                    x_mask ^= lsb
                    
                y_mask = nv
                while y_mask:
                    lsb = y_mask & (-y_mask)
                    y = lsb.bit_length() - 1
                    common[u][y] += 1
                    common[y][u] += 1
                    y_mask ^= lsb
                    
        md = min(deg)
        if md >= n - 39:
            edge_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edge_list.append([i, j])
            return {'n': n, 'edges': edge_list}
        return None

    best = None
    for n in range(47, 38, -1):
        found = False
        for seed in range(200):
            if time.time() > deadline:
                break
            res = try_build(n, seed)
            if res is not None:
                # Verify
                n_val = res['n']
                edges = res['edges']
                deg = [0] * n_val
                adj = [0] * n_val
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # Check min degree
                if min(deg) < n_val - 39:
                    continue
                    
                # Check C4
                c4_found = False
                for u in range(n_val):
                    for v in range(u + 1, n_val):
                        common = adj[u] & adj[v]
                        if common & (common - 1):
                            c4_found = True
                            break
                    if c4_found:
                        break
                            
                if not c4_found:
                    if best is None or n_val > best['n']:
                        best = res
                    found = True
                    break
        if found:
            break
            
    if best is not None:
        return best
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
