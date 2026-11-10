import time
import random
import math
from collections import deque

def construct(k, g):
    deadline = time.time() + 105
    n = 320
    
    def gen_k_regular(n, k):
        if (n * k) % 2:
            return None
        stubs = [v for v in range(n) for _ in range(k)]
        for _ in range(200):
            random.shuffle(stubs)
            adj = [[False]*n for _ in range(n)]
            ok = True
            for i in range(0, n*k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v or adj[u][v]:
                    ok = False
                    break
                adj[u][v] = adj[v][u] = True
            if ok:
                return adj
        return None
    
    def count_short_cycles(adj, n, max_len):
        """Count cycles of length < max_len."""
        count = 0
        for s in range(n):
            dist = [-1]*n
            par = [-1]*n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] + 1 >= max_len:
                    continue
                for v in range(n):
                    if not adj[u][v]:
                        continue
                    if v == par[u]:
                        continue
                    if v == s:
                        if dist[u] + 1 >= 3 and dist[u] + 1 < max_len:
                            count += 1
                    elif dist[v] == -1:
                        dist[v] = dist[u] + 1
                        par[v] = u
                        q.append(v)
                    else:
                        if dist[u] + dist[v] + 1 < max_len:
                            count += 1
        return count
    
    def girth(adj, n):
        best_g = 10**9
        for s in range(n):
            dist = [-1]*n
            par = [-1]*n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] + 1 >= best_g:
                    continue
                for v in range(n):
                    if not adj[u][v]:
                        continue
                    if v == par[u]:
                        continue
                    if v == s:
                        if dist[u] + 1 >= 3:
                            best_g = min(best_g, dist[u] + 1)
                    elif dist[v] == -1:
                        dist[v] = dist[u] + 1
                        par[v] = u
                        q.append(v)
                    else:
                        best_g = min(best_g, dist[u] + dist[v] + 1)
        return best_g
    
    adj = gen_k_regular(n, k)
    if adj is None:
        return (0, [])
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.99
    
    for step in range(5000):
        if time.time() > deadline - 5:
            break
        # Try a random edge swap
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
        if len(all_edges) < 2:
            break
        e1, e2 = random.sample(all_edges, 2)
        a, b = e1
        c, d = e2
        
        new_adj = None
        # Try (a,c) and (b,d)
        if a != c and b != d and not adj[a][c] and not adj[b][d]:
            new_adj = [row[:] for row in adj]
            new_adj[a][b] = new_adj[b][a] = False
            new_adj[c][d] = new_adj[d][c] = False
            new_adj[a][c] = new_adj[c][a] = True
            new_adj[b][d] = new_adj[d][b] = True
        # Try (a,d) and (b,c)
        if new_adj is None and a != d and b != c and not adj[a][d] and not adj[b][c]:
            new_adj = [row[:] for row in adj]
            new_adj[a][b] = new_adj[b][a] = False
            new_adj[c][d] = new_adj[d][c] = False
            new_adj[a][d] = new_adj[d][a] = True
            new_adj[b][c] = new_adj[c][b] = True
        
        if new_adj is None:
            continue
        
        # Compute energy change
        old_energy = count_short_cycles(adj, n, g)
        new_energy = count_short_cycles(new_adj, n, g)
        delta = new_energy - old_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            adj = new_adj
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    g_val = girth(adj, n)
    if g_val >= g:
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
        return n, edges
    
    # Fallback
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
    return n, edges
