import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0

    def time_left():
        return deadline - (time.time() - start_time)

    def adj_to_orient(n, adj):
        orient = []
        for i in range(n):
            for j in range(i+1, n):
                orient.append(0 if adj[i][j] == 1 else 1)
        return orient

    def count_violations(n, adj, out_mask):
        count = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            found = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    found = True
                    break
            if not found:
                count += 1
        return count

    def simulate_annealing(n, initial_adj, max_iters=50000):
        adj = [row[:] for row in initial_adj]
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
            
        current_cost = count_violations(n, adj, out_mask)
        best_cost = current_cost
        best_adj = [row[:] for row in adj]
        
        T = 1.0
        cooling = 0.9999
        
        for i in range(max_iters):
            if time_left() < 5:
                break
            
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
                
            old_uv = adj[u][v]
            adj[u][v] = 1 - old_uv
            adj[v][u] = 1 - adj[u][v]
            
            if adj[u][v]:
                out_mask[u] |= (1 << v)
                out_mask[v] &= ~(1 << u)
            else:
                out_mask[u] &= ~(1 << v)
                out_mask[v] |= (1 << u)
                
            new_cost = count_violations(n, adj, out_mask)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = [row[:] for row in adj]
                    if best_cost == 0:
                        break
            else:
                adj[u][v] = old_uv
                adj[v][u] = 1 - old_uv
                if old_uv:
                    out_mask[u] |= (1 << v)
                    out_mask[v] &= ~(1 << u)
                else:
                    out_mask[u] &= ~(1 << v)
                    out_mask[v] |= (1 << u)
                    
            T *= cooling
            
        return best_adj, best_cost

    # Fallback: Paley(83)
    p = 83
    squares = set(x * x % p for x in range(1, p))
    adj_paley = [[0]*p for _ in range(p)]
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                adj_paley[i][j] = 1
            else:
                adj_paley[j][i] = 1
    best_n = p
    best_orient = adj_to_orient(p, adj_paley)
    
    # Try Simulated Annealing from random starts for smaller n
    for n in range(67, 83):
        if time_left() < 15:
            break
        for seed in range(10):
            if time_left() < 10:
                break
            random.seed(seed)
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                    else:
                        adj[j][i] = 1
            best_adj, best_cost = simulate_annealing(n, adj, max_iters=20000)
            if best_cost == 0:
                best_n = n
                best_orient = adj_to_orient(n, best_adj)
                break
        if best_n < 83:
            break
            
    return {'n': best_n, 'orientation': best_orient}
