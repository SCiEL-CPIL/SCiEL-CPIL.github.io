import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0

def _score_from_adj(adj, deg):
    best = 0
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j]:
                for k in range(j + 1, N):
                    if adj[i][k] and adj[j][k]:
                        s = deg[i] + deg[j] + deg[k]
                        if s > best:
                            best = s
    return best

def construct():
    t0 = time.time()
    rng = random.Random(12345)
    
    best_adj = None
    best_deg = None
    best_score = float('inf')
    
    num_restarts = 40
    
    for r in range(num_restarts):
        if time.time() - t0 > DEADLINE - 5:
            break
        
        # Initialize
        adj = [[False] * N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        
        if r % 2 == 0:
            u, v = rng.sample(range(8), 2)
        else:
            u, v = rng.sample(range(8, 16), 2)
        adj[u][v] = adj[v][u] = True
        
        deg = [0] * N
        for i in range(N):
            for j in range(N):
                if adj[i][j]:
                    deg[i] += 1
        
        cur_score = _score_from_adj(adj, deg)
        
        if cur_score < best_score:
            best_score = cur_score
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
        
        edges = []
        nonedges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i][j]:
                    edges.append((i, j))
                else:
                    nonedges.append((i, j))
        
        start_temp = 5.0
        end_temp = 0.1
        max_iters = 8000
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        
        for it in range(max_iters):
            if time.time() - t0 > DEADLINE - 2:
                break
            
            # Move Strategy
            if rng.random() < 0.5:
                e_remove = edges[rng.randrange(len(edges))]
                e_add = nonedges[rng.randrange(len(nonedges))]
            else:
                # Balancing: Remove high-deg edge, add low-deg edge
                # Find max deg sum edge
                max_sum = -1
                e_remove = edges[0]
                for e in edges:
                    u, v = e
                    s = deg[u] + deg[v]
                    if s > max_sum:
                        max_sum = s
                        e_remove = e
                
                min_sum = float('inf')
                e_add = nonedges[0]
                for e in nonedges:
                    u, v = e
                    s = deg[u] + deg[v]
                    if s < min_sum:
                        min_sum = s
                        e_add = e
                
                # Noise
                if rng.random() < 0.3:
                    e_remove = edges[rng.randrange(len(edges))]
                    e_add = nonedges[rng.randrange(len(nonedges))]
            
            u, v = e_remove
            x, y = e_add
            
            if (u == x and v == y) or (u == y and v == x):
                temp *= cooling_rate
                continue
            
            # Apply
            adj[u][v] = adj[v][u] = False
            deg[u] -= 1
            deg[v] -= 1
            adj[x][y] = adj[y][x] = True
            deg[x] += 1
            deg[y] += 1
            
            new_score = _score_from_adj(adj, deg)
            
            if new_score < cur_score:
                accept = True
            elif new_score == cur_score:
                accept = rng.random() < 0.5
            else:
                delta = new_score - cur_score
                if temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-delta / temp) > rng.random()
            
            if accept:
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                
                edges.remove(e_remove)
                nonedges.append(e_remove)
                nonedges.remove(e_add)
                edges.append(e_add)
            else:
                # Undo
                adj[x][y] = adj[y][x] = False
                deg[x] -= 1
                deg[y] -= 1
                adj[u][v] = adj[v][u] = True
                deg[u] += 1
                deg[v] += 1
            
            temp *= cooling_rate
        
        # Local Search Polish
        # Run a few iterations of strict hill climbing on the best found so far
        # This helps if SA stopped just above a better solution
        if time.time() - t0 < DEADLINE - 5:
            cur_adj = [row[:] for row in best_adj]
            cur_deg = best_deg[:]
            cur_s = best_score
            improved = True
            while improved and time.time() - t0 < DEADLINE - 2:
                improved = False
                edges_ls, nonedges_ls = [], []
                for i in range(N):
                    for j in range(i + 1, N):
                        if cur_adj[i][j]:
                            edges_ls.append((i, j))
                        else:
                            nonedges_ls.append((i, j))
                
                # Try random swaps
                for _ in range(500):
                    if time.time() - t0 > DEADLINE - 2:
                        break
                    e_rm = edges_ls[rng.randrange(len(edges_ls))]
                    e_ad = nonedges_ls[rng.randrange(len(nonedges_ls))]
                    
                    u, v = e_rm
                    x, y = e_ad
                    if (u == x and v == y) or (u == y and v == x):
                        continue
                    
                    # Apply
                    cur_adj[u][v] = cur_adj[v][u] = False
                    cur_deg[u] -= 1
                    cur_deg[v] -= 1
                    cur_adj[x][y] = cur_adj[y][x] = True
                    cur_deg[x] += 1
                    cur_deg[y] += 1
                    
                    new_s = _score_from_adj(cur_adj, cur_deg)
                    
                    if new_s < cur_s:
                        cur_s = new_s
                        improved = True
                        # Update lists
                        edges_ls.remove(e_rm)
                        nonedges_ls.append(e_rm)
                        nonedges_ls.remove(e_ad)
                        edges_ls.append(e_ad)
                    else:
                        # Undo
                        cur_adj[x][y] = cur_adj[y][x] = False
                        cur_deg[x] -= 1
                        cur_deg[y] -= 1
                        cur_adj[u][v] = cur_adj[v][u] = True
                        cur_deg[u] += 1
                        cur_deg[v] += 1
                
                if improved:
                    best_score = cur_s
                    best_adj = [row[:] for row in cur_adj]
                    best_deg = cur_deg[:]

    if best_adj is None:
        adj = [[False] * N for _ in range(N)]
        for i in range(8):
            for j in range(8, 16):
                adj[i][j] = adj[j][i] = True
        adj[0][1] = adj[1][0] = True
        best_adj = adj

    return {'n': N, 'edges': [[i, j] for i in range(N) for j in range(i + 1, N) if best_adj[i][j]]}
