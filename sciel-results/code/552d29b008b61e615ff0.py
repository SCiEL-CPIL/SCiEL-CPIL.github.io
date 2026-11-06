import random
import time
import math

def construct():
    n = 16
    target_edges = 65
    
    # Initial state: K_{8,8} + edge (0,1)
    adj = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    
    # Calculate initial degrees
    deg = [bin(x).count('1') for x in adj]
    
    def get_score(adj_list, deg_list):
        max_sum = 0
        for i in range(n):
            di = deg_list[i]
            for j in range(i+1, n):
                if not (adj_list[i] & (1 << j)):
                    continue
                dij = di + deg_list[j]
                for k in range(j+1, n):
                    if (adj_list[i] & (1 << k)) and (adj_list[j] & (1 << k)):
                        s = dij + deg_list[k]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    current_score = get_score(adj, deg)
    best_score = current_score
    best_adj = adj[:]
    best_deg = deg[:]
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    T = 5.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick random edge to remove
        while True:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            if adj[i] & (1 << j):
                e_rem = (i, j)
                break
        
        # Pick random non-edge to add
        while True:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            if i > j: i, j = j, i
            if not (adj[i] & (1 << j)):
                e_add = (i, j)
                break
        
        # Apply swap to new arrays
        new_adj = adj[:]
        new_deg = deg[:]
        
        u, v = e_rem
        new_adj[u] &= ~(1 << v)
        new_adj[v] &= ~(1 << u)
        new_deg[u] -= 1
        new_deg[v] -= 1
        
        u, v = e_add
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        new_deg[u] += 1
        new_deg[v] += 1
        
        new_score = get_score(new_adj, new_deg)
        
        if new_score < current_score:
            current_score = new_score
            adj = new_adj
            deg = new_deg
            if current_score < best_score:
                best_score = current_score
                best_adj = adj[:]
                best_deg = deg[:]
        else:
            delta = new_score - current_score
            if delta > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                    adj = new_adj
                    deg = new_deg
            
        T *= cooling
        if T < 0.5:
            T = 0.5
            
    # Convert best_adj back to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
