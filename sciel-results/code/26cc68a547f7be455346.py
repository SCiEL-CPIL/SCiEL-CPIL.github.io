import random
import time

def has_k4(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v]
                c = common
                while c:
                    lsb = c & -c
                    w = lsb.bit_length() - 1
                    if adj[w] & common:
                        return True
                    c ^= lsb
    return False

def has_independent_set_6(adj, n):
    # Build complement
    comp = [0] * n
    for i in range(n):
        mask = (1 << n) - 1
        mask ^= (1 << i)
        comp[i] = mask & ~adj[i]
        
    def dfs(candidates_mask, size):
        if size == 0:
            return True
        if candidates_mask == 0:
            return False
        if size == 1:
            return True
        # Count bits
        if bin(candidates_mask).count('1') < size:
            return False
        
        # Pick a vertex (lowest set bit)
        lsb = candidates_mask & -candidates_mask
        v = lsb.bit_length() - 1
        
        # Include v
        new_cands = candidates_mask & comp[v]
        if dfs(new_cands, size - 1):
            return True
        
        # Exclude v
        candidates_mask ^= lsb
        if dfs(candidates_mask, size):
            return True
        
        return False

    full_mask = (1 << n) - 1
    return dfs(full_mask, 6)

def build_adj_from_distances(n, distances):
    adj = [0] * n
    for d in distances:
        if d == 0:
            continue
        for i in range(n):
            j = (i + d) % n
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    # Ensure no self loops
    for i in range(n):
        adj[i] &= ~(1 << i)
    return adj

def construct():
    best_graph = None
    best_n = 0
    
    random.seed(42)
    start_time = time.time()
    time_limit = 110.0
    
    for n in range(35, 29, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Strategy 1: Circulant Graphs
        for _ in range(2000):
            if time.time() - start_time > time_limit:
                break
            max_d = n // 2
            distances = set()
            for d in range(1, max_d + 1):
                if random.random() < 0.5:
                    distances.add(d)
            
            adj = build_adj_from_distances(n, distances)
            
            if not has_k4(adj, n) and not has_independent_set_6(adj, n):
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
        
        # Strategy 2: Random General Graph Local Search
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        for _ in range(2000):
            if time.time() - start_time > time_limit:
                break
                
            if not has_k4(adj, n) and not has_independent_set_6(adj, n):
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
    # Fallback: Return a small valid graph
    # n=1 is always valid
    return {'n': 1, 'edges': []}
