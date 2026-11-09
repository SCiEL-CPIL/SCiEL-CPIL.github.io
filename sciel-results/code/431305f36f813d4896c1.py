import time
import random

def construct():
    n = 20
    adj = [0] * n
    
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')
        
    def can_add(u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
            
        # Check neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            if popcount(adj[w] & adj[v]) >= 2:
                return False
            mask_u ^= lsb
            
        # Check neighbors of v
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            x = lsb.bit_length() - 1
            if popcount(adj[u] & adj[x]) >= 2:
                return False
            mask_v ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def count_edges():
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    def get_edges_list():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    def find_violation():
        for u in range(n):
            for v in range(u+1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return u, v
        return None

    def clean_graph():
        while True:
            viol = find_violation()
            if not viol:
                break
            u, v = viol
            common = adj[u] & adj[v]
            ws = []
            mask = common
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                ws.append(w)
                mask ^= lsb
            if not ws:
                break
            w = random.choice(ws)
            if random.random() < 0.5:
                remove_edge(u, w)
            else:
                remove_edge(v, w)

    start_time = time.time()
    deadline = start_time + 115.0
    
    best_adj = [0] * n
    best_count = 0
    
    while time.time() < deadline:
        p = random.choice([0.4, 0.6, 0.8])
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    add_edge(i, j)
        
        clean_graph()
        
        while time.time() < deadline:
            added = False
            for i in range(n):
                for j in range(i+1, n):
                    if not ((adj[i] >> j) & 1):
                        if can_add(i, j):
                            add_edge(i, j)
                            added = True
                            
            current_count = count_edges()
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
                
            if not added:
                existing = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            existing.append((i, j))
                        mask ^= lsb
                            
                if not existing:
                    break
                    
                u, v = random.choice(existing)
                remove_edge(u, v)
                
            if time.time() > deadline:
                break

    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges
