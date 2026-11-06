import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_edges = []
    best_count = 0
    
    def recompute(adj):
        N1 = adj[:]
        N2 = [0] * n
        for i in range(n):
            mask = 0
            x = N1[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N1[j]
                x ^= lsb
            N2[i] = mask
        
        N4 = [0] * n
        for i in range(n):
            mask = 0
            x = N2[i]
            while x:
                lsb = x & -x
                j = lsb.bit_length() - 1
                mask |= N2[j]
                x ^= lsb
            N4[i] = mask
        return N1, N2, N4

    def is_safe(u, v, N1, N4):
        return (N1[u] & N4[v]) == 0 and (N1[v] & N4[u]) == 0

    def greedy_from(start_adj, order=None):
        adj = start_adj[:]
        N1, N2, N4 = recompute(adj)
        
        if order is None:
            order = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(order)
        
        added = 0
        for u, v in order:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if is_safe(u, v, N1, N4):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    added += 1
                    N1, N2, N4 = recompute(adj)
        
        # Convert to list of edges
        edges = []
        for u in range(n):
            x = adj[u]
            while x:
                lsb = x & -x
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                x ^= lsb
        return edges, len(edges)

    # Strategy 1: Random greedy from empty
    # Strategy 2: Greedy from seed (8xK5)
    
    # Seed graph: 8 disjoint K5s
    seed_adj = [0] * n
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                seed_adj[u] |= 1 << v
                seed_adj[v] |= 1 << u
    
    # Run multiple random restarts
    while time.time() < deadline:
        # Random greedy from empty
        edges, count = greedy_from([0] * n)
        if count > best_count:
            best_count = count
            best_edges = edges
            
        # Occasionally try from seed
        if random.random() < 0.1:
            edges, count = greedy_from(seed_adj)
            if count > best_count:
                best_count = count
                best_edges = edges

    return best_edges
