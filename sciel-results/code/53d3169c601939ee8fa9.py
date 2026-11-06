import random
import time

def construct():
    start_time = time.time()
    time_limit = 118
    n = 16
    
    def find_q3(adj):
        for v0 in range(n):
            mask_v0 = adj[v0]
            neighs = []
            m = mask_v0
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                neighs.append(v)
                m ^= lsb
            
            for i in range(len(neighs)):
                v1 = neighs[i]
                for j in range(i+1, len(neighs)):
                    v2 = neighs[j]
                    for k in range(j+1, len(neighs)):
                        v4 = neighs[k]
                        
                        c12 = adj[v1] & adj[v2] & ~mask_v0
                        c14 = adj[v1] & adj[v4] & ~mask_v0
                        c24 = adj[v2] & adj[v4] & ~mask_v0
                        
                        m3 = c12
                        while m3:
                            lsb = m3 & -m3
                            v3 = lsb.bit_length() - 1
                            m3 ^= lsb
                            
                            if v3 == v1 or v3 == v2 or v3 == v4: continue
                            
                            m5 = c14
                            while m5:
                                lsb = m5 & -m5
                                v5 = lsb.bit_length() - 1
                                m5 ^= lsb
                                
                                if v5 == v1 or v5 == v2 or v5 == v4 or v5 == v3: continue
                                
                                m6 = c24
                                while m6:
                                    lsb = m6 & -m6
                                    v6 = lsb.bit_length() - 1
                                    m6 ^= lsb
                                    
                                    if v6 == v1 or v6 == v2 or v6 == v4 or v6 == v3 or v6 == v5: continue
                                    
                                    c356 = adj[v3] & adj[v5] & adj[v6]
                                    c356 &= ~mask_v0
                                    c356 &= ~(1 << v1)
                                    c356 &= ~(1 << v2)
                                    c356 &= ~(1 << v4)
                                    c356 &= ~(1 << v3)
                                    c356 &= ~(1 << v5)
                                    c356 &= ~(1 << v6)
                                    
                                    if c356:
                                        return [
                                            (v0, v1), (v0, v2), (v0, v4),
                                            (v1, v3), (v1, v5),
                                            (v2, v3), (v2, v6),
                                            (v3, v7),
                                            (v4, v5), (v4, v6),
                                            (v5, v7), (v6, v7)
                                        ]
        return None

    def edge_count(adj):
        return sum(bin(x).count('1') for x in adj) // 2

    best_adj = None
    best_edges = 0
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(123 + restart)
        
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.7:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        while True:
            if time.time() - start_time > time_limit:
                break
            q3 = find_q3(adj)
            if q3 is None:
                break
            u, v = random.choice(q3)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
        if time.time() - start_time > time_limit:
            break
            
        ec = edge_count(adj)
        
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
            
            if random.random() < 0.8:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if (adj[u] >> v) & 1: continue
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if find_q3(adj) is None:
                    ec += 1
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                edges = []
                for i in range(n):
                    m = adj[i]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append((i,j))
                        m ^= lsb
                            
                if not edges: continue
                u, v = random.choice(edges)
                
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if (x,y) == (u,v) or (x,y) == (v,u): continue
                if (adj[x] >> y) & 1: continue
                
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
                
                if find_q3(adj) is None:
                    pass
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    adj[x] &= ~(1 << y)
                    adj[y] &= ~(1 << x)
                    
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
            
    if best_adj is None:
        return []
        
    res = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
    return res
