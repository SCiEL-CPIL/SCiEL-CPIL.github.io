import random
from itertools import combinations

def construct():
    n = 0
    edges = []
    adj = []
    
    def is_valid_M(M, n_old, adj):
        # Check for k from 6 to n_old + 1
        for k in range(6, n_old + 2):
            size_s = k - 1
            if size_s > n_old:
                continue
            for S in combinations(range(n_old), size_s):
                S_mask = 0
                for u in S:
                    S_mask |= (1 << u)
                
                d_v = (S_mask & M).bit_count()
                
                valid_subgraph = True
                for u in S:
                    deg_u = (adj[u] & S_mask).bit_count()
                    edge_u = 1 if (M & (1 << u)) else 0
                    if deg_u + edge_u != d_v:
                        valid_subgraph = False
                        break
                
                if valid_subgraph:
                    return False
        return True

    for new_n in range(1, 23):
        n_old = new_n - 1
        if n_old == 0:
            adj.append(0)
            n = 1
            continue
            
        found_valid_M = False
        
        candidates = []
        if n_old <= 10:
            for m in range(2**n_old):
                candidates.append(m)
        else:
            candidates.append(0)
            candidates.append((1 << n_old) - 1)
            
            # Number of random candidates
            if n_old <= 15:
                num_rand = 2000
            else:
                num_rand = 500
                
            for _ in range(num_rand):
                candidates.append(random.randint(0, (1 << n_old) - 1))
                
            # Add some structured candidates
            for _ in range(100):
                m = 0
                for i in range(n_old):
                    if random.random() < 0.5:
                        m |= (1 << i)
                candidates.append(m)

        random.shuffle(candidates)
        
        for M in candidates:
            if is_valid_M(M, n_old, adj):
                adj.append(M)
                n = new_n
                found_valid_M = True
                break
                
        if not found_valid_M:
            break
            
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
