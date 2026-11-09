import numpy as np
import random
import time

def construct():
    # New Approach: Local Search with Constraint Repair
    # We search for a graph on n vertices that has no K4 and no independent set of size 6.
    # We start with a random graph and iteratively fix violations.
    
    for n in range(35, 0, -1):
        start_time = time.time()
        max_time_per_n = 15.0
        
        # Try different edge probabilities
        for p in [0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]:
            if time.time() - start_time > max_time_per_n:
                break
                
            for attempt in range(20):
                if time.time() - start_time > max_time_per_n:
                    break
                    
                # Initialize random graph
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            
                # Local Search Loop
                for step in range(10000):
                    if time.time() - start_time > max_time_per_n:
                        break
                        
                    # 1. Check for K4
                    k4 = _find_k4(adj, n)
                    if k4:
                        edges_in_k4 = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]), 
                                       (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
                        u, v = random.choice(edges_in_k4)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        continue
                    
                    # 2. Check for Independent Set of 6
                    indep6 = _find_indep6(adj, n)
                    if indep6:
                        pair_idx = random.sample(range(6), 2)
                        u = indep6[pair_idx[0]]
                        v = indep6[pair_idx[1]]
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        continue
                    
                    # No violations found
                    if not _has_k4(adj, n) and not _has_indep6(adj, n):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i] & (1 << j):
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                
                # Random kick to escape local minimum
                for _ in range(20):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u != v:
                        if adj[u] & (1 << v):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

    # Fallback: Paley graph of order 17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def _find_k4(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                continue
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            c = common
            while c:
                lsb_w = c & -c
                w = lsb_w.bit_length() - 1
                c ^= lsb_w
                neighbors_in_common = adj[w] & common
                neighbors_in_common &= ~(1 << w)
                if neighbors_in_common:
                    lsb_x = neighbors_in_common & -neighbors_in_common
                    x = lsb_x.bit_length() - 1
                    return (u, v, w, x)
    return None

def _find_indep6(adj, n):
    comp_adj = [0] * n
    mask_all = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = mask_all ^ (1 << i) ^ adj[i]
        
    def find_clique_path(mask, k, path):
        if k == 0:
            return list(path)
        if k == 1:
            if mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                return path + [v]
            return None
        c = mask
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            next_mask = mask & comp_adj[v] & (mask_all << (v + 1))
            if next_mask:
                res = find_clique_path(next_mask, k - 1, path + [v])
                if res:
                    return res
        return None

    return find_clique_path(mask_all, 6, [])

def _has_k4(adj, n):
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] & (1 << v)):
                continue
            common = adj[u] & adj[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            c = common
            while c:
                lsb_w = c & -c
                w = lsb_w.bit_length() - 1
                c ^= lsb_w
                neighbors_in_common = adj[w] & common
                neighbors_in_common &= ~(1 << w)
                if neighbors_in_common:
                    return True
    return False

def _has_indep6(adj, n):
    comp_adj = [0] * n
    mask_all = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = mask_all ^ (1 << i) ^ adj[i]
        
    def find_clique(mask, k):
        if k == 0:
            return True
        if k == 1:
            return bool(mask)
        c = mask
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c ^= lsb
            next_mask = mask & comp_adj[v] & (mask_all << (v + 1))
            if next_mask:
                if find_clique(next_mask, k - 1):
                    return True
        return False

    return find_clique(mask_all, 6)
