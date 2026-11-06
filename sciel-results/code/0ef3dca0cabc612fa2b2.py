import time
import random

DEADLINE = 145

def _has_k4_in_subgraph(adj, mask):
    m = mask
    while m:
        b_u = m & (-m)
        u = b_u.bit_length() - 1
        remaining = mask & ~( (1 << (u+1)) - 1 )
        while remaining:
            b_v = remaining & (-remaining)
            v = b_v.bit_length() - 1
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v] & mask
                m_c = common
                while m_c:
                    b_w = m_c & (-m_c)
                    w = b_w.bit_length() - 1
                    if (adj[w] & common):
                        return True
                    m_c ^= b_w
            remaining ^= b_v
        m ^= b_u
    return False

def _has_k4(adj, n):
    return _has_k4_in_subgraph(adj, (1 << n) - 1)

def _has_independent_6_fast(adj, n):
    comp_adj = [0] * n
    for i in range(n):
        mask = (1 << n) - 1
        comp_adj[i] = mask & ~(1 << i) & ~adj[i]
    
    for u in range(n):
        for v in range(u+1, n):
            if (comp_adj[u] >> v) & 1:
                common = comp_adj[u] & comp_adj[v]
                if _has_k4_in_subgraph(comp_adj, common):
                    return True
    return False

def _validate(adj, n):
    if n < 1 or n > 40:
        return False
    if _has_k4(adj, n):
        return False
    if _has_independent_6_fast(adj, n):
        return False
    return True

def _local_search(n, time_limit, seed=0):
    t0 = time.time()
    random.seed(seed)
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_adj = adj[:]
    best_violations = (1 if _has_k4(adj, n) else 0) + (1 if _has_independent_6_fast(adj, n) else 0)
    if best_violations == 0:
        return best_adj
    
    temperature = 1.0
    cooling = 0.999
    
    while time.time() - t0 < time_limit:
        u = random.randint(0, n-1)
        v = random.randint(u+1, n-1)
        
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        has_k4 = _has_k4(adj, n)
        has_i6 = _has_independent_6_fast(adj, n)
        violations = (1 if has_k4 else 0) + (1 if has_i6 else 0)
        
        if violations < best_violations:
            best_violations = violations
            best_adj = adj[:]
            if best_violations == 0:
                return best_adj
        
        temperature *= cooling

    return best_adj

def construct():
    t0 = time.time()
    
    for n_target in [36, 35]:
        for seed in range(50):
            if time.time() - t0 > DEADLINE - 10:
                break
            adj = _local_search(n_target, time_limit=2.0, seed=seed)
            if _validate(adj, n_target):
                edge_set = []
                for i in range(n_target):
                    for j in range(i+1, n_target):
                        if (adj[i] >> j) & 1:
                            edge_set.append([i, j])
                return {'n': n_target, 'edges': edge_set}
                
    return {'n': 5, 'edges': [[0,1],[1,2],[2,3],[3,4],[4,0]]}
