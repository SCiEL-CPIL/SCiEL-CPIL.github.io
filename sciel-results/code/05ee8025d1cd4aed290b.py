import time
import random

def _check_k33(adj):
    n = 20
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                common = adj[a] & adj[b] & adj[c]
                common &= ~((1 << a) | (1 << b) | (1 << c))
                if bin(common).count('1') >= 3:
                    verts = [i for i in range(n) if common & (1 << i)]
                    for i in range(len(verts)):
                        for j in range(i + 1, len(verts)):
                            for k in range(j + 1, len(verts)):
                                vi, vj, vk = verts[i], verts[j], verts[k]
                                if (adj[vi] & (1 << vj)) and (adj[vi] & (1 << vk)) and (adj[vj] & (1 << vk)):
                                    return True
    return False

def _edge_count(adj):
    return sum(bin(adj[i]).count('1') for i in range(20)) // 2

def _greedy_build(seed):
    random.seed(seed)
    adj = [0] * 20
    edges = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    random.shuffle(edges)
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if _check_k33(adj):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return adj

def _try_add_edge(adj, u, v):
    """Try to add edge (u,v). Return True if successful."""
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    if _check_k33(adj):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False
    return True

def _local_search(adj, max_time=5.0):
    start = time.time()
    n = 20
    current = adj[:]
    best_count = _edge_count(current)
    
    while time.time() - start < max_time:
        improved = False
        # Try adding each non-edge
        for i in range(n):
            for j in range(i + 1, n):
                if not (current[i] & (1 << j)):
                    if _try_add_edge(current, i, j):
                        new_count = _edge_count(current)
                        if new_count > best_count:
                            best_count = new_count
                        improved = True
        if not improved:
            break
    return current, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_adj = None
    best_count = 0
    
    for seed in range(200):
        if time.time() > deadline:
            break
        
        adj = _greedy_build(seed)
        count = _edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        if time.time() > deadline - 30:
            break
        
        adj_ls, count_ls = _local_search(adj, max_time=3.0)
        if count_ls > best_count:
            best_count = count_ls
            best_adj = adj_ls[:]
    
    if best_adj is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(20):
        for j in range(i + 1, 20):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    return edges
