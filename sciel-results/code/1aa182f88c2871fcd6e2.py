"""Minimize 4-cycles in a 31-vertex, 91-edge graph using polarity graph + optimal extra edge."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 31
    m = 91
    
    # Enumerate points of PG(2,5)
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)
                        vec = [(x * inv) % 5 for x in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31
    
    def orthogonal(u, v):
        return (u[0]*v[1] + u[1]*v[0] + u[2]*v[2]) % 5 == 0
    
    # Build polarity graph using numpy
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if orthogonal(points[i], points[j]):
                adj[i, j] = adj[j, i] = 1
                edges.append((i, j))
    
    polarity_edge_count = len(edges)
    
    def count_c4s(adj_mat):
        count = 0
        for a in range(n):
            for c in range(a+1, n):
                common = int(np.sum(adj_mat[a] * adj_mat[c]))
                count += common * (common - 1) // 2
        return count // 2
    
    base_c4s = count_c4s(adj)
    
    if polarity_edge_count == m:
        return [list(e) for e in edges]
    
    if polarity_edge_count > m:
        return [list(e) for e in edges[:m]]
    
    num_to_add = m - polarity_edge_count
    
    non_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i, j]:
                non_edges.append((i, j))
    
    def c4s_created(u, v, adj_mat):
        """C4s created by adding edge (u,v) = number of length-3 paths u-a-b-v."""
        count = 0
        for a in range(n):
            if a == u or a == v or not adj_mat[u, a]:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if adj_mat[v, b] and adj_mat[a, b]:
                    count += 1
        return count
    
    cur_adj = adj.copy()
    cur_edges = list(edges)
    
    for _ in range(num_to_add):
        if time.time() > deadline - 5:
            break
        best_edge = None
        best_c4s = float('inf')
        for (u, v) in non_edges:
            if cur_adj[u, v]:
                continue
            c4s = c4s_created(u, v, cur_adj)
            if c4s < best_c4s:
                best_c4s = c4s
                best_edge = (u, v)
        if best_edge is None:
            break
        u, v = best_edge
        cur_adj[u, v] = cur_adj[v, u] = 1
        cur_edges.append((u, v))
    
    if time.time() > deadline - 10:
        return [list(e) for e in cur_edges]
    
    def c4s_containing(u, v, adj_mat):
        count = 0
        for a in range(n):
            if a == u or a == v or not adj_mat[u, a]:
                continue
            for b in range(n):
                if b == u or b == v or b == a:
                    continue
                if adj_mat[v, b] and adj_mat[a, b]:
                    count += 1
        return count
    
    edge_set = set(cur_edges)
    non_edge_set = set(non_edges) - set(cur_edges)
    
    improved = True
    max_iters = 50
    iter_count = 0
    while improved and time.time() < deadline - 5 and iter_count < max_iters:
        improved = False
        best_swap = None
        best_gain = 0
        iter_count += 1
        
        edge_list = list(edge_set)
        random.seed(42 + iter_count)
        sample_size = min(len(edge_list), 30)
        sampled = random.sample(edge_list, sample_size)
        
        for e1 in sampled:
            u1, v1 = e1
            destroyed = c4s_containing(u1, v1, cur_adj)
            if destroyed <= 0:
                continue
            non_edge_list = list(non_edge_set)
            non_sample_size = min(len(non_edge_list), 30)
            sampled_ne = random.sample(non_edge_list, non_sample_size)
            for e2 in sampled_ne:
                u2, v2 = e2
                created = c4s_created(u2, v2, cur_adj)
                gain = destroyed - created
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (e1, e2)
        
        if best_swap is not None:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            cur_adj[u1, v1] = cur_adj[v1, u1] = 0
            cur_adj[u2, v2] = cur_adj[v2, u2] = 1
            edge_set.remove(e1)
            non_edge_set.remove(e2)
            edge_set.add(e2)
            non_edge_set.add(e1)
            improved = True
    
    return [list(e) for e in edge_set]
