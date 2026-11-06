"""Graph on 16 vertices, 65 edges: minimize max triangle degree sum."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n = 16
    target_edges = 65
    
    def compute_score_adj(adj, n):
        """Compute max triangle degree sum with early termination."""
        deg = [0] * n
        for i in range(n):
            deg[i] = bin(adj[i]).count('1')
        
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    while common:
                        k_bit = common & (-common)
                        k = k_bit.bit_length() - 1
                        common ^= k_bit
                        if k > j:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
                                if max_sum >= 24:
                                    return max_sum
        return max_sum
    
    def edges_to_adj(edges, n):
        """Convert edge list to bitmask adjacency."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def build_bipartite(a_size, internal_a, internal_b, n):
        """Build edges for bipartite graph with internal edges."""
        edges = []
        for u in range(a_size):
            for v in range(a_size, n):
                edges.append([u, v])
        for e in internal_a:
            edges.append([e[0], e[1]])
        for e in internal_b:
            edges.append([e[0], e[1]])
        return edges
    
    def local_search(adj, edges, score, time_limit):
        """Single-edge swap local search with time limit."""
        cur_adj = adj[:]
        cur_edges = edges[:]
        cur_score = score
        improved = True
        round_num = 0
        max_rounds = 8
        
        while improved and round_num < max_rounds and time.time() < time_limit:
            improved = False
            round_num += 1
            for _ in range(50000):
                if time.time() > time_limit:
                    break
                
                idx = random.randint(0, len(cur_edges) - 1)
                u, v = cur_edges[idx]
                
                for attempt in range(20):
                    ru = random.randint(0, n-1)
                    rv = random.randint(0, n-1)
                    if ru == rv:
                        continue
                    if ru > rv:
                        ru, rv = rv, ru
                    if cur_adj[ru] & (1 << rv):
                        continue
                    if (ru == u and rv == v) or (ru == v and rv == u):
                        continue
                    
                    old_adj = cur_adj[:]
                    old_edges = cur_edges[:]
                    
                    cur_adj[u] &= ~(1 << v)
                    cur_adj[v] &= ~(1 << u)
                    cur_edges[idx] = [ru, rv]
                    cur_adj[ru] |= (1 << rv)
                    cur_adj[rv] |= (1 << ru)
                    
                    new_score = compute_score_adj(cur_adj, n)
                    
                    if new_score < cur_score:
                        cur_score = new_score
                        improved = True
                        break
                    else:
                        cur_adj = old_adj
                        cur_edges = old_edges
                        break
                else:
                    continue
                
                if time.time() > time_limit:
                    break
        
        return cur_edges, cur_adj, cur_score
    
    def two_opt(adj, edges, score, time_limit):
        """2-opt local search."""
        cur_adj = adj[:]
        cur_edges = edges[:]
        cur_score = score
        improved = True
        round_num = 0
        max_rounds = 4
        
        while improved and round_num < max_rounds and time.time() < time_limit:
            improved = False
            round_num += 1
            for _ in range(20000):
                if time.time() > time_limit:
                    break
                
                idx1 = random.randint(0, len(cur_edges) - 1)
                idx2 = random.randint(0, len(cur_edges) - 1)
                if idx1 == idx2:
                    continue
                u1, v1 = cur_edges[idx1]
                u2, v2 = cur_edges[idx2]
                
                for attempt in range(10):
                    ru1 = random.randint(0, n-1)
                    rv1 = random.randint(0, n-1)
                    ru2 = random.randint(0, n-1)
                    rv2 = random.randint(0, n-1)
                    if ru1 == rv1 or ru2 == rv2:
                        continue
                    if ru1 > rv1:
                        ru1, rv1 = rv1, ru1
                    if ru2 > rv2:
                        ru2, rv2 = rv2, ru2
                    
                    if cur_adj[ru1] & (1 << rv1):
                        continue
                    if cur_adj[ru2] & (1 << rv2):
                        continue
                    if (ru1 == u1 and rv1 == v1) or (ru1 == v1 and rv1 == u1):
                        continue
                    if (ru2 == u2 and rv2 == v2) or (ru2 == v2 and rv2 == u2):
                        continue
                    if (ru1 == u2 and rv1 == v2) or (ru1 == v2 and rv1 == u2):
                        continue
                    if (ru1 == u1 and rv1 == v2) or (ru1 == v2 and rv1 == u1):
                        continue
                    if ru1 == ru2 and rv1 == rv2:
                        continue
                    
                    old_adj = cur_adj[:]
                    old_edges = cur_edges[:]
                    
                    cur_adj[u1] &= ~(1 << v1)
                    cur_adj[v1] &= ~(1 << u1)
                    cur_adj[u2] &= ~(1 << v2)
                    cur_adj[v2] &= ~(1 << u2)
                    cur_edges[idx1] = [ru1, rv1]
                    cur_edges[idx2] = [ru2, rv2]
                    
                    cur_adj[ru1] |= (1 << rv1)
                    cur_adj[rv1] |= (1 << ru1)
                    cur_adj[ru2] |= (1 << rv2)
                    cur_adj[rv2] |= (1 << ru2)
                    
                    new_score = compute_score_adj(cur_adj, n)
                    
                    if new_score < cur_score:
                        cur_score = new_score
                        improved = True
                        break
                    else:
                        cur_adj = old_adj
                        cur_edges = old_edges
                        break
                    break
                
                if time.time() > time_limit:
                    break
        
        return cur_edges, cur_adj, cur_score
    
    random.seed(42)
    best_edges = None
    best_adj = None
    best_score = 999
    
    # Configuration: (a_size, n_internal_a, n_internal_b)
    # Must satisfy: a_size*(n-a_size) + n_internal_a + n_internal_b == 65
    configs = [
        (6, 0, 5),   # 6*10=60 + 5 = 65
        (7, 0, 2),   # 7*9=63 + 2 = 65
        (8, 0, 1),   # 8*8=64 + 1 = 65
        (6, 1, 4),   # 6*10=60 + 1 + 4 = 65
        (7, 1, 1),   # 7*9=63 + 1 + 1 = 65
        (5, 0, 8),   # 5*11=55 + 8 = 63... no, 55+8=63, need 65. Skip.
    ]
    
    # Filter valid configs
    valid_configs = []
    for a_size, na, nb in configs:
        if a_size * (n - a_size) + na + nb == target_edges:
            valid_configs.append((a_size, na, nb))
    
    # If no valid configs, add fallback
    if not valid_configs:
        valid_configs = [(6, 0, 5)]
    
    for a_size, na, nb in valid_configs:
        if time.time() > deadline - 20.0:
            break
        
        for trial in range(4):
            if time.time() > deadline - 15.0:
                break
            
            a_verts = list(range(a_size))
            b_verts = list(range(a_size, n))
            random.shuffle(a_verts)
            random.shuffle(b_verts)
            
            internal_a = []
            for i in range(na):
                internal_a.append([a_verts[2*i], a_verts[2*i+1]])
            
            internal_b = []
            for i in range(nb):
                internal_b.append([b_verts[2*i], b_verts[2*i+1]])
            
            edges = build_bipartite(a_size, internal_a, internal_b, n)
            if len(edges) != target_edges:
                continue
            
            adj = edges_to_adj(edges, n)
            score = compute_score_adj(adj, n)
            
            if score < best_score:
                best_edges = edges[:]
                best_adj = adj[:]
                best_score = score
            
            # Local search
            if time.time() < deadline - 15.0:
                e, a, s = local_search(adj, edges, score, deadline - 10.0)
                if s < best_score:
                    best_edges = e[:]
                    best_adj = a[:]
                    best_score = s
            
            # 2-opt
            if time.time() < deadline - 10.0 and best_score <= 22:
                e, a, s = two_opt(best_adj, best_edges, best_score, deadline - 5.0)
                if s < best_score:
                    best_edges = e[:]
                    best_adj = a[:]
                    best_score = s
    
    # Final polish: more aggressive single-edge search on best
    if best_score < 999 and time.time() < deadline - 5.0:
        e, a, s = local_search(best_adj, best_edges, best_score, deadline - 2.0)
        if s < best_score:
            best_edges = e[:]
            best_adj = a[:]
            best_score = s
    
    return {'n': n, 'edges': best_edges}
