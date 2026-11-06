import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin under 120s limit
    
    n = 40
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_pairs = len(all_pairs)
    
    # Map pair to index
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    
    # Helper to check if adding edge (u, v) creates a C6
    # adj is a list of bitmasks
    def can_add_edge(adj, u, v):
        nu_mask = adj[u] & ~(1 << v)
        nv_mask = adj[v] & ~(1 << u)
        
        a = nu_mask
        while a:
            lb_a = a & (-a)
            ai = lb_a.bit_length() - 1
            a ^= lb_a
            
            d = nv_mask & ~lb_a
            while d:
                ld_d = d & (-d)
                di = ld_d.bit_length() - 1
                d ^= ld_d
                
                b_candidates = adj[ai] & ~(1 << u) & ~(1 << v) & ~(1 << di) & ~(1 << ai)
                c_candidates = adj[di] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << di)
                
                union_neighbors = 0
                bm = b_candidates
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    union_neighbors |= adj[bi]
                    if union_neighbors & c_candidates:
                        return False
        return True

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    def adj_from_edges(edge_list):
        adj = [0] * n
        for u, v in edge_list:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def edges_from_adj(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    best_adj = None
    best_count = 0
    
    # Phase 1: Bipartite Union Initialization (per Directive)
    # Create two random bipartitions
    for restart in range(5):
        if time.time() > deadline - 60:
            break
            
        random.seed(42 + restart)
        
        # Partition 1
        perm1 = list(range(n))
        random.shuffle(perm1)
        A1 = set(perm1[:20])
        B1 = set(perm1[20:])
        
        # Partition 2
        perm2 = list(range(n))
        random.shuffle(perm2)
        A2 = set(perm2[:20])
        B2 = set(perm2[20:])
        
        # Generate E1 (edges between A1 and B1)
        E1 = set()
        for u in A1:
            for v in B1:
                if random.random() < 0.5:
                    if u < v:
                        E1.add((u, v))
                    else:
                        E1.add((v, u))
                        
        # Generate E2 (edges between A2 and B2), disjoint from E1
        E2 = set()
        for u in A2:
            for v in B2:
                if u < v:
                    p = (u, v)
                else:
                    p = (v, u)
                if p not in E1 and random.random() < 0.5:
                    E2.add(p)
                    
        # Union
        E_init = E1.union(E2)
        
        # Convert to adj
        current_adj = [0] * n
        for u, v in E_init:
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
        # Prune C6s
        # Find C6s and remove edges
        def find_c6(adj):
            for u in range(n):
                nu = adj[u]
                nv_candidates = ~nu & ~(1 << u) & ((1 << n) - 1)
                
                v = nv_candidates
                while v:
                    lb_v = v & (-v)
                    vi = lb_v.bit_length() - 1
                    v ^= lb_v
                    
                    a_mask = nu
                    d_mask = adj[vi]
                    
                    am = a_mask
                    while am:
                        lb_a = am & (-am)
                        ai = lb_a.bit_length() - 1
                        am ^= lb_a
                        
                        dm = d_mask & ~lb_a
                        while dm:
                            lb_d = dm & (-dm)
                            di = lb_d.bit_length() - 1
                            dm ^= lb_d
                            
                            b_cand = adj[ai] & ~(1 << u) & ~(1 << vi) & ~lb_d & ~lb_a
                            c_cand = adj[di] & ~(1 << u) & ~(1 << vi) & ~lb_a & ~lb_d
                            
                            bm = b_cand
                            union_b = 0
                            while bm:
                                lb_b = bm & (-bm)
                                bi = lb_b.bit_length() - 1
                                bm ^= lb_b
                                union_b |= adj[bi]
                                if union_b & c_cand:
                                    bm2 = b_cand
                                    while bm2:
                                        lb_b2 = bm2 & (-bm2)
                                        bi2 = lb_b2.bit_length() - 1
                                        bm2 ^= lb_b2
                                        if adj[bi2] & c_cand:
                                            cm = adj[bi2] & c_cand
                                            lb_c = cm & (-cm)
                                            ci = lb_c.bit_length() - 1
                                            return [(u, ai), (ai, bi2), (bi2, ci), (ci, di), (di, vi), (vi, u)]
            return None

        iterations = 0
        max_iterations = 200
        while iterations < max_iterations:
            if time.time() > deadline - 30:
                break
            c6 = find_c6(current_adj)
            if c6 is None:
                break
            idx = random.randint(0, 5)
            u, v = c6[idx]
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            iterations += 1
            
        current_count = count_edges(current_adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]

    if best_adj is None:
        best_adj = [0] * n
        best_count = 0

    # Phase 2: SA
    current_adj = best_adj[:]
    current_count = best_count
    
    def get_edge_lists(adj):
        existing = []
        non_existing = []
        for u, v in all_pairs:
            if (adj[u] >> v) & 1:
                existing.append((u, v))
            else:
                non_existing.append((u, v))
        return existing, non_existing

    existing_edges_list, non_existing_edges_list = get_edge_lists(current_adj)
    
    # Greedy Add
    random.shuffle(non_existing_edges_list)
    for u, v in non_existing_edges_list:
        if time.time() > deadline - 10:
            break
        if not ((current_adj[u] >> v) & 1):
            if can_add_edge(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_count += 1
                non_existing_edges_list.remove((u, v))
                existing_edges_list.append((u, v))
                
    if current_count > best_count:
        best_count = current_count
        best_edges_list = edges_from_adj(current_adj)
    else:
        best_edges_list = edges_from_adj(current_adj)

    # SA
    T = 10.0
    cooling_rate = 0.99
    min_T = 0.1
    
    while time.time() < deadline - 2:
        move_type = random.random()
        
        if move_type < 0.5:
            if not non_existing_edges_list:
                continue
            idx = random.randint(0, len(non_existing_edges_list) - 1)
            u, v = non_existing_edges_list[idx]
            
            if can_add_edge(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_count += 1
                
                non_existing_edges_list.pop(idx)
                existing_edges_list.append((u, v))
                
                if current_count > best_count:
                    best_count = current_count
                    best_edges_list = edges_from_adj(current_adj)
                    if best_count >= 105:
                        break
        else:
            if not existing_edges_list or not non_existing_edges_list:
                continue
            
            idx_rem = random.randint(0, len(existing_edges_list) - 1)
            u1, v1 = existing_edges_list[idx_rem]
            
            idx_add = random.randint(0, len(non_existing_edges_list) - 1)
            u2, v2 = non_existing_edges_list[idx_add]
            
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            
            if can_add_edge(current_adj, u2, v2):
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                
                existing_edges_list.pop(idx_rem)
                non_existing_edges_list.append((u1, v1))
                
                non_existing_edges_list.pop(idx_add)
                existing_edges_list.append((u2, v2))
            else:
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                
        T *= cooling_rate
        if T < min_T:
            T = min_T

    if not best_edges_list:
        best_edges_list = edges_from_adj(current_adj)

    return best_edges_list
