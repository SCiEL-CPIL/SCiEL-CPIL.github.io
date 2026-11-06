import time
import random
import numpy as np

def _compute_max_book_bitmask(adj_list, n=40):
    """Compute max book size using bitmasks. adj_list[i] is a bitmask of neighbors of i."""
    max_book = 0
    for i in range(n):
        mask_i = adj_list[i]
        # Iterate over neighbors of i
        m = mask_i
        while m:
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            m ^= lsb
            if j > i:  # only count each edge once
                common = bin(adj_list[i] & adj_list[j]).count('1')
                if common > max_book:
                    max_book = common
    return max_book

def _build_adj_list(adj_mask, n=40):
    """Build bitmask adjacency list from boolean adjacency matrix."""
    adj_list = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if adj_mask[i, j]:
                adj_list[i] |= (1 << j)
                adj_list[j] |= (1 << i)
    return adj_list

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    best_max = 999
    best_adj = None
    
    random.seed(12345)
    
    configurations = []
    
    # Config 1: K20,20 + 1 edge (baseline, book=20)
    adj1 = np.zeros((n, n), dtype=bool)
    for a in range(20):
        for b in range(20, 40):
            adj1[a, b] = True
            adj1[b, a] = True
    adj1[0, 1] = True
    adj1[1, 0] = True
    configurations.append(adj1.copy())
    
    # Config 2: Split into two groups of 20, but with a more balanced intra-group structure
    for seed_val in [1, 2, 3, 4, 5]:
        random.seed(seed_val)
        adj = np.zeros((n, n), dtype=bool)
        for a in range(20):
            for b in range(20, 40):
                adj[a, b] = True
                adj[b, a] = True
        possible_intra = [(a, b) for a in range(20) for b in range(a+1, 20)]
        possible_intra += [(a, b) for a in range(20, 40) for b in range(a+1, 40)]
        u, v = random.choice(possible_intra)
        adj[u, v] = True
        adj[v, u] = True
        configurations.append(adj.copy())
    
    # Config 3: Try a 3-partite structure: 13, 13, 14
    sizes = [13, 13, 14]
    adj3 = np.zeros((n, n), dtype=bool)
    part_starts = [0, 13, 26]
    for pi in range(3):
        for pj in range(pi+1, 3):
            for i in range(part_starts[pi], part_starts[pi]+sizes[pi]):
                for j in range(part_starts[pj], part_starts[pj]+sizes[pj]):
                    adj3[i, j] = True
                    adj3[j, i] = True
    edge_count = np.sum(adj3) // 2
    random.seed(77)
    intra_candidates = []
    for pi in range(3):
        start = part_starts[pi]
        end = start + sizes[pi]
        for i in range(start, end):
            for j in range(i+1, end):
                intra_candidates.append((i, j))
    random.shuffle(intra_candidates)
    for u, v in intra_candidates:
        if edge_count >= target_edges:
            break
        adj3[u, v] = True
        adj3[v, u] = True
        edge_count += 1
    if edge_count == target_edges:
        configurations.append(adj3.copy())
    
    # Config 4: Regular-ish graph with degree ~20 (circulant)
    adj4 = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for k in range(1, 11):
            adj4[i, (i+k) % n] = True
            adj4[(i+k) % n, i] = True
    non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not adj4[i, j]]
    if non_edges:
        u, v = non_edges[0]
        adj4[u, v] = True
        adj4[v, u] = True
    configurations.append(adj4.copy())
    
    # Hill climb each configuration using bitmask-based book computation
    for init_adj in configurations:
        adj = init_adj.copy()
        adj_list = _build_adj_list(adj, n)
        current_max = _compute_max_book_bitmask(adj_list, n)
        
        if current_max < best_max:
            best_max = current_max
            best_adj = adj.copy()
        
        no_improve = 0
        max_no_improve = 20000
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Get all edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]:
                        edges.append((i, j))
            
            if len(edges) != target_edges:
                break
            
            # Pick random edge to remove
            rem_u, rem_v = random.choice(edges)
            
            # Try several candidate new edges
            improved = False
            for attempt in range(5):
                new_u = random.randint(0, n-1)
                new_v = random.randint(0, n-1)
                if new_u == new_v:
                    continue
                if new_u > new_v:
                    new_u, new_v = new_v, new_u
                if new_u == rem_u and new_v == rem_v:
                    continue
                if adj[new_u, new_v]:
                    continue
                
                # Tentative swap
                adj[rem_u, rem_v] = False
                adj[rem_v, rem_u] = False
                adj[new_u, new_v] = True
                adj[new_v, new_u] = True
                adj_list[rem_u] &= ~(1 << rem_v)
                adj_list[rem_v] &= ~(1 << rem_u)
                adj_list[new_u] |= (1 << new_v)
                adj_list[new_v] |= (1 << new_u)
                
                new_max = _compute_max_book_bitmask(adj_list, n)
                
                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_adj = adj.copy()
                    no_improve = 0
                    improved = True
                    break
                else:
                    # Revert
                    adj[new_u, new_v] = False
                    adj[new_v, new_u] = False
                    adj[rem_u, rem_v] = True
                    adj[rem_v, rem_u] = True
                    adj_list[rem_u] |= (1 << rem_v)
                    adj_list[rem_v] |= (1 << rem_u)
                    adj_list[new_u] &= ~(1 << new_v)
                    adj_list[new_v] &= ~(1 << new_u)
            
            if not improved:
                no_improve += 1
            
            if current_max <= 6:
                break
        
        if time.time() > deadline:
            break
    
    if best_adj is None:
        adj = np.zeros((n, n), dtype=bool)
        for a in range(20):
            for b in range(20, 40):
                adj[a, b] = True
                adj[b, a] = True
        adj[0, 1] = True
        adj[1, 0] = True
        best_adj = adj
    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                result.append([i, j])
    
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    
    return result
# EVOLVE-BLOCK-END
