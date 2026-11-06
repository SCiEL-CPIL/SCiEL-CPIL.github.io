import time
import random

def construct():
    d = 7
    n_vertices = 1 << d
    
    # Precompute all edges and their witnesses
    # Edge is represented by (u, v) with u < v
    all_edges = []
    edge_to_idx = {}
    idx_to_edge = {}
    
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(all_edges)
                all_edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                idx_to_edge[idx] = (u, v)
                
    num_edges = len(all_edges)
    
    # Precompute witnesses for each edge
    # witnesses[i] is a list of lists of 3 edge indices
    # witnesses[i][j] corresponds to bit j != bit_i
    # The 3 edges are (u, u^e_j), (v, v^e_j), (u^e_j, v^e_j)
    witnesses = [[] for _ in range(num_edges)]
    
    for idx, (u, v) in enumerate(all_edges):
        # Find bit i
        diff = u ^ v
        i = diff.bit_length() - 1
        
        for j in range(d):
            if j == i:
                continue
            e_j = 1 << j
            u_j = u ^ e_j
            v_j = v ^ e_j
            
            # Edge 1: (u, u_j)
            e1 = (min(u, u_j), max(u, u_j))
            # Edge 2: (v, v_j)
            e2 = (min(v, v_j), max(v, v_j))
            # Edge 3: (u_j, v_j)
            e3 = (min(u_j, v_j), max(u_j, v_j))
            
            w1 = edge_to_idx[e1]
            w2 = edge_to_idx[e2]
            w3 = edge_to_idx[e3]
            
            witnesses[idx].append([w1, w2, w3])

    def is_valid(S, e_idx):
        """Check if adding edge e_idx to set S of edge indices creates a C4."""
        for w in witnesses[e_idx]:
            if w[0] in S and w[1] in S and w[2] in S:
                return False
        return True

    best_set = set()
    best_size = 0
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    while time.time() < deadline:
        # Randomized Greedy
        S = set()
        edges_list = list(range(num_edges))
        random.shuffle(edges_list)
        
        for e_idx in edges_list:
            if is_valid(S, e_idx):
                S.add(e_idx)
        
        if len(S) > best_size:
            best_set = S.copy()
            best_size = len(S)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            random.shuffle(S_list)
            
            for e_remove in S_list:
                if time.time() > deadline:
                    break
                
                S_new = S.copy()
                S_new.discard(e_remove)
                
                # Try to add any valid edge
                # We can just iterate through all edges not in S_new
                # To be faster, we could track which edges were blocked by e_remove,
                # but for now, full scan of non-edges is O(150 * 6) which is fine.
                
                added_any = False
                # We want to maximize size, so we add as many as possible?
                # Or just one?
                # Greedy addition:
                candidates = [e for e in range(num_edges) if e not in S_new]
                random.shuffle(candidates)
                
                for e_add in candidates:
                    if is_valid(S_new, e_add):
                        S_new.add(e_add)
                        added_any = True
                
                if len(S_new) > len(S):
                    S = S_new
                    improved = True
                    break
                elif len(S_new) == len(S) and added_any and random.random() < 0.2:
                    # Sideways move to escape local minima
                    S = S_new
                    improved = True
                    break
            
            if len(S) > best_size:
                best_set = S.copy()
                best_size = len(S)

    # Convert best_set (indices) back to edges [u, v]
    result = []
    for idx in best_set:
        u, v = idx_to_edge[idx]
        result.append([u, v])
        
    return result
