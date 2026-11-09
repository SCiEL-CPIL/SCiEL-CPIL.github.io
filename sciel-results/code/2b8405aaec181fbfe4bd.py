import time
import random
import sys
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

    n = 16
    num_edges_total = n * (n - 1) // 2  # 120

    # Map edge (u, v) to an index 0..119
    def edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    # Precompute edge indices for all pairs
    edge_idx_map = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = edge_index(u, v)

    # Inverse map: index -> (u, v)
    inv_edge_map = [None] * num_edges_total
    for (u, v), idx in edge_idx_map.items():
        inv_edge_map[idx] = (u, v)

    # Q3 abstract edges (standard labeling of 3-cube)
    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    # Local edge index mapping for K8 (28 edges)
    local_edge_to_idx = {}
    idx = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = idx
            idx += 1

    # Precompute local Q3 edge sets as bitmasks
    local_q3_bitmasks = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_to_idx[(min(u_val, v_val), max(u_val, v_val))]
            edges_set.add(le)
        if len(edges_set) == 12:
            mask_int = 0
            for e in edges_set:
                mask_int |= (1 << e)
            local_q3_bitmasks.add(mask_int)
    
    local_q3_list = list(local_q3_bitmasks)

    # Precompute global Q3 masks and mapping to edges
    global_q3_masks = []
    q3s_for_edge = [[] for _ in range(num_edges_total)]
    
    all_subsets = list(combinations(range(n), 8))
    
    for subset in all_subsets:
        l2g = [0] * 28
        for u_loc in range(8):
            for v_loc in range(u_loc + 1, 8):
                le = local_edge_to_idx[(u_loc, v_loc)]
                u_glob = subset[u_loc]
                v_glob = subset[v_loc]
                ge = edge_idx_map[(min(u_glob, v_glob), max(u_glob, v_glob))]
                l2g[le] = ge
                
        for l_mask in local_q3_list:
            g_mask = 0
            m = l_mask
            while m:
                lsb = m & -m
                le = lsb.bit_length() - 1
                ge = l2g[le]
                g_mask |= (1 << ge)
                m ^= lsb
                
            idx_g = len(global_q3_masks)
            global_q3_masks.append(g_mask)
            
            m = g_mask
            while m:
                lsb = m & -m
                e_idx = lsb.bit_length() - 1
                q3s_for_edge[e_idx].append(idx_g)
                m ^= lsb

    if time.time() > deadline - 5:
        return [[i, i+1] for i in range(0, 16, 2)]

    best_edge_count = 0
    best_g_mask = 0
    
    while time.time() < deadline - 10:
        g_mask = 0
        edges = list(range(num_edges_total))
        random.shuffle(edges)
        
        for e in edges:
            e_bit = 1 << e
            can_add = True
            for q_idx in q3s_for_edge[e]:
                q_mask = global_q3_masks[q_idx]
                missing = q_mask & ~g_mask
                if missing == e_bit:
                    can_add = False
                    break
            
            if can_add:
                g_mask |= e_bit
        
        count = bin(g_mask).count('1')
        if count > best_edge_count:
            best_edge_count = count
            best_g_mask = g_mask

    final_edges = []
    for e in range(num_edges_total):
        if (best_g_mask >> e) & 1:
            u, v = inv_edge_map[e]
            final_edges.append([u, v])
            
    return final_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
