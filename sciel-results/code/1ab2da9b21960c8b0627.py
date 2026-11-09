import time
import sys
from itertools import combinations, permutations
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    num_edges_total = n * (n - 1) // 2

    def edge_index(u, v):
        if u > v:
            u, v = v, u
        idx = 0
        for i in range(u):
            idx += (n - 1 - i)
        idx += (v - u - 1)
        return idx

    edge_idx_map = {}
    inv_edge_map = [None] * num_edges_total
    for u in range(n):
        for v in range(u + 1, n):
            idx = edge_index(u, v)
            edge_idx_map[(u, v)] = idx
            inv_edge_map[idx] = (u, v)

    q3_abstract_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]

    local_edge_to_idx = {}
    local_idx_to_edge = [None] * 28
    idx = 0
    for u in range(8):
        for v in range(u + 1, 8):
            local_edge_to_idx[(u, v)] = idx
            local_idx_to_edge[idx] = (u, v)
            idx += 1

    local_q3_edge_sets = set()
    for p in permutations(range(8)):
        edges_set = set()
        for u_abs, v_abs in q3_abstract_edges:
            u_val = p[u_abs]
            v_val = p[v_abs]
            le = local_edge_to_idx[(min(u_val, v_val), max(u_val, v_val))]
            edges_set.add(le)
        if len(edges_set) == 12:
            local_q3_edge_sets.add(frozenset(edges_set))
            
    local_q3_bitmasks = []
    for mask_set in local_q3_edge_sets:
        mask_int = 0
        for e in mask_set:
            mask_int |= (1 << e)
        local_q3_bitmasks.append(mask_int)
        
    local_q3_edge_lists = []
    for mask in local_q3_bitmasks:
        edges = []
        m = mask
        while m:
            lsb = m & -m
            edges.append(lsb.bit_length() - 1)
            m ^= lsb
        local_q3_edge_lists.append(edges)

    global_q3_masks = []
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
                
        for edge_list in local_q3_edge_lists:
            global_mask = 0
            for le in edge_list:
                ge = l2g[le]
                global_mask |= (1 << ge)
            global_q3_masks.append(global_mask)
            
    if time.time() > deadline - 10:
        return [[i, i+1] for i in range(0, 16, 2)]

    num_q3s = len(global_q3_masks)
    
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    counts = [0] * num_edges_total
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    active_q3s = [True] * num_q3s
    removed_edges = [False] * num_edges_total
    
    heap = [(-counts[e], e) for e in range(num_edges_total)]
    heapq.heapify(heap)
    
    while True:
        if time.time() > deadline - 15:
            break
            
        best_edge = -1
        max_count = -1
        
        while heap:
            neg_cnt, e = heapq.heappop(heap)
            if removed_edges[e]:
                continue
            if -neg_cnt != counts[e]:
                continue
            best_edge = e
            max_count = -neg_cnt
            break
            
        if best_edge == -1 or max_count == 0:
            break
            
        removed_edges[best_edge] = True
        
        for q3_idx in q3s_containing_edge[best_edge]:
            if active_q3s[q3_idx]:
                active_q3s[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                        heapq.heappush(heap, (-counts[e_idx], e_idx))
                    m ^= lsb

    removed_count = [0] * num_q3s
    for i, mask in enumerate(global_q3_masks):
        m = mask
        cnt = 0
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            if removed_edges[e_idx]:
                cnt += 1
            m ^= lsb
        removed_count[i] = cnt
        
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        for e in range(num_edges_total):
            if not removed_edges[e]:
                continue
            
            can_readd = True
            for q3_idx in q3s_containing_edge[e]:
                if removed_count[q3_idx] < 2:
                    can_readd = False
                    break
                    
            if can_readd:
                removed_edges[e] = False
                improved = True
                for q3_idx in q3s_containing_edge[e]:
                    removed_count[q3_idx] -= 1
                break
                
        if improved:
            continue
            
        for e_out in range(num_edges_total):
            if not removed_edges[e_out]:
                continue
                
            found_swap = False
            for e_in in range(num_edges_total):
                if removed_edges[e_in]:
                    continue
                    
                valid = True
                for q3_idx in q3s_containing_edge[e_out]:
                    mask = global_q3_masks[q3_idx]
                    if (mask >> e_in) & 1:
                        continue
                    if removed_count[q3_idx] < 2:
                        valid = False
                        break
                        
                if valid:
                    removed_edges[e_out] = False
                    removed_edges[e_in] = True
                    improved = True
                    found_swap = True
                    
                    for q3_idx in q3s_containing_edge[e_out]:
                        removed_count[q3_idx] -= 1
                    for q3_idx in q3s_containing_edge[e_in]:
                        removed_count[q3_idx] += 1
                    break
                    
            if found_swap:
                break
                
    final_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            final_edges.append([u, v])
            
    return final_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
