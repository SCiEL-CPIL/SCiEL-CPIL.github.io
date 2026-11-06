import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Precompute Q3 masks in K8
    q3_edges_local = set()
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_edges_local.add((i, j))
                
    local_edge_to_bit = {}
    k8_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i,j)] = len(k8_edges)
            k8_edges.append((i,j))
            
    q3_masks = set()
    for p in permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            a, b = p[u], p[v]
            if a > b: a, b = b, a
            mask |= (1 << local_edge_to_bit[(a,b)])
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    
    # Precompute subsets
    subsets = list(combinations(range(n), 8))
    num_subsets = len(subsets)
    
    # local_bit_for_edge[s_idx][ge_idx] = le_bit or -1
    local_bit_for_edge = [[-1]*num_edges for _ in range(num_subsets)]
    subsets_with_edge = [[] for _ in range(num_edges)]
    
    for s_idx, subset in enumerate(subsets):
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u,v)]
                le_bit = local_edge_to_bit[(i,j)]
                local_bit_for_edge[s_idx][ge_idx] = le_bit
                subsets_with_edge[ge_idx].append(s_idx)
                
    # Precompute Q3 masks by bit for faster checking
    q3_masks_by_bit = [[] for _ in range(28)]
    for qm in q3_masks_list:
        for bit in range(28):
            if (qm >> bit) & 1:
                q3_masks_by_bit[bit].append(qm)

    best_mask = 0
    best_count = 0
    
    random.seed(42)
    
    for trial in range(200):
        if time.time() > deadline:
            break
            
        current_mask = 0
        for i in range(num_edges):
            if random.random() < 0.7:
                current_mask |= (1 << i)
                
        local_masks = [0] * num_subsets
        bad_counts = [0] * num_subsets
        
        for s_idx in range(num_subsets):
            lm = 0
            for ge_idx in range(num_edges):
                le_bit = local_bit_for_edge[s_idx][ge_idx]
                if le_bit != -1 and (current_mask >> ge_idx) & 1:
                    lm |= (1 << le_bit)
            local_masks[s_idx] = lm
            bc = 0
            for qm in q3_masks_list:
                if (lm & qm) == qm:
                    bc += 1
            bad_counts[s_idx] = bc
            
        # Repair
        for _ in range(100):
            if time.time() > deadline:
                break
            bad_idx = -1
            for i in range(num_subsets):
                if bad_counts[i] > 0:
                    bad_idx = i
                    break
            if bad_idx == -1:
                break
                
            lm = local_masks[bad_idx]
            present_edges = []
            for ge_idx in range(num_edges):
                le_bit = local_bit_for_edge[bad_idx][ge_idx]
                if le_bit != -1 and (lm >> le_bit) & 1:
                    present_edges.append((le_bit, ge_idx))
                    
            if not present_edges:
                break
                
            le_bit, ge_idx = random.choice(present_edges)
            current_mask &= ~(1 << ge_idx)
            
            for s_idx in subsets_with_edge[ge_idx]:
                lb = local_bit_for_edge[s_idx][ge_idx]
                if lb != -1:
                    local_masks[s_idx] &= ~(1 << lb)
                    lm = local_masks[s_idx]
                    bc = 0
                    for qm in q3_masks_list:
                        if (lm & qm) == qm:
                            bc += 1
                    bad_counts[s_idx] = bc
                    
        if any(bc > 0 for bc in bad_counts):
            continue
            
        # Improve
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
            random.shuffle(missing_edges)
            
            for ge_idx in missing_edges:
                if time.time() > deadline:
                    break
                    
                can_add = True
                for s_idx in subsets_with_edge[ge_idx]:
                    le_bit = local_bit_for_edge[s_idx][ge_idx]
                    if le_bit == -1:
                        continue
                    lm = local_masks[s_idx]
                    # Check if adding le_bit to lm creates a Q3
                    # Equivalent to: exists qm in q3_masks_by_bit[le_bit] such that (qm & ~bit) == (qm & lm)
                    for qm in q3_masks_by_bit[le_bit]:
                        if (qm & ~(1 << le_bit)) == (qm & lm):
                            can_add = False
                            break
                    if not can_add:
                        break
                        
                if can_add:
                    current_mask |= (1 << ge_idx)
                    improved = True
                    for s_idx in subsets_with_edge[ge_idx]:
                        le_bit = local_bit_for_edge[s_idx][ge_idx]
                        if le_bit != -1:
                            local_masks[s_idx] |= (1 << le_bit)
                            
        count = bin(current_mask).count('1')
        if count > best_count:
            best_count = count
            best_mask = current_mask

    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result
