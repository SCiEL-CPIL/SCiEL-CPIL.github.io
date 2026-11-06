import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    
    n = 16
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # 1. Precompute Q3 masks in K8 (28-bit)
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
            
    from itertools import permutations
    q3_masks = set()
    perm_list = list(permutations(range(8)))
    for p in perm_list:
        mask = 0
        for u, v in q3_edges_local:
            a, b = p[u], p[v]
            if a > b: a, b = b, a
            mask |= (1 << local_edge_to_bit[(a,b)])
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    
    # 2. Precompute has_q3 table (size 2^28)
    # This might take ~30s
    if time.time() < deadline - 40:
        has_q3 = np.zeros(2**28, dtype=np.uint8)
        full_mask = (1 << 28) - 1
        for qm in q3_masks_list:
            comp = full_mask ^ qm
            sub = comp
            while True:
                L = qm | sub
                has_q3[L] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & comp
    else:
        # Fallback if precomputation is too slow (unlikely)
        has_q3 = None

    # 3. Precompute subset edge mappings
    # For each 8-subset, store the 28 global edge indices
    subset_edge_maps = {}
    for subset in combinations(range(n), 8):
        mapping = []
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u,v)]
                mapping.append(ge_idx)
        subset_edge_maps[subset] = mapping
        
    # Also precompute for each edge, the list of subsets containing it
    # edge_idx -> list of subsets
    edge_to_subsets = {i: [] for i in range(num_edges)}
    for subset, mapping in subset_edge_maps.items():
        for ge_idx in mapping:
            edge_to_subsets[ge_idx].append(subset)

    def get_local_mask(mask, subset):
        mapping = subset_edge_maps[subset]
        local_mask = 0
        for k, ge_idx in enumerate(mapping):
            if (mask >> ge_idx) & 1:
                local_mask |= (1 << k)
        return local_mask

    def is_q3_free(mask, subsets_to_check=None):
        if subsets_to_check is None:
            subsets_to_check = subset_edge_maps.keys()
            
        if has_q3 is not None:
            for subset in subsets_to_check:
                lm = get_local_mask(mask, subset)
                if has_q3[lm]:
                    return False
            return True
        else:
            # Fallback slow check
            for subset in subsets_to_check:
                lm = get_local_mask(mask, subset)
                for qm in q3_masks_list:
                    if (lm & qm) == qm:
                        return False
            return True

    def find_violation(mask, subsets_to_check=None):
        if subsets_to_check is None:
            subsets_to_check = subset_edge_maps.keys()
            
        for subset in subsets_to_check:
            lm = get_local_mask(mask, subset)
            if has_q3 is not None:
                if has_q3[lm]:
                    # Find a specific Q3 mask contained in lm
                    for qm in q3_masks_list:
                        if (lm & qm) == qm:
                            # Return global edges of this Q3
                            edges = []
                            mapping = subset_edge_maps[subset]
                            for k in range(28):
                                if (qm >> k) & 1:
                                    edges.append(mapping[k])
                            return edges
            else:
                for qm in q3_masks_list:
                    if (lm & qm) == qm:
                        edges = []
                        mapping = subset_edge_maps[subset]
                        for k in range(28):
                            if (qm >> k) & 1:
                                edges.append(mapping[k])
                        return edges
        return None

    # Initial best: K3,13
    best_mask = 0
    for u in range(3):
        for v in range(3, 16):
            idx = edge_index[(u,v)]
            best_mask |= (1 << idx)
            
    best_count = bin(best_mask).count('1')
    
    random.seed(42)
    
    # Local search
    for trial in range(100):
        if time.time() > deadline - 5:
            break
            
        # Start with random dense graph
        p = random.uniform(0.6, 0.8)
        current_mask = 0
        for i in range(num_edges):
            if random.random() < p:
                current_mask |= (1 << i)
                
        if bin(current_mask).count('1') <= best_count:
            continue
            
        # Repair
        for _ in range(50):
            viol = find_violation(current_mask)
            if viol is None:
                break
            rem_idx = random.choice(viol)
            current_mask &= ~(1 << rem_idx)
            
        if find_violation(current_mask) is not None:
            continue
            
        # Improve
        missing_edges = [i for i in range(num_edges) if not (current_mask >> i) & 1]
        random.shuffle(missing_edges)
        
        improved = True
        while improved:
            improved = False
            random.shuffle(missing_edges)
            for idx in missing_edges:
                if time.time() > deadline - 5:
                    break
                test_mask = current_mask | (1 << idx)
                # Only check subsets containing this edge
                subs = edge_to_subsets[idx]
                if is_q3_free(test_mask, subs):
                    current_mask = test_mask
                    missing_edges.remove(idx)
                    improved = True
                    cnt = bin(current_mask).count('1')
                    if cnt > best_count:
                        best_count = cnt
                        best_mask = current_mask

    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result
