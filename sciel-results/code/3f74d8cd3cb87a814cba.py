import time
import random
from itertools import combinations

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
    
    # Precompute Q3 edge masks in K8
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
    num_q3_masks = len(q3_masks_list)
    
    # Precompute for each 8-subset: mapping from local edge bit to global edge index
    # Use arrays for fast lookup
    subset_list = list(combinations(range(n), 8))
    num_subsets = len(subset_list)
    
    # For each subset, store:
    # - subset_vertices: the 8 vertices
    # - local_to_global: array of 28 global edge indices (indexed by local edge bit)
    subset_data = []
    for subset in subset_list:
        local_to_global = [0] * 28
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                ge_idx = edge_index[(u,v)]
                le_bit = local_edge_to_bit[(i,j)]
                local_to_global[le_bit] = ge_idx
        subset_data.append(local_to_global)
    
    # Precompute: for each global edge, which subsets contain it
    # edge_to_subsets[edge_idx] = list of (subset_idx, local_bit)
    edge_to_subsets = [[] for _ in range(num_edges)]
    for si, ltg in enumerate(subset_data):
        for le_bit in range(28):
            ge_idx = ltg[le_bit]
            edge_to_subsets[ge_idx].append((si, le_bit))
    
    def find_violating_q3(mask):
        """Find a Q3 in the graph. Returns list of global edge indices or None."""
        for si in range(num_subsets):
            ltg = subset_data[si]
            local_mask = 0
            for le_bit in range(28):
                ge_idx = ltg[le_bit]
                if (mask >> ge_idx) & 1:
                    local_mask |= (1 << le_bit)
            for qm in q3_masks_list:
                if (local_mask & qm) == qm:
                    # Found violation - return the edges in this Q3
                    edges = []
                    for le_bit in range(28):
                        if (qm >> le_bit) & 1:
                            edges.append(ltg[le_bit])
                    return edges
        return None
    
    def is_q3_free(mask):
        return find_violating_q3(mask) is None
    
    # Greedy initialization: start with K16, remove edges to eliminate Q3
    mask = (1 << num_edges) - 1  # All edges present
    edge_count = num_edges
    
    # Greedy removal: while Q3 exists, remove a random edge from it
    max_removals = 200
    for _ in range(max_removals):
        if time.time() > deadline - 30:
            break
        viol = find_violating_q3(mask)
        if viol is None:
            break
        # Remove a random edge from the violation
        rem_idx = random.choice(viol)
        mask &= ~(1 << rem_idx)
        edge_count -= 1
    
    best_mask = mask
    best_count = edge_count
    
    random.seed(42)
    
    # Simulated annealing / local search
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    trial = 0
    while time.time() < deadline - 5:
        trial += 1
        
        # Occasionally restart from a perturbed version of best
        if trial % 50 == 0 or trial == 1:
            if trial > 1:
                # Perturb best: remove some random edges
                mask = best_mask
                edge_count = best_count
                for _ in range(random.randint(3, 10)):
                    if edge_count <= best_count:
                        break
                    # Pick a random edge that's present
                    present = [i for i in range(num_edges) if (mask >> i) & 1]
                    if not present:
                        break
                    rem = random.choice(present)
                    mask &= ~(1 << rem)
                    edge_count -= 1
                # Repair if needed
                for _ in range(50):
                    viol = find_violating_q3(mask)
                    if viol is None:
                        break
                    rem_idx = random.choice(viol)
                    mask &= ~(1 << rem_idx)
                    edge_count -= 1
                if find_violating_q3(mask) is not None:
                    # Failed, skip
                    continue
            else:
                # First trial: use the greedy result
                pass
        else:
            # Continue from current state
            pass
        
        # Local search: try adding edges
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try to add random missing edges
            missing = [i for i in range(num_edges) if not (mask >> i) & 1]
            random.shuffle(missing)
            for idx in missing:
                if time.time() > deadline - 5:
                    break
                test_mask = mask | (1 << idx)
                if find_violating_q3(test_mask) is None:
                    mask = test_mask
                    edge_count += 1
                    improved = True
                    if edge_count > best_count:
                        best_count = edge_count
                        best_mask = mask
                    T = max(T_min, T * cooling)
                else:
                    # Could accept with probability exp(-delta/T) but delta=1 here
                    # For now, just skip
                    pass
        
        # Occasionally do a swap: remove one edge, try to add two
        if random.random() < 0.3 and time.time() < deadline - 5:
            present = [i for i in range(num_edges) if (mask >> i) & 1]
            missing = [i for i in range(num_edges) if not (mask >> i) & 1]
            if present and len(missing) >= 2:
                rem = random.choice(present)
                test_mask = mask & ~(1 << rem)
                # Try to add back 2 edges
                added = 0
                random.shuffle(missing)
                for add in missing:
                    if added >= 2:
                        break
                    test2 = test_mask | (1 << add)
                    if find_violating_q3(test2) is None:
                        test_mask = test2
                        added += 1
                if added >= 1:
                    new_count = bin(test_mask).count('1')
                    if new_count > edge_count:
                        mask = test_mask
                        edge_count = new_count
                        if edge_count > best_count:
                            best_count = edge_count
                            best_mask = mask
                    elif random.random() < min(1.0, T / max(1, edge_count - new_count)):
                        mask = test_mask
                        edge_count = new_count
                T = max(T_min, T * cooling)
        
        T = max(T_min, T * cooling)
    
    result = []
    for i in range(num_edges):
        if (best_mask >> i) & 1:
            u, v = all_edges[i]
            result.append([u, v])
            
    return result
