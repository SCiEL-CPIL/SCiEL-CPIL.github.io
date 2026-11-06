import random
import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n = 16
    core_n = 11
    new_vertices = list(range(core_n, n))
    num_new = len(new_vertices)

    # 1. Build Grötzsch Graph (Vertices 0..10)
    # C5: 0-1-2-3-4-0
    # Mycielski:
    # Copies: 5..9 (5 is copy of 0, ..., 9 is copy of 4)
    # Top: 10
    core_edges = []
    c5_edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in c5_edges:
        core_edges.append([u, v])
    
    for u, v in c5_edges:
        core_edges.append([u, 5+v])
        core_edges.append([v, 5+u])
        
    for i in range(5):
        core_edges.append([10, 5+i])
        
    # Build adjacency bitmask for core
    adj_core = [0] * core_n
    for u, v in core_edges:
        adj_core[u] |= (1 << v)
        adj_core[v] |= (1 << u)
        
    # 2. Find all independent sets in the core
    independent_sets = []
    for mask in range(1 << core_n):
        is_ind = True
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            if adj_core[u] & mask:
                is_ind = False
                break
            m ^= lsb
        if is_ind:
            independent_sets.append(mask)
            
    # Precompute popcount for all masks up to 2^11
    popcount = [0] * (1 << core_n)
    for i in range(1, 1 << core_n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Precompute pairs for new vertices
    pairs = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    num_pairs = len(pairs)
    
    # Precompute which new-edge masks are triangle-free
    valid_new_masks = []
    for mask in range(1 << num_pairs):
        adj_new = [0] * num_new
        for k, (i, j) in enumerate(pairs):
            if mask & (1 << k):
                adj_new[i] |= (1 << j)
                adj_new[j] |= (1 << i)
        is_tf = True
        for u in range(num_new):
            for v in range(u+1, num_new):
                if adj_new[u] & (1 << v):
                    if adj_new[u] & adj_new[v]:
                        is_tf = False
                        break
            if not is_tf:
                break
        if is_tf:
            valid_new_masks.append(mask)
    
    # Precompute popcount for new edge masks
    new_popcount = [0] * (1 << num_pairs)
    for i in range(1, 1 << num_pairs):
        new_popcount[i] = new_popcount[i >> 1] + (i & 1)

    def evaluate_state(sets, new_edges_mask):
        # sets: list of 5 bitmasks
        # new_edges_mask: 10-bit integer
        # Check triangle-free in new vertices (precomputed)
        if new_edges_mask not in valid_new_masks_set:
            return -1
        
        score = len(core_edges)
        for s in sets:
            score += popcount[s]
        score += new_popcount[new_edges_mask]
        
        # Check disjointness for edges
        for k, (i, j) in enumerate(pairs):
            if new_edges_mask & (1 << k):
                if sets[i] & sets[j]:
                    return -1 # Triangle via core
                    
        return score

    # Simulated Annealing
    best_score = -1
    best_state = None
    
    rng = random.Random(42)
    
    # Convert valid_new_masks to set for O(1) lookup
    valid_new_masks_set = set(valid_new_masks)
    
    # Greedy initialization
    def greedy_init(seed):
        rng_local = random.Random(seed)
        sets = [0] * num_new
        new_edges_mask = 0
        
        # Add edges between new vertices greedily
        for k in range(num_pairs):
            candidate = new_edges_mask | (1 << k)
            if candidate in valid_new_masks_set:
                new_edges_mask = candidate
        
        # Greedily add core vertices to each new vertex's set
        for i in range(num_new):
            for core_v in range(core_n):
                bit = 1 << core_v
                if sets[i] & bit:
                    continue
                if adj_core[core_v] & sets[i]:
                    continue
                conflict = False
                for k, (a, b) in enumerate(pairs):
                    if new_edges_mask & (1 << k):
                        if a == i or b == i:
                            other = b if a == i else a
                            if sets[other] & bit:
                                conflict = True
                                break
                if not conflict:
                    sets[i] |= bit
        
        return sets, new_edges_mask

    # Try multiple initializations
    for seed in range(15):
        sets, new_edges_mask = greedy_init(seed)
        score = evaluate_state(sets, new_edges_mask)
        if score > best_score:
            best_score = score
            best_state = (list(sets), new_edges_mask)
    
    # Also try random initializations
    for _ in range(15):
        sets = [rng.choice(independent_sets) for _ in range(num_new)]
        best_local_score = -1
        best_local_mask = 0
        for mask in valid_new_masks:
            s = evaluate_state(sets, mask)
            if s > best_local_score:
                best_local_score = s
                best_local_mask = mask
        if best_local_score > best_score:
            best_score = best_local_score
            best_state = (list(sets), best_local_mask)
    
    if best_state is None:
        return core_edges
    
    current_sets, current_new_edges = best_state
    best_local_score = best_score
    
    T = 50.0
    T_min = 1.0
    alpha = 0.9995
    
    while T > T_min and time.time() < deadline:
        # Perturb
        new_sets = list(current_sets)
        new_new_edges = current_new_edges
        
        # Change one set
        i = rng.randint(0, num_new - 1)
        new_sets[i] = rng.choice(independent_sets)
        
        # Re-optimize new_edges for new_sets
        local_best_score = -1
        local_best_mask = 0
        for mask in valid_new_masks:
            s = evaluate_state(new_sets, mask)
            if s > local_best_score:
                local_best_score = s
                local_best_mask = mask
                
        delta = local_best_score - best_local_score
        
        if delta >= 0 or rng.random() < pow(2.71828, delta / T):
            current_sets = new_sets
            current_new_edges = local_best_mask
            best_local_score = local_best_score
            
            if best_local_score > best_score:
                best_score = best_local_score
                best_state = (list(current_sets), local_best_mask)
                
        T *= alpha

    if best_state is None:
        return core_edges
        
    sets, new_edges_mask = best_state
    
    # Construct final edges
    final_edges = list(core_edges)
    
    for i, v_new in enumerate(new_vertices):
        s = sets[i]
        m = s
        while m:
            lsb = m & -m
            v_core = lsb.bit_length() - 1
            final_edges.append([v_new, v_core])
            m ^= lsb
            
    for k, (i, j) in enumerate(pairs):
        if new_edges_mask & (1 << k):
            u = new_vertices[i]
            v = new_vertices[j]
            final_edges.append([u, v])
            
    # Final Verification
    adj = [0] * n
    for u, v in final_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    is_tf = True
    for u in range(n):
        for v in range(u+1, n):
            if adj[u] & (1 << v):
                if adj[u] & adj[v]:
                    is_tf = False
                    break
        if not is_tf:
            break
            
    if not is_tf:
        return core_edges
        
    # 3-colorability check
    def is_3_colorable():
        degrees = [bin(adj[i]).count("1") for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        color = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = [False] * 3
            neighbors = adj[v]
            m = neighbors
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if color[u] != -1:
                    used[color[u]] = True
                m ^= lsb
                
            for c in range(3):
                if not used[c]:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
            
        return backtrack(0)
        
    if is_3_colorable():
        return core_edges

    return final_edges
