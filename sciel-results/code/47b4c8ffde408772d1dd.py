import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    
    # Precomputation
    # 1. Edge indices
    edge_index = [[-1]*n for _ in range(n)]
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u+1, n):
            edge_index[u][v] = idx
            edge_index[v][u] = idx
            edge_list.append((u, v))
            idx += 1
    total_edges = 120
    
    # 2. Subsets of size 8
    subsets_8 = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets_8)
    
    # 3. Local edge indices for K8
    local_edge_idx = [[-1]*8 for _ in range(8)]
    l_idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_idx[i][j] = l_idx
            local_edge_idx[j][i] = l_idx
            l_idx += 1
            
    # 4. Generate Q3 masks for K8
    q3_canonical_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            d = i ^ j
            if d & (d-1) == 0:
                q3_canonical_edges.append((i, j))
    
    q3_masks_for_k8 = []
    seen_masks = set()
    
    for perm in itertools.permutations(range(8)):
        mask = 0
        for (u, v) in q3_canonical_edges:
            pu = perm[u]
            pv = perm[v]
            e_idx = local_edge_idx[pu][pv]
            mask |= (1 << e_idx)
        if mask not in seen_masks:
            seen_masks.add(mask)
            q3_masks_for_k8.append(mask)
            
    q3_masks_np = np.array(q3_masks_for_k8, dtype=np.uint64)
    
    # 5. Map global edges to subsets and local indices
    edge_to_subsets = [[] for _ in range(total_edges)]
    local_idx = [[-1]*total_edges for _ in range(num_subsets)]
    
    for s_idx, s in enumerate(subsets_8):
        for i in range(8):
            for j in range(i+1, 8):
                u_glob = s[i]
                v_glob = s[j]
                g_idx = edge_index[u_glob][v_glob]
                l_pos = local_edge_idx[i][j]
                
                edge_to_subsets[g_idx].append(s_idx)
                local_idx[s_idx][g_idx] = l_pos

    # Helper to check if a 28-bit mask contains a Q3
    def contains_q3(mask):
        if mask.bit_count() < 12:
            return False
        for q in q3_masks_for_k8:
            if (mask & q) == q:
                return True
        return False

    # Vectorized Q3 check using numpy
    # subset_masks_np: shape (num_subsets,), each element is a 28-bit mask
    def check_all_q3_free(subset_masks_np):
        """Returns True if no subset contains a Q3"""
        if len(subset_masks_np) == 0:
            return True
        # For each subset, check if it contains any Q3
        # We can do this by checking (mask & q) == q for all q
        # This is expensive to do vectorized in pure numpy without loops
        # But we can batch it: for each q, check which subsets contain it
        # Actually, let's just do a loop over subsets but use the precomputed masks
        # Given num_subsets ~ 12870, and 840 Q3 masks, this is 12870*840 = ~10M ops
        # That might be too slow per step.
        
        # Alternative: Only check subsets that have changed
        # But for now, let's try a different approach:
        # Precompute for each edge, which subsets it affects.
        # When we add/remove an edge, we only need to re-check those subsets.
        
        return True  # Placeholder, actual check done incrementally

    best_G = 0
    best_count = 0
    best_subset_masks = [0] * num_subsets
    
    # Greedy construction with multiple strategies
    def greedy_construct(order):
        G = 0
        subset_masks = [0] * num_subsets
        
        for e_idx in order:
            can_add = True
            for s_idx in edge_to_subsets[e_idx]:
                l_pos = local_idx[s_idx][e_idx]
                new_mask = subset_masks[s_idx] | (1 << l_pos)
                if contains_q3(new_mask):
                    can_add = False
                    break
            
            if can_add:
                G |= (1 << e_idx)
                for s_idx in edge_to_subsets[e_idx]:
                    l_pos = local_idx[s_idx][e_idx]
                    subset_masks[s_idx] |= (1 << l_pos)
        
        return G, subset_masks

    # Strategy 1: Random restarts
    attempts = 0
    while time.time() < deadline - 20:
        attempts += 1
        edges = list(range(total_edges))
        random.shuffle(edges)
        G, subset_masks = greedy_construct(edges)
        count = G.bit_count()
        if count > best_count:
            best_count = count
            best_G = G
            best_subset_masks = subset_masks[:]

    # Convert best subset masks to numpy for SA
    best_subset_masks_np = np.array(best_subset_masks, dtype=np.uint64)
    
    # Simulated Annealing
    # We need to be able to:
    # 1. Add an edge: check if it creates a Q3 in any of its subsets
    # 2. Remove an edge: always safe
    # 3. Swap: remove one, add another (or more)
    
    # Let's implement a simpler SA: 
    # State: G (bitmask), subset_masks (list of 28-bit ints)
    # Move: pick a random missing edge, try to add it. If it creates a Q3, reject.
    # Or: pick a random present edge, remove it, then try to add random missing edges.
    
    # Actually, a better move set:
    # 1. Try to add a random missing edge (if Q3-free)
    # 2. Try to remove a random present edge and add 1-2 random missing edges (if Q3-free)
    
    def try_add_edge(G, subset_masks, e_idx):
        """Try to add edge e_idx. Returns (success, new_G, new_subset_masks)"""
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_mask = subset_masks[s_idx] | (1 << l_pos)
            if contains_q3(new_mask):
                return False, G, subset_masks
        # All good, add it
        new_G = G | (1 << e_idx)
        new_subset_masks = subset_masks[:]
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_subset_masks[s_idx] |= (1 << l_pos)
        return True, new_G, new_subset_masks

    def remove_edge(G, subset_masks, e_idx):
        """Remove edge e_idx. Returns (new_G, new_subset_masks)"""
        new_G = G & ~(1 << e_idx)
        new_subset_masks = subset_masks[:]
        for s_idx in edge_to_subsets[e_idx]:
            l_pos = local_idx[s_idx][e_idx]
            new_subset_masks[s_idx] &= ~(1 << l_pos)
        return new_G, new_subset_masks

    # Run SA
    if time.time() < deadline - 10:
        current_G = best_G
        current_subset_masks = best_subset_masks[:]
        current_count = current_G.bit_count()
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T and time.time() < deadline - 10:
            improved = False
            
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a random missing edge
                missing_edges = [i for i in range(total_edges) if not (current_G >> i) & 1]
                if not missing_edges:
                    T *= cooling
                    continue
                e_add = random.choice(missing_edges)
                success, trial_G, trial_masks = try_add_edge(current_G, current_subset_masks, e_add)
                if success:
                    trial_count = trial_G.bit_count()
                    delta = current_count - trial_count  # Should be -1
                    if delta <= 0 or random.random() < np.exp(-delta / T):
                        current_G = trial_G
                        current_subset_masks = trial_masks
                        current_count = trial_count
                        if current_count > best_count:
                            best_count = current_count
                            best_G = current_G
                            best_subset_masks = current_subset_masks[:]
                            improved = True
            else:
                # Try to remove one edge and add one or two
                present_edges = [i for i in range(total_edges) if (current_G >> i) & 1]
                if not present_edges:
                    T *= cooling
                    continue
                e_remove = random.choice(present_edges)
                trial_G, trial_masks = remove_edge(current_G, current_subset_masks, e_remove)
                
                # Try to add 1 or 2 random missing edges
                missing_edges = [i for i in range(total_edges) if not (trial_G >> i) & 1]
                num_add = random.randint(1, 2)
                added = []
                for _ in range(num_add):
                    if not missing_edges:
                        break
                    e_add = random.choice(missing_edges)
                    success, trial_G, trial_masks = try_add_edge(trial_G, trial_masks, e_add)
                    if success:
                        added.append(e_add)
                        missing_edges.remove(e_add)
                
                trial_count = trial_G.bit_count()
                delta = current_count - trial_count
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    current_G = trial_G
                    current_subset_masks = trial_masks
                    current_count = trial_count
                    if current_count > best_count:
                        best_count = current_count
                        best_G = current_G
                        best_subset_masks = current_subset_masks[:]
                        improved = True
            
            T *= cooling

    result = []
    for e_idx in range(total_edges):
        if (best_G >> e_idx) & 1:
            u, v = edge_list[e_idx]
            result.append([u, v])
            
    return result
