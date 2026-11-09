"""Dense Q3-free graph on 16 vertices via local search with bitmask checks."""
import random
import itertools
import time
import math

# Precompute Q3 edge patterns as 28-bit masks for 8-vertex subsets
_Q3_EDGES = [
    (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),
    (3,7),(4,5),(4,6),(5,7),(6,7)
]

_EDGE_INDEX = {}
_idx = 0
for i in range(8):
    for j in range(i+1, 8):
        _EDGE_INDEX[(i, j)] = _idx
        _idx += 1

def _build_q3_masks():
    base_mask = 0
    for u, v in _Q3_EDGES:
        base_mask |= (1 << _EDGE_INDEX[(u, v)])
    masks = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in _Q3_EDGES:
            pu, pv = perm[u], perm[v]
            if pu > pv:
                pu, pv = pv, pu
            mask |= (1 << _EDGE_INDEX[(pu, pv)])
        masks.add(mask)
    return list(masks)

_Q3_MASKS = _build_q3_masks()

def construct():
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges_total = len(all_edges)
    
    all_subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(all_subsets)
    
    edge_to_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx
    
    # Precompute subset edge mappings
    subset_edge_maps = []
    for s in all_subsets:
        mappings = []
        for i in range(8):
            for j in range(i+1, 8):
                local_idx = _EDGE_INDEX[(i, j)]
                gu, gv = s[i], s[j]
                if gu > gv:
                    gu, gv = gv, gu
                mappings.append((local_idx, gu, gv))
        subset_edge_maps.append(mappings)
    
    # For each edge, which subsets contain it, and what's the local idx in each
    edge_to_subsets = [[] for _ in range(num_edges_total)]
    for s_idx, s in enumerate(all_subsets):
        mappings = subset_edge_maps[s_idx]
        for local_idx, gu, gv in mappings:
            eidx = edge_to_idx[(gu, gv)]
            edge_to_subsets[eidx].append((s_idx, local_idx))
    
    # Initialize with all edges
    edge_present = [True] * num_edges_total
    
    # Compute initial masks
    subset_masks = [0] * num_subsets
    for s_idx in range(num_subsets):
        mask = 0
        for local_idx, gu, gv in subset_edge_maps[s_idx]:
            if edge_present[edge_to_idx[(gu, gv)]]:
                mask |= (1 << local_idx)
        subset_masks[s_idx] = mask
    
    def current_energy():
        total = 0
        for s_idx in range(num_subsets):
            mask = subset_masks[s_idx]
            for pm in _Q3_MASKS:
                if (mask & pm) == pm:
                    total += 1
        return total
    
    start_time = time.time()
    max_time = 100
    
    # Phase 1: Greedy removal to reach 0 violations
    energy = current_energy()
    for it in range(10000):
        if time.time() - start_time > max_time - 40:
            break
        if energy == 0:
            break
        
        # Find a violated subset
        found = False
        for s_idx in range(num_subsets):
            mask = subset_masks[s_idx]
            for pm in _Q3_MASKS:
                if (mask & pm) == pm:
                    # Find edges in this pattern
                    candidates = []
                    for local_idx, gu, gv in subset_edge_maps[s_idx]:
                        if (pm >> local_idx) & 1:
                            eidx = edge_to_idx[(gu, gv)]
                            if edge_present[eidx]:
                                candidates.append(eidx)
                    if candidates:
                        eidx_to_remove = random.choice(candidates)
                        edge_present[eidx_to_remove] = False
                        # Update masks
                        for s2, li in edge_to_subsets[eidx_to_remove]:
                            subset_masks[s2] &= ~(1 << li)
                        found = True
                        break
            if found:
                break
        if not found:
            break
        energy = current_energy()
    
    # Phase 2: Local search to maximize edges
    T = 30.0
    T_min = 0.1
    cooling = 0.9995
    max_sa_iters = 80000
    
    for sa_iter in range(max_sa_iters):
        if time.time() - start_time > max_time - 10:
            break
        if T < T_min:
            break
        
        eidx = random.randint(0, num_edges_total - 1)
        old_state = edge_present[eidx]
        new_state = not old_state
        
        # Compute delta energy
        delta_energy = 0
        for s_idx, local_idx in edge_to_subsets[eidx]:
            old_mask = subset_masks[s_idx]
            if new_state:
                new_mask = old_mask | (1 << local_idx)
            else:
                new_mask = old_mask & ~(1 << local_idx)
            
            for pm in _Q3_MASKS:
                old_v = 1 if (old_mask & pm) == pm else 0
                new_v = 1 if (new_mask & pm) == pm else 0
                delta_energy += (new_v - old_v)
        
        delta_edges = 1 if new_state else -1
        
        if delta_energy <= 0 and delta_edges > 0:
            accept = True
        elif delta_energy > 0 and delta_edges > 0:
            accept = random.random() < math.exp(-delta_energy * 3 / T) if T > 0.01 else False
        elif delta_energy < 0 and delta_edges < 0:
            accept = random.random() < math.exp(delta_edges * 0.5 / T) if T > 0.01 else False
        else:
            accept = False
        
        if accept:
            edge_present[eidx] = new_state
            for s_idx, local_idx in edge_to_subsets[eidx]:
                if new_state:
                    subset_masks[s_idx] |= (1 << local_idx)
                else:
                    subset_masks[s_idx] &= ~(1 << local_idx)
        
        T *= cooling
    
    # Phase 3: Greedy add back
    for eidx in range(num_edges_total):
        if time.time() - start_time > max_time - 2:
            break
        if edge_present[eidx]:
            continue
        ok = True
        for s_idx, local_idx in edge_to_subsets[eidx]:
            new_mask = subset_masks[s_idx] | (1 << local_idx)
            for pm in _Q3_MASKS:
                if (new_mask & pm) == pm:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            edge_present[eidx] = True
            for s_idx, local_idx in edge_to_subsets[eidx]:
                subset_masks[s_idx] |= (1 << local_idx)
    
    result = []
    for idx, (u, v) in enumerate(all_edges):
        if edge_present[idx]:
            result.append([u, v])
    
    return result
