import time
import itertools
import random
import numpy as np

N = 16
ALL_PAIRS = list(itertools.combinations(range(N), 2))
NUM_PAIRS = len(ALL_PAIRS)
EDGE_INDEX = {pair: i for i, pair in enumerate(ALL_PAIRS)}

# Precompute Q3 masks on 8 vertices
def _generate_q3_masks_int():
    masks = []
    vertices = list(range(8))
    all_local_edges = list(itertools.combinations(vertices, 2))
    local_edge_idx = {e: i for i, e in enumerate(all_local_edges)}
    
    for a_set in itertools.combinations(vertices, 4):
        a_list = list(a_set)
        b_list = [v for v in vertices if v not in a_set]
        
        for perm in itertools.permutations(b_list):
            matching_edges = set()
            for i in range(4):
                u, v = a_list[i], perm[i]
                matching_edges.add((min(u,v), max(u,v)))
            
            q3_edges = set()
            for a in a_list:
                for b in b_list:
                    e = (min(a,b), max(a,b))
                    if e not in matching_edges:
                        q3_edges.add(e)
            
            mask = 0
            for e in q3_edges:
                idx = local_edge_idx[e]
                mask |= (1 << idx)
            masks.append(mask)
            
    return masks

Q3_MASKS_INT = _generate_q3_masks_int()
Q3_MASKS_ARR = np.array(Q3_MASKS_INT, dtype=np.uint32)

# Precompute subset edge indices
SUBSETS = list(itertools.combinations(range(N), 8))
NUM_SUBSETS = len(SUBSETS)
LOCAL_EDGE_LIST = list(itertools.combinations(range(8), 2))

SUBSET_EDGE_INDICES = []
for vs in SUBSETS:
    indices = []
    for u, v in LOCAL_EDGE_LIST:
        gu, gv = vs[u], vs[v]
        if gu > gv:
            gu, gv = gv, gu
        indices.append(EDGE_INDEX[(gu, gv)])
    SUBSET_EDGE_INDICES.append(indices)

def construct():
    start = time.time()
    deadline = start + 140.0
    rng = random.Random(42)
    
    # Initialize with all edges
    graph = [1] * NUM_PAIRS
    
    def get_subset_mask(subset_idx):
        edge_indices = SUBSET_EDGE_INDICES[subset_idx]
        mask = 0
        for i, global_idx in enumerate(edge_indices):
            if graph[global_idx]:
                mask |= (1 << i)
        return mask
        
    def find_q3(mask):
        check = (mask & Q3_MASKS_ARR) == Q3_MASKS_ARR
        if np.any(check):
            idx = np.argmax(check)
            return Q3_MASKS_ARR[idx]
        return -1

    # Repair Loop: Iterate through all subsets, remove edges from Q3s found
    while time.time() < deadline - 5.0:
        found_any = False
        for subset_idx in range(NUM_SUBSETS):
            if time.time() > deadline - 5.0:
                break
            mask = get_subset_mask(subset_idx)
            q3_mask = find_q3(mask)
            if q3_mask != -1:
                found_any = True
                edges_to_remove = [i for i in range(28) if (q3_mask >> i) & 1]
                local_e = rng.choice(edges_to_remove)
                global_e = SUBSET_EDGE_INDICES[subset_idx][local_e]
                graph[global_e] = 0
        if not found_any:
            break

    result = []
    for i in range(NUM_PAIRS):
        if graph[i]:
            result.append(list(ALL_PAIRS[i]))
            
    return result
