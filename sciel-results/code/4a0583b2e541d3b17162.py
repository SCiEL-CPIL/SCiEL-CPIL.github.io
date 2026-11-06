import itertools
import random
import time

def construct():
    n = 16
    vertices = list(range(n))
    
    # 1. Generate all edges
    all_edges = list(itertools.combinations(vertices, 3))
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # 2. Generate all K(2,2,2) structures
    # A structure is a partition of 6 vertices into 3 pairs.
    # We represent a structure by a tuple of 3 pairs, sorted.
    # e.g., ((v1, v2), (v3, v4), (v5, v6)) with v1<v2, v3<v4, v5<v6, and (v1,v2)<(v3,v4)<(v5,v6)
    
    structures = []
    struct_edges = [] # list of 8 edges for each structure
    
    # Iterate over all 6-subsets
    for subset in itertools.combinations(vertices, 6):
        # Generate all 15 pairings of 6 elements
        # Elements: 0,1,2,3,4,5
        # Pair 0 with 1,2,3,4,5
        for i in range(1, 6):
            # 0 is paired with i
            # Remaining 4 elements:
            rem = [j for j in range(6) if j != 0 and j != i]
            # Pair remaining 4 in 3 ways:
            # (rem[0], rem[1]) and (rem[2], rem[3])
            # (rem[0], rem[2]) and (rem[1], rem[3])
            # (rem[0], rem[3]) and (rem[1], rem[2])
            pairs1 = (tuple(sorted((subset[0], subset[i]))), tuple(sorted((subset[rem[0]], subset[rem[1]]))), tuple(sorted((subset[rem[2]], subset[rem[3]]))))
            pairs2 = (tuple(sorted((subset[0], subset[i]))), tuple(sorted((subset[rem[0]], subset[rem[2]]))), tuple(sorted((subset[rem[1]], subset[rem[3]]))))
            pairs3 = (tuple(sorted((subset[0], subset[i]))), tuple(sorted((subset[rem[0]], subset[rem[3]]))), tuple(sorted((subset[rem[1]], subset[rem[2]]))))
            
            for p in [pairs1, pairs2, pairs3]:
                # Sort the pairs to have a canonical representation
                p_canon = tuple(sorted(p))
                structures.append(p_canon)
                
                # Determine the 8 edges
                # Parts are p[0], p[1], p[2]
                e = []
                for a in p[0]:
                    for b in p[1]:
                        for c in p[2]:
                            e.append(tuple(sorted((a, b, c))))
                struct_edges.append(e)
                
    # Map structure to ID
    struct_id = {s: i for i, s in enumerate(structures)}
    num_structs = len(structures)
    
    # Map edge to list of structure IDs
    # edge_structs[edge] = [struct_id_1, struct_id_2, ...]
    edge_structs = [[] for _ in range(len(all_edges))]
    
    for s_id, edges in enumerate(struct_edges):
        for e in edges:
            e_idx = edge_index[e]
            edge_structs[e_idx].append(s_id)
            
    # Greedy construction
    def greedy(seed):
        random.seed(seed)
        edge_indices = list(range(len(all_edges)))
        random.shuffle(edge_indices)
        
        counts = [0] * num_structs
        selected = []
        
        for e_idx in edge_indices:
            can_add = True
            for s_id in edge_structs[e_idx]:
                if counts[s_id] == 7:
                    can_add = False
                    break
            if can_add:
                selected.append(all_edges[e_idx])
                for s_id in edge_structs[e_idx]:
                    counts[s_id] += 1
                    
        return selected

    best = []
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    seed = 0
    while time.time() < deadline:
        res = greedy(seed)
        if len(res) > len(best):
            best = res
        seed += 1
        
    # Convert tuples to lists as required
    return [list(e) for e in best]
