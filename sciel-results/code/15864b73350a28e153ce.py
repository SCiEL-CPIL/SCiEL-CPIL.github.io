import time
import random
from itertools import combinations

def construct():
    n = 9
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute edge masks and vertex-to-edge incidence
    edge_masks = []
    for e in all_edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # vertex_edges[v] = list of edge indices containing v
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(all_edges):
        for v in e:
            vertex_edges[v].append(i)

    def is_non_2_colorable(edge_indices):
        if not edge_indices:
            return False
        
        # Convert to set for fast lookup? Or use a boolean array?
        # For n=9, E=126, a boolean array of size 126 is fine.
        edge_set = set(edge_indices)
        
        def backtrack(v_idx, c0, c1):
            if v_idx == n:
                return False # Successfully colored
            
            # Try color 0
            valid_c0 = True
            for e_idx in vertex_edges[v_idx]:
                if e_idx in edge_set:
                    mask = edge_masks[e_idx]
                    rest_mask = mask & ~(1 << v_idx)
                    if rest_mask and (rest_mask & ~c0) == 0:
                        valid_c0 = False
                        break
            if valid_c0:
                if not backtrack(v_idx + 1, c0 | (1 << v_idx), c1):
                    return False
            
            # Try color 1
            valid_c1 = True
            for e_idx in vertex_edges[v_idx]:
                if e_idx in edge_set:
                    mask = edge_masks[e_idx]
                    rest_mask = mask & ~(1 << v_idx)
                    if rest_mask and (rest_mask & ~c1) == 0:
                        valid_c1 = False
                        break
            if valid_c1:
                if not backtrack(v_idx + 1, c0, c1 | (1 << v_idx)):
                    return False
            
            return True

        return backtrack(0, 0, 0)

    best_edges = None
    best_size = num_edges
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    # Random restarts
    while time.time() - start_time < time_limit:
        current_edges = set(range(num_edges))
        shuffled = list(range(num_edges))
        random.shuffle(shuffled)
        
        for e_idx in shuffled:
            if e_idx not in current_edges:
                continue
            
            new_edges = current_edges - {e_idx}
            
            if is_non_2_colorable(new_edges):
                current_edges = new_edges
        
        if len(current_edges) < best_size:
            best_size = len(current_edges)
            best_edges = list(current_edges)
            
            if best_size == 51:
                break
    
    if best_edges is None:
        best_edges = list(range(num_edges))
        
    result_edges = [list(all_edges[i]) for i in best_edges]
    
    return {'n': n, 'edges': result_edges}
