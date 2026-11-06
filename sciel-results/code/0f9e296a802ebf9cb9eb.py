import numpy as np
from itertools import combinations

def construct():
    # Paley tournament on 67 vertices
    p = 67
    squares = {x * x % p for x in range(1, p)}
    
    # out[i] is a bitmask of vertices j such that i -> j
    out = [0] * p
    for i in range(p):
        mask = 0
        for j in range(p):
            if i == j:
                continue
            if (j - i) % p in squares:
                mask |= (1 << j)
        out[i] = mask
        
    # Generate all 4-sets and map to index
    vertices = list(range(p))
    all_4sets = list(combinations(vertices, 4))
    num_sets = len(all_4sets)
    
    # Map tuple to index for quick lookup
    tuple_to_idx = {s: i for i, s in enumerate(all_4sets)}
    
    # Precompute set masks (bitmask of vertices in the set)
    set_masks = [0] * num_sets
    for idx, s in enumerate(all_4sets):
        m = 0
        for v in s:
            m |= (1 << v)
        set_masks[idx] = m
            
    # dom_mask[i] is bitmask of vertices dominating set i
    dom_mask = [0] * num_sets
    
    # sets_containing[v] is list of set indices containing v
    sets_containing = [[] for _ in range(p)]
    
    # sets_dominated_by[v] is list of set indices dominated by v
    sets_dominated_by = [[] for _ in range(p)]
    
    # Populate sets_containing
    for idx, s in enumerate(all_4sets):
        for v in s:
            sets_containing[v].append(idx)
            
    # Populate dom_mask and sets_dominated_by
    # For each v, find all 4-subsets of out-neighbors
    for v in range(p):
        neighbors = [j for j in range(p) if (out[v] >> j) & 1]
        for s in combinations(neighbors, 4):
            idx = tuple_to_idx[s]
            dom_mask[idx] |= (1 << v)
            sets_dominated_by[v].append(idx)
            
    # Initialize unique_count[v]: number of sets for which v is the UNIQUE dominator
    unique_count = [0] * p
    
    # unique_dominator_of_set[i]: the unique dominator if popcount(dom_mask[i])==1, else -1
    unique_dom = [-1] * num_sets
    
    for i in range(num_sets):
        m = dom_mask[i]
        # Check popcount
        # Python 3.8+ has int.bit_count()
        if m.bit_count() == 1:
            # Find the bit
            v = m.bit_length() - 1
            unique_dom[i] = v
            unique_count[v] += 1
            
    # active_set[i] indicates if set i is still active
    active_set = [True] * num_sets
    
    # active_vertex[v] indicates if vertex v is still active
    active_vertex = [True] * p
    
    # Greedy removal
    # We want to remove vertices with unique_count[v] == 0
    # We need to be careful: removing a vertex might change unique_count of others.
    
    # To speed up finding candidates, we can just iterate through vertices.
    # Since p is small (67), linear scan is fine.
    
    # Optimization: 
    # When we remove v, we update unique_count for others.
    # We can just loop while we find any v with unique_count[v]==0.
    
    while True:
        # Find a removable vertex
        rem_v = -1
        for v in range(p):
            if active_vertex[v] and unique_count[v] == 0:
                rem_v = v
                break
                
        if rem_v == -1:
            break
            
        v = rem_v
        active_vertex[v] = False
        
        # 1. Process sets containing v (these become inactive)
        for idx in sets_containing[v]:
            if not active_set[idx]:
                continue
            active_set[idx] = False
            # If this set had a unique dominator u (u != v), u loses protection
            u = unique_dom[idx]
            if u != -1 and u != v:
                # u was the unique dominator. Since set is removed, u no longer protects this set.
                unique_count[u] -= 1
                # Note: unique_dom[idx] is no longer relevant, but we can leave it.
                # We must ensure we don't decrement again if we process this set elsewhere.
                # This set is now inactive, so it won't be processed in "sets_dominated_by" step?
                # Wait, "sets_dominated_by" step iterates over sets dominated by v.
                # If a set contains v, it cannot be dominated by v (since v is not in N+(v)).
                # So sets containing v are disjoint from sets dominated by v?
                # Yes, because if v dominates S, v not in S.
                # So no double counting issue between these two loops.
                
        # 2. Process sets dominated by v (these lose v as a dominator)
        for idx in sets_dominated_by[v]:
            if not active_set[idx]:
                continue
            # v was a dominator of this set.
            # Remove v from dom_mask[idx]
            old_mask = dom_mask[idx]
            new_mask = old_mask & ~(1 << v)
            dom_mask[idx] = new_mask
            
            # Check if unique dominator status changes
            old_pop = old_mask.bit_count()
            new_pop = new_mask.bit_count()
            
            # Case A: old_pop == 2
            # Then new_pop == 1. The remaining vertex becomes the unique dominator.
            if old_pop == 2:
                # Find the remaining vertex
                u = new_mask.bit_length() - 1
                # It becomes unique dominator
                unique_count[u] += 1
                unique_dom[idx] = u
            # Case B: old_pop > 2
            # new_pop >= 2. No unique dominator created.
            # If old_pop was 1, v would have been unique, but we only remove v if unique_count[v]==0.
            # So old_pop cannot be 1.
            
    # Construct result
    remaining_vertices = [v for v in range(p) if active_vertex[v]]
    n = len(remaining_vertices)
    
    # Map old indices to new indices
    old_to_new = {old: new for new, old in enumerate(remaining_vertices)}
    
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            old_i = remaining_vertices[i]
            old_j = remaining_vertices[j]
            if (out[old_i] >> old_j) & 1:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': n, 'orientation': orientation}
