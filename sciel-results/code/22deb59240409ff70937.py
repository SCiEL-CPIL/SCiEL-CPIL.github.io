import numpy as np
from itertools import combinations

def construct():
    # Paley tournament on 67 vertices
    p = 67
    # squares mod 67
    squares = {x * x % p for x in range(1, p)}
    
    # out[i] is a bitmask of vertices j such that i -> j
    out_mask = [0] * p
    for i in range(p):
        mask = 0
        for j in range(p):
            if i == j:
                continue
            if (j - i) % p in squares:
                mask |= (1 << j)
        out_mask[i] = mask
        
    # Generate all 4-sets
    vertices = list(range(p))
    all_4sets = list(combinations(vertices, 4))
    num_sets = len(all_4sets)
    
    # Precompute masks for each 4-set
    masks = [0] * num_sets
    for idx, s in enumerate(all_4sets):
        m = 0
        for v in s:
            m |= (1 << v)
        masks[idx] = m
        
    # Precompute doms_full[idx]: list of vertices v such that all_4sets[idx] is subset of N+(v)
    doms_full = [[] for _ in range(num_sets)]
    # To make it faster, for each v, find all 4-sets of N+(v) and add v to doms_full
    # Or iterate over all sets and check all v?
    # num_sets ~ 750k, p=67. 750k * 67 = 50M checks.
    # Each check is (mask & ~out_mask[v]) == 0.
    # This is doable in ~5-10s in Python if optimized?
    # Let's try the other way: for each v, generate 4-subsets of N+(v).
    
    # Map from set tuple to index
    set_to_idx = {s: i for i, s in enumerate(all_4sets)}
    
    for v in range(p):
        neighbors = [j for j in range(p) if (out_mask[v] >> j) & 1]
        for s in combinations(neighbors, 4):
            idx = set_to_idx[s]
            doms_full[idx].append(v)
            
    # Initialize active
    active = set(range(p))
    
    # Initialize unique_count
    # unique_count[v] = number of 4-sets S (disjoint from v, S subset of active)
    # such that v is the unique dominator of S among active vertices.
    unique_count = [0] * p
    
    for idx in range(num_sets):
        s = all_4sets[idx]
        # Find dominators in active
        # s is a tuple of 4 vertices.
        # A vertex v dominates s if s is subset of N+(v).
        # This is equivalent to v being in doms_full[idx] AND v in active.
        # Note: if v is in s, v cannot dominate s.
        # doms_full[idx] only contains v if s subset N+(v), which implies v not in s.
        # So we just intersect doms_full[idx] with active.
        
        # Optimization: doms_full[idx] is small.
        doms_active = [v for v in doms_full[idx] if v in active]
        
        if len(doms_active) == 1:
            unique_count[doms_active[0]] += 1
            
    # Pruning loop
    while True:
        # Find a free vertex
        free_v = -1
        for v in range(p):
            if v in active and unique_count[v] == 0:
                free_v = v
                break
                
        if free_v == -1:
            break
            
        v = free_v
        active.remove(v)
        
        # Update unique_count
        # 1. Sets containing v
        # For each 4-set S containing v, if S \ {v} is subset of active:
        #    Check if S had a unique dominator in old active set.
        #    If yes, decrement unique_count[that dominator].
        
        # We need to iterate over 4-sets containing v.
        # Generate all 4-sets containing v from the remaining active vertices?
        # No, we need to consider sets that WERE active before removal.
        # A set S containing v was active if S \ {v} subset of active_old.
        # active_old = active_new U {v}.
        # So S \ {v} subset of active_new U {v}.
        # Since v is in S, S \ {v} does not contain v.
        # So S \ {v} subset of active_new.
        # So we iterate over 4-sets S such that v in S and S \ {v} subset of active.
        
        # How to iterate?
        # Generate all 3-subsets of active. Let T be a 3-subset.
        # S = T U {v}.
        # Check if S is in set_to_idx.
        # If yes, get idx.
        # Check if doms_active (in old active) had size 1.
        # doms_old = [u for u in doms_full[idx] if u in active OR u == v]
        # But v cannot be in doms_full[idx] because v in S.
        # So doms_old = [u for u in doms_full[idx] if u in active]
        # Wait, active is now active_new.
        # doms_old should be calculated with active_old.
        # active_old = active U {v}.
        # So doms_old = [u for u in doms_full[idx] if u in active or u == v]
        # Since v not in doms_full[idx], doms_old = [u for u in doms_full[idx] if u in active]
        
        for T in combinations(active, 3):
            S = tuple(sorted(T + (v,)))
            if S in set_to_idx:
                idx = set_to_idx[S]
                # Calculate doms_old
                # doms_full[idx] contains vertices that dominate S.
                # None of them is v.
                # So we just check which are in active (which is active_new).
                # Wait, is it possible that a vertex u in doms_full[idx] was removed earlier?
                # Yes. But `active` contains only currently active vertices.
                # So `u in active` is the correct check for "u was active before removal".
                # Because we only remove one vertex at a time, and `active` is updated.
                
                doms_old = [u for u in doms_full[idx] if u in active]
                if len(doms_old) == 1:
                    unique_count[doms_old[0]] -= 1

        # 2. Sets NOT containing v
        # For each 4-set S not containing v, if S subset of active:
        #    If v in doms_full[idx]:
        #       Check if len(doms_old) == 2.
        #       If yes, find the other dominator u in doms_old (u != v).
        #       unique_count[u] += 1.
        
        # Iterate over 4-subsets of active.
        for S in combinations(active, 4):
            if S in set_to_idx:
                idx = set_to_idx[S]
                # Check if v dominates S
                # v dominates S if v in doms_full[idx]
                if v in doms_full[idx]:
                    # Calculate doms_old
                    # doms_old includes v (since v dominates S and v was active)
                    # and other active vertices that dominate S.
                    doms_old = [u for u in doms_full[idx] if u in active or u == v]
                    # We know v is in doms_old.
                    if len(doms_old) == 2:
                        # Find the other one
                        for u in doms_old:
                            if u != v:
                                unique_count[u] += 1
                                break
                    # If len(doms_old) > 2, no change.
                    # If len(doms_old) == 1, impossible because v is in it and unique_count[v]==0.

    # Construct the result
    remaining_vertices = sorted(list(active))
    n = len(remaining_vertices)
    
    # Map old indices to new indices
    old_to_new = {old: new for new, old in enumerate(remaining_vertices)}
    
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            old_i = remaining_vertices[i]
            old_j = remaining_vertices[j]
            # Check orientation in original graph
            # 0 means i -> j, 1 means j -> i
            if (out_mask[old_i] >> old_j) & 1:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': n, 'orientation': orientation}
