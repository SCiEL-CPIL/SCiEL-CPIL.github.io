import numpy as np
from itertools import combinations
from collections import deque
import math

def construct():
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
        
    # Precompute binomial coefficients C(n, k) for n in 0..p, k in 0..4
    # C(n, k) = 0 if n < k
    max_n = p
    max_k = 4
    C = [[0] * (max_k + 1) for _ in range(max_n + 1)]
    for n in range(max_n + 1):
        C[n][0] = 1
        for k in range(1, max_k + 1):
            if k <= n:
                C[n][k] = C[n-1][k-1] + C[n-1][k]
            else:
                C[n][k] = 0
                
    # Function to get index of 4-set (a,b,c,d)
    def get_idx(a, b, c, d):
        return C[a][1] + C[b][2] + C[c][3] + C[d][4]
        
    total_sets = C[p][4]
    
    # Initialize structures
    # doms[idx] = list of vertices dominating set idx
    doms = [[] for _ in range(total_sets)]
    # dominated_by[v] = list of set indices dominated by v
    dominated_by = [[] for _ in range(p)]
    # sets_with[v] = list of set indices containing v
    sets_with = [[] for _ in range(p)]
    
    # Populate doms and dominated_by
    for v in range(p):
        neighbors = [j for j in range(p) if (out[v] >> j) & 1]
        for s in combinations(neighbors, 4):
            idx = get_idx(s[0], s[1], s[2], s[3])
            doms[idx].append(v)
            dominated_by[v].append(idx)
            
    # Populate sets_with
    # Iterate all 4-sets
    all_4sets = list(combinations(range(p), 4))
    for s in all_4sets:
        idx = get_idx(s[0], s[1], s[2], s[3])
        for v in s:
            sets_with[v].append(idx)
            
    # Initialize counts and unique_counts
    counts = np.zeros(total_sets, dtype=np.int32)
    unique_counts = np.zeros(p, dtype=np.int32)
    is_active = np.ones(total_sets, dtype=bool)
    
    for idx in range(total_sets):
        c = len(doms[idx])
        counts[idx] = c
        if c == 1:
            v = doms[idx][0]
            unique_counts[v] += 1
            
    active = set(range(p))
    queue = deque()
    
    # Initialize queue with free vertices
    for v in range(p):
        if unique_counts[v] == 0:
            queue.append(v)
            
    # Pruning loop
    while queue:
        v = queue.popleft()
        
        # Check if v is still active and still free
        if v not in active:
            continue
        if unique_counts[v] != 0:
            continue
            
        # Remove v
        active.remove(v)
        
        # 1. Update sets dominated by v (v is not in these sets)
        # These sets remain active (since v not in them), but their count decreases
        for idx in dominated_by[v]:
            if not is_active[idx]:
                continue
                
            old_count = counts[idx]
            # v is a dominator, so it is in doms[idx]
            # We are removing v, so count decreases by 1
            new_count = old_count - 1
            counts[idx] = new_count
            
            if old_count == 2:
                # Now count is 1. Find the new unique dominator.
                # The new unique dominator is the only one in doms[idx] that is in active.
                # doms[idx] has 2 elements: v and u.
                # u must be in active.
                # We can just find the one in active.
                for u in doms[idx]:
                    if u in active:
                        unique_counts[u] += 1
                        # If u becomes free? No, unique_counts[u] increased.
                        # We are looking for vertices with count 0.
                        # So no need to add to queue.
                        break
                        
            # If old_count == 1, this would mean v was the unique dominator.
            # But we only remove v if unique_counts[v] == 0.
            # So v is not the unique dominator of any active set.
            # So old_count cannot be 1 for any active set dominated by v.
            
        # 2. Update sets containing v (deactivation)
        for idx in sets_with[v]:
            if not is_active[idx]:
                continue
                
            # Check if all other vertices in the set are active
            s = all_4sets[idx] # We need the actual vertices to check
            # Wait, all_4sets is a list of tuples.
            # We can check if all(u in active for u in s if u != v)
            
            # Optimization: s has 4 elements.
            # Since v is in s, we check the other 3.
            # We can just check if the set is still "full" in active.
            # Since we are removing v, and v was in active, the set becomes inactive
            # iff all other vertices were in active.
            
            # Iterate s
            active_count = 0
            for u in s:
                if u == v:
                    continue
                if u in active:
                    active_count += 1
            
            if active_count == 3:
                # The set was active, now it is inactive
                is_active[idx] = False
                old_count = counts[idx]
                
                if old_count == 1:
                    # v was the unique dominator?
                    # No, old_count is the number of dominators in active set BEFORE removal?
                    # Wait.
                    # `counts[idx]` was updated in step 1 if v dominated idx.
                    # But v is IN idx, so v does NOT dominate idx.
                    # So `counts[idx]` was NOT changed in step 1 for this idx.
                    # So `old_count` is the number of dominators in `active` BEFORE removing v.
                    # Since v is in the set, v is not a dominator.
                    # So `old_count` is the number of dominators in `active \ {v}`.
                    # When we deactivate the set, we don't care about its count anymore.
                    # BUT, if `old_count == 1`, there was a unique dominator `u` in `active`.
                    # `unique_counts[u]` was incremented when this set became unique.
                    # Now the set is inactive, so `u` no longer uniquely dominates this set.
                    # So we must decrement `unique_counts[u]`.
                    
                    # Find u
                    for u in doms[idx]:
                        if u in active: # u is not v, and u is in active
                            unique_counts[u] -= 1
                            if unique_counts[u] == 0:
                                queue.append(u)
                            break

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
            if (out[old_i] >> old_j) & 1:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': n, 'orientation': orientation}
