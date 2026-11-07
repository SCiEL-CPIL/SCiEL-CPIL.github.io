import time
import sys
from itertools import combinations

sys.setrecursionlimit(10000)

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute: for each subset (as bitmask), check if it's a clique or independent set
    # A subset is a clique if every pair is adjacent
    # A subset is independent if no pair is adjacent
    
    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be adjacent to all in rest, and rest must be clique
        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True
        
        # Check independent: v must not be adjacent to any in rest, and rest must be independent
        if is_indep[rest] and (adj_mask[v] & rest) == 0:
            is_indep[mask] = True
    
    # A mask is "homogeneous" if it's a clique or independent set
    is_homo = [False] * (1 << n)
    for mask in range(1 << n):
        is_homo[mask] = is_clique[mask] or is_indep[mask]
    
    # Now search for minimum number of homogeneous parts that partition all vertices
    # Use DFS with pruning
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Order vertices for search
    degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
    constraint_score = [max(degrees[i], n - 1 - degrees[i]) for i in range(n)]
    vertices = list(range(n))
    vertices.sort(key=lambda x: constraint_score[x], reverse=True)
    
    # For the search, we assign vertices one by one to parts
    # Each part is represented by its current mask and type (clique/indep/both)
    
    best_result = None
    best_k = n + 1
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        
        masks = [0] * k
        is_c = [True] * k
        is_i = [True] * k
        
        def forward_check(idx):
            """Check if all remaining vertices can be added to some part"""
            for j in range(idx, n):
                v = vertices[j]
                v_bit = 1 << v
                found = False
                for i in range(k):
                    if masks[i] == 0:
                        if i > 0 and masks[i - 1] == 0:
                            continue
                        # Single vertex is always both clique and independent
                        found = True
                        break
                    new_mask = masks[i] | v_bit
                    if is_homo[new_mask]:
                        found = True
                        break
                if not found:
                    return False
            return True
        
        def dfs(idx):
            if time.time() > deadline:
                return None
            
            if idx == n:
                res = []
                for i in range(k):
                    if masks[i] != 0:
                        part_verts = []
                        for v in range(n):
                            if masks[i] & (1 << v):
                                part_verts.append(v)
                        res.append(part_verts)
                return res
            
            if not forward_check(idx):
                return None
            
            v = vertices[idx]
            v_bit = 1 << v
            
            # Try adding v to existing parts
            # Symmetry breaking: only use first empty part
            first_empty = -1
            for i in range(k):
                if masks[i] == 0:
                    first_empty = i
                    break
            
            # Order parts: prefer parts where adding v keeps more flexibility
            # Try parts where the result is still both clique and indep first (most flexible)
            part_order = list(range(k))
            
            def part_key(i):
                if masks[i] == 0:
                    return (0, 0)
                new_mask = masks[i] | v_bit
                if not is_homo[new_mask]:
                    return (3, 0)  # Invalid, won't be tried
                # Count how many types remain
                # If new_mask is both clique and indep, it's most flexible
                if is_clique[new_mask] and is_indep[new_mask]:
                    type_flex = 2
                elif is_clique[new_mask] or is_indep[new_mask]:
                    type_flex = 1
                else:
                    type_flex = 0
                size = bin(masks[i]).count('1')
                return (type_flex, -size)
            
            part_order.sort(key=part_key, reverse=True)
            
            for i in part_order:
                if masks[i] == 0:
                    if i != first_empty:
                        continue
                
                new_mask = masks[i] | v_bit
                if not is_homo[new_mask]:
                    continue
                
                old_mask = masks[i]
                old_c = is_c[i]
                old_i = is_i[i]
                
                masks[i] = new_mask
                is_c[i] = is_clique[new_mask]
                is_i[i] = is_indep[new_mask]
                
                res = dfs(idx + 1)
                if res is not None:
                    return res
                
                masks[i] = old_mask
                is_c[i] = old_c
                is_i[i] = old_i
            
            return None
        
        res = dfs(0)
        if res is not None:
            final_parts = []
            for part in res:
                final_parts.append([int(v) for v in part])
            return final_parts
    
    # Fallback: singleton partition
    return [[i] for i in range(n)]
