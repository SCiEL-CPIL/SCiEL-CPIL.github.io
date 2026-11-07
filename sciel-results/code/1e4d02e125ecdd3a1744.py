import time
import sys

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
        
    # Order vertices by degree descending
    degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
    vertices = list(range(n))
    vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    # Map original vertex to sorted index? 
    # We process vertices in sorted order.
    # We need to return original labels.
    # vertices[idx] gives the original label.
    
    start_time = time.time()
    deadline = start_time + 80.0 # 80s margin
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # State:
        # masks: list of k ints (bitmasks of vertices in each part)
        # is_c: list of k bools (can this part still be a clique?)
        # is_i: list of k bools (can this part still be an independent set?)
        
        # Initialize
        masks = [0] * k
        is_c = [True] * k
        is_i = [True] * k
        
        # We will use a recursive helper
        # To avoid overhead of passing lists, we modify in place and backtrack
        
        def dfs(idx):
            if time.time() > deadline:
                return None
            
            if idx == n:
                # Check if all parts are non-empty?
                # The problem requires non-empty lists in the output.
                # If a part is empty, we just don't include it in the final result.
                # But wait, if we allocated k parts, and some are empty, 
                # we effectively used < k parts.
                # Since we iterate k from 1 up, if we find a solution with k parts,
                # and some are empty, it means a solution with fewer parts exists.
                # But we would have found it in a previous iteration.
                # So if we reach here with k, all parts should be non-empty?
                # Not necessarily. If k=3, and we fill part 0 and 1, part 2 is empty.
                # This is a valid partition into 2 parts.
                # But we would have found it when k=2.
                # So if we are in k=3, and find a solution, it might use fewer.
                # We should check if the number of non-empty parts is minimal?
                # No, we iterate k upwards. The first k for which a solution exists
                # is the minimum number of parts *if* we force all parts to be non-empty?
                # No, the problem allows empty parts in the search structure, 
                # but the output must be non-empty lists.
                # If I find a solution for k=3 that uses only 2 parts, 
                # it means k=2 was possible.
                # So if k=2 failed, k=3 must use all 3 parts?
                # Not necessarily. k=2 might fail because of how vertices were assigned,
                # but a different assignment with 2 parts might exist?
                # No, if a partition into 2 parts exists, the search for k=2 should find it.
                # The search for k=2 tries all assignments of vertices to 2 parts.
                # If it fails, no partition into 2 parts exists.
                # So if k=3 succeeds, it must use exactly 3 non-empty parts.
                
                # Extract non-empty parts
                res = []
                for i in range(k):
                    if masks[i] != 0:
                        # Convert mask to list of original vertex labels
                        part_verts = []
                        for v in range(n):
                            if masks[i] & (1 << v):
                                part_verts.append(v)
                        res.append(part_verts)
                return res
            
            v = vertices[idx]
            v_bit = 1 << v
            
            # Try adding v to existing parts
            for i in range(k):
                # Symmetry breaking: if part i is empty, it must be the first empty
                if masks[i] == 0:
                    # Check if there is a previous empty part
                    # Since we iterate i from 0, if masks[i] is 0, 
                    # and masks[i-1] was 0, we skip?
                    # Actually, if masks[i] is 0, and i>0, and masks[i-1] is 0, 
                    # then we should have used part i-1.
                    # So we only use the first empty part.
                    if i > 0 and masks[i-1] == 0:
                        continue
                
                # Check validity
                new_is_c = is_c[i]
                new_is_i = is_i[i]
                
                # Check clique constraint
                if new_is_c:
                    # v must be adjacent to all in masks[i]
                    if (adj_mask[v] & masks[i]) != masks[i]:
                        new_is_c = False
                
                # Check independent constraint
                if new_is_i:
                    # v must be non-adjacent to all in masks[i]
                    if (adj_mask[v] & masks[i]) != 0:
                        new_is_i = False
                
                if new_is_c or new_is_i:
                    # Save state
                    old_mask = masks[i]
                    old_c = is_c[i]
                    old_i = is_i[i]
                    
                    # Update state
                    masks[i] |= v_bit
                    is_c[i] = new_is_c
                    is_i[i] = new_is_i
                    
                    res = dfs(idx + 1)
                    if res is not None:
                        return res
                    
                    # Backtrack
                    masks[i] = old_mask
                    is_c[i] = old_c
                    is_i[i] = old_i
            
            return None
        
        res = dfs(0)
        if res is not None:
            # Ensure types
            final_parts = []
            for part in res:
                final_parts.append([int(v) for v in part])
            return final_parts

    # Fallback
    return [[i] for i in range(n)]
