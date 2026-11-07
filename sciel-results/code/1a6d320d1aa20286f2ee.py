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
    
    full_mask = (1 << n) - 1
    
    # Precompute for all subsets: is it a clique? is it an independent set?
    # Use bitmasks: clique_mask[s] = True if subset s is a clique
    # indep_mask[s] = True if subset s is an independent set
    num_subsets = 1 << n
    is_clique = [False] * num_subsets
    is_indep = [False] * num_subsets
    
    # A subset is a clique if every pair of vertices in it is connected
    # A subset is an independent set if no pair of vertices in it is connected
    for s in range(1, num_subsets):
        # Get lowest set bit
        lsb = s & (-s)
        v = lsb.bit_length() - 1
        rest = s ^ lsb
        
        if rest == 0:
            is_clique[s] = True
            is_indep[s] = True
        else:
            # For clique: v must be adjacent to all vertices in rest
            if is_clique[rest] and (adj_mask[v] & rest) == rest:
                is_clique[s] = True
            # For independent set: v must not be adjacent to any vertex in rest
            if is_indep[rest] and (adj_mask[v] & rest) == 0:
                is_indep[s] = True
    
    # Order vertices by constraint: vertices that are in fewer valid subsets first
    # Count how many subsets containing vertex v are valid (clique or independent)
    vertex_constraint = [0] * n
    for v in range(n):
        count = 0
        for s in range(1, num_subsets):
            if s & (1 << v):
                if is_clique[s] or is_indep[s]:
                    count += 1
        vertex_constraint[v] = count
    
    vertices = list(range(n))
    vertices.sort(key=lambda x: vertex_constraint[x])
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
            
        masks = [0] * k
        
        # For each part, track whether it's still potentially a clique and/or independent set
        # part_type[i] = 0: both possible, 1: only clique, 2: only independent, 3: neither (invalid)
        part_type = [0] * k
        
        def get_part_status(part_mask, v):
            """Check if adding vertex v to part_mask keeps it valid.
            Returns new status: 0=both, 1=clique only, 2=indep only, 3=invalid"""
            new_mask = part_mask | (1 << v)
            c_ok = is_clique[new_mask]
            i_ok = is_indep[new_mask]
            if c_ok and i_ok:
                return 0
            elif c_ok:
                return 1
            elif i_ok:
                return 2
            else:
                return 3
        
        def forward_check(idx):
            """Check if all remaining vertices can be added to some part"""
            for j in range(idx, n):
                v = vertices[j]
                found = False
                for i in range(k):
                    if masks[i] == 0:
                        if i > 0 and masks[i-1] == 0:
                            continue
                        found = True
                        break
                    status = get_part_status(masks[i], v)
                    if status != 3:
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
            
            # Forward check
            if not forward_check(idx):
                return None
            
            v = vertices[idx]
            v_bit = 1 << v
            
            # Try adding v to existing parts
            # Order parts: try parts that become more constrained first
            part_order = list(range(k))
            
            def part_sort_key(i):
                if masks[i] == 0:
                    return 0
                status = get_part_status(masks[i], v)
                if status == 3:
                    return 999  # Invalid, skip
                elif status == 1 or status == 2:
                    return 1  # Becomes more constrained
                else:
                    return 2  # Still flexible
            
            part_order.sort(key=part_sort_key)
            
            for i in part_order:
                if masks[i] == 0:
                    if i > 0 and masks[i-1] == 0:
                        continue
                
                status = get_part_status(masks[i], v)
                if status != 3:
                    old_mask = masks[i]
                    old_type = part_type[i]
                    
                    masks[i] |= v_bit
                    part_type[i] = status
                    
                    res = dfs(idx + 1)
                    if res is not None:
                        return res
                    
                    masks[i] = old_mask
                    part_type[i] = old_type
            
            return None
        
        res = dfs(0)
        if res is not None:
            final_parts = []
            for part in res:
                final_parts.append([int(v) for v in part])
            return final_parts

    return [[i] for i in range(n)]
