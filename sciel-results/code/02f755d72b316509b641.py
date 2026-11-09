    def construct():
        import sys
        sys.setrecursionlimit(10000)
        
        def instance(splits, seed, palette=5):
            # ... same as before ...
            
        def solve_mwis(n, edges, lists):
            # 1. Build Conflict Graph
            # Node index: v * 3 + c
            M = n * 3
            neighbors = [0] * M
            
            # Add intra-vertex edges (cliques of 3)
            for v in range(n):
                for c1 in range(3):
                    for c2 in range(3):
                        if c1 != c2:
                            idx1 = v * 3 + c1
                            idx2 = v * 3 + c2
                            neighbors[idx1] |= (1 << idx2)
                            neighbors[idx2] |= (1 << idx1)
                            
            # Add inter-vertex edges
            for u, v in edges:
                # Colors must be same to conflict
                # But lists might be different?
                # Conflict exists if c is in L_u AND c is in L_v.
                # Node (u, c) exists only if c in L_u.
                # Node (v, c) exists only if c in L_v.
                # If c is in both, add edge.
                
                # We need to map colors to indices correctly.
                # The node index is v*3 + c. This assumes colors are 0,1,2?
                # NO. The colors are from 0..4.
                # The lists contain specific colors.
                # My index mapping `v*3 + c` is wrong if c is not 0,1,2.
                
            # CORRECTION:
            # The nodes are (v, c) where c is in lists[v].
            # We need a mapping from (v, c) to a unique index 0..M-1.
            
            node_id = {}
            id_to_node = []
            idx = 0
            for v in range(n):
                for c in lists[v]:
                    node_id[(v, c)] = idx
                    id_to_node.append((v, c))
                    idx += 1
            M = idx
            
            neighbors = [0] * M
            
            # Intra-vertex edges
            for v in range(n):
                ids = [node_id[(v, c)] for c in lists[v]]
                for i in range(len(ids)):
                    for j in range(len(ids)):
                        if i != j:
                            neighbors[ids[i]] |= (1 << ids[j])
                            
            # Inter-vertex edges
            for u, v in edges:
                # Find common colors
                # lists are small (size 3)
                for c in lists[u]:
                    if c in lists[v]:
                        id_u = node_id[(u, c)]
                        id_v = node_id[(v, c)]
                        neighbors[id_u] |= (1 << id_v)
                        neighbors[id_v] |= (1 << id_u)
                        
            # 2. Solve MIS
            target = n
            full_mask = (1 << M) - 1
            
            def search(P, count):
                if count == target:
                    return True
                if P == 0:
                    return False
                if count + P.bit_count() < target:
                    return False
                
                # Heuristic: Max degree in P
                best_v = -1
                best_deg = -1
                p_temp = P
                while p_temp:
                    # Get lowest set bit
                    lsb = p_temp & -p_temp
                    v = lsb.bit_length() - 1
                    deg = (P & neighbors[v]).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    p_temp ^= lsb
                    
                if best_v == -1:
                    return False
                
                # Branch 1: Include best_v
                new_P = P & ~neighbors[best_v] & ~(1 << best_v)
                if search(new_P, count + 1):
                    return True
                    
                # Branch 2: Exclude best_v
                new_P = P & ~(1 << best_v)
                if search(new_P, count):
                    return True
                    
                return False
                
            # Run search
            if search(full_mask, 0):
                # Extract solution
                # We need to reconstruct which nodes were picked.
                # The search above returns bool. We need to modify it to return the set.
                
                # Let's modify search to return the set of nodes.
                
                def search_ret(P, count):
                    if count == target:
                        return set() # Base case, actually we need to accumulate
                    # This is tricky with recursion.
                    
                # Better: Pass a list to collect nodes.
                
                solution_nodes = []
                
                def search_collect(P, count):
                    if count == target:
                        return True
                    if P == 0:
                        return False
                    if count + P.bit_count() < target:
                        return False
                    
                    best_v = -1
                    best_deg = -1
                    p_temp = P
                    while p_temp:
                        lsb = p_temp & -p_temp
                        v = lsb.bit_length() - 1
                        deg = (P & neighbors[v]).bit_count()
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                        p_temp ^= lsb
                        
                    if best_v == -1:
                        return False
                    
                    # Try Include
                    new_P = P & ~neighbors[best_v] & ~(1 << best_v)
                    if search_collect(new_P, count + 1):
                        solution_nodes.append(best_v)
                        return True
                        
                    # Try Exclude
                    new_P = P & ~(1 << best_v)
                    if search_collect(new_P, count):
                        return True
                        
                    return False
                
                if search_collect(full_mask, 0):
                    # solution_nodes has the indices
                    # Map back to colors
                    col = [0] * n
                    for node_idx in solution_nodes:
                        v, c = id_to_node[node_idx]
                        col[v] = c
                    return col
                else:
                    return [0] * n
            else:
                return [0] * n

        # ... main loop ...
    