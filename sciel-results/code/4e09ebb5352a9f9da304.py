import time
import numpy as np

def construct():
    start_time = time.time()
    
    # Try n=18 first, then 17
    for n in [18, 17]:
        if time.time() - start_time > 115:
            continue
            
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # Use bitmasks for adjacency for speed
        # adj[c][u] is a bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * num_edges
        
        # Symmetry breaking:
        # Fix (0,1)=0
        # Fix (0,2)=0
        # Fix (0,3)=0
        # Fix (0,4)=0
        # Fix (0,5)=1
        # This breaks color symmetry and some vertex symmetry.
        # Indices:
        # (0,1) is index 0
        # (0,2) is index 1
        # (0,3) is index 2
        # (0,4) is index 3
        # (0,5) is index 4
        
        def set_edge(u, v, c, idx):
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            colors[idx] = c

        def unset_edge(u, v, c, idx):
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            colors[idx] = -1

        # Apply symmetry breaking
        # We need to be careful not to fix too many edges that might make it unsolvable
        # Standard: Fix (0,1)=0.
        # Then for (0,2), try 0 or 1? No, fix to 0 to break color perm further?
        # If we fix (0,1)=0, (0,2)=0, (0,3)=0, (0,4)=0, (0,5)=1.
        # This assumes a specific structure.
        
        # Let's use a safer symmetry breaking:
        # (0,1) = 0
        # (0,2) = 0
        # (0,3) = 0
        # (0,4) = 0
        # (0,5) = 1
        # This might be too restrictive?
        # Let's stick to:
        # (0,1)=0, (0,2)=0, (0,3)=0, (0,4)=1.
        
        fixed_edges = [
            (0, 1, 0),
            (0, 2, 0),
            (0, 3, 0),
            (0, 4, 1)
        ]
        
        start_idx = 0
        for u, v, c in fixed_edges:
            idx = 0
            # Find index
            for k, (eu, ev) in enumerate(edges):
                if eu == u and ev == v:
                    idx = k
                    break
            set_edge(u, v, c, idx)
            if idx > start_idx:
                start_idx = idx + 1
            else:
                start_idx = max(start_idx, idx + 1)
                
        # Actually, lexicographic order:
        # (0,1) is 0
        # (0,2) is 1
        # (0,3) is 2
        # (0,4) is 3
        # So start_idx = 4.
        
        def has_path_len_5(u, v, c):
            # Check if there is a simple path of length 5 from u to v in graph c
            # Use DFS with bitmasks
            # We want path u -> n1 -> n2 -> n3 -> n4 -> v
            # This is equivalent to:
            # Is there a node n2 such that:
            #   dist(u, n2) = 2 and dist(v, n2) = 3?
            #   OR dist(u, n2) = 3 and dist(v, n2) = 2?
            #   OR dist(u, n2) = 1 and dist(v, n2) = 4? (No, n2 must be internal)
            
            # Let's just do a simple DFS from u, depth 5, target v.
            # Optimization: Prune if visited.
            
            def dfs(node, depth, visited):
                if depth == 5:
                    return node == v
                if depth == 4:
                    # Check if v is a neighbor
                    if adj[c][node] & (1 << v):
                        return True
                    return False
                
                neighbors = adj[c][node] & ~visited
                while neighbors:
                    lsb = neighbors & -neighbors
                    nxt = lsb.bit_length() - 1
                    neighbors ^= lsb
                    if dfs(nxt, depth + 1, visited | lsb):
                        return True
                return False

            return dfs(u, 0, 1 << u)

        def backtrack(idx):
            if time.time() - start_time > 110:
                return False
            if idx == num_edges:
                return True
            
            u, v = edges[idx]
            
            # Heuristic: Try colors in a specific order?
            # Maybe try colors that have fewer constraints?
            # For now, 0, 1, 2, 3
            
            for c in range(4):
                if has_path_len_5(u, v, c):
                    continue
                
                set_edge(u, v, c, idx)
                
                if backtrack(idx + 1):
                    return True
                
                unset_edge(u, v, c, idx)
            
            return False

        if backtrack(start_idx):
            return {'n': n, 'colors': colors}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
