def sign_edges(graph, budget):
        import numpy as np
        import time
        
        start_time = time.time()
        deadline = start_time + 85.0
        
        n = graph['n']
        edges = graph['edges']
        m = n // 2
        E = len(edges)
        
        bot_idx = []
        top_idx = []
        match_idx = []
        
        for i, (u, v) in enumerate(edges):
            if u < m and v < m:
                bot_idx.append(i)
            elif u >= m and v >= m:
                top_idx.append(i)
            else:
                match_idx.append(i)
                
        # Precompute adjacency for unbalanced check
        # adj[u] = list of (v, edge_index)
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
            
        def is_unbalanced(signs):
            # signs is a numpy array of shape (E,)
            pot = np.zeros(n, dtype=np.int8)
            pot[0] = 1
            stack = [0]
            while stack:
                u = stack.pop()
                pu = pot[u]
                for v, ei in adj[u]:
                    s = signs[ei]
                    want = pu * s
                    if pot[v] == 0:
                        pot[v] = want
                        stack.append(v)
                    elif pot[v] != want:
                        return True
            return False
            
        def spectral_radius(signs):
            A = np.zeros((n, n))
            for i, (u, v) in enumerate(edges):
                s = signs[i]
                A[u, v] = s
                A[v, u] = s
            w = np.linalg.eigvalsh(A)
            return float(np.max(np.abs(w)))
            
        best_signs = None
        best_radius = float('inf')
        
        # Enumerate bottom and top patterns
        # We fix matching to +1
        for bot_mask in range(2 ** m):
            if time.time() > deadline:
                break
            # Construct bottom signs
            # bot_edge_indices[k] corresponds to edge k in the cycle?
            # The problem states edges are an ordered list.
            # We assume the edges in bot_idx are in cycle order 0-1, 1-2, ..., m-1-0.
            # Same for top.
            # If the order is different, we just map the mask bits to the edge indices.
            
            bottom_signs = np.ones(m, dtype=np.int8)
            for j in range(m):
                if (bot_mask >> j) & 1:
                    bottom_signs[j] = -1
                    
            for top_mask in range(2 ** m):
                if time.time() > deadline:
                    break
                top_signs = np.ones(m, dtype=np.int8)
                for j in range(m):
                    if (top_mask >> j) & 1:
                        top_signs[j] = -1
                        
                # Build full signs
                signs = np.ones(E, dtype=np.int8)
                for k, ei in enumerate(bot_idx):
                    signs[ei] = bottom_signs[k]
                for k, ei in enumerate(top_idx):
                    signs[ei] = top_signs[k]
                # Matching is already +1
                
                if not is_unbalanced(signs):
                    continue
                    
                r = spectral_radius(signs)
                if r < best_radius:
                    best_radius = r
                    best_signs = signs.copy()
                    
        if best_signs is None:
            # Fallback: all +1 is balanced (cycle product 1).
            # We need an unbalanced one.
            # Flip one edge in a cycle.
            best_signs = np.ones(E, dtype=np.int8)
            best_signs[bot_idx[0]] = -1
            # Check if this is unbalanced.
            # A cycle with one -1 is unbalanced.
            # So this should be valid.
            
        return np.asarray(best_signs, dtype=np.int8)
