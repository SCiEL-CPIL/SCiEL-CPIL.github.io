import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n = 2 * m
    
    # Identify edge roles and build spanning tree
    # Tree: top path (m-1 edges), bottom path (m-1 edges), matching edges except match_0 (m-1 edges)
    # Non-tree: top_close, bot_close, match_0
    # Wait, that's only 3 non-tree edges, but we need m+1.
    
    # Let me reconsider. Total edges = 3m. Tree edges = 2m-1. Non-tree = m+1.
    # 
    # Let's use this tree:
    # - All matching edges: m edges
    # - Top path: (0,1), (1,2), ..., (m-2,m-1): m-1 edges
    # - Bottom path: (m,m+1), ..., (2m-2,2m-1): m-1 edges
    # Total tree: m + (m-1) + (m-1) = 3m-2
    # But we need 2m-1 tree edges. 3m-2 > 2m-1 for m>1.
    #
    # I need to remove (3m-2)-(2m-1) = m-1 edges from this set.
    # 
    # Alternative: use a proper spanning tree.
    # Start with vertex 0. Add edge (0,1), (0,m). Now we have {0,1,m}.
    # This is getting complicated. Let me just identify all edges and pick a spanning tree via BFS.
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[int(u)].append((int(v), idx))
        adj[int(v)].append((int(u), idx))
    
    # BFS to find spanning tree
    from collections import deque
    visited = [False] * n
    tree_edges = set()
    queue = deque([0])
    visited[0] = True
    while queue:
        u = queue.popleft()
        for v, idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(idx)
                queue.append(v)
    
    non_tree_indices = [i for i in range(n_edges) if i not in tree_edges]
    num_non_tree = len(non_tree_indices)
    
    # Verify we have the right count
    if num_non_tree != m + 1:
        # Fallback: just use a simple pattern
        signs = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if abs(int(u) - int(v)) == m:
                top = int(u) if int(u) < m else int(v)
                if top % 2 == 0:
                    signs[i] = -1
        return signs
    
    # Phase 1: Fix tree edges to +1, search over non-tree edges
    # Build base matrix with tree edges = +1
    A_tree = np.zeros((n, n), dtype=float)
    for idx in tree_edges:
        u, v = edges[idx]
        A_tree[int(u), int(v)] = 1.0
        A_tree[int(v), int(u)] = 1.0
    
    best_radius = float('inf')
    best_signs = None
    
    # Search over all 2^(num_non_tree) combinations
    total_combos = 1 << num_non_tree
    for mask in range(total_combos):
        if time.time() > deadline:
            break
        
        # Connectivity: at least one non-tree edge must have sign -1
        # But wait, this is only true if tree edges are all +1.
        # With tree edges fixed to +1, the cycle product for non-tree edge j is just its sign.
        # So we need at least one non-tree sign to be -1.
        if mask == 0:
            continue
        
        # Build signed adjacency matrix
        A_signed = A_tree.copy()
        for j, idx in enumerate(non_tree_indices):
            if (mask >> j) & 1:
                s = -1.0
            else:
                s = 1.0
            u, v = edges[idx]
            A_signed[int(u), int(v)] = s
            A_signed[int(v), int(u)] = s
        
        # Compute spectral radius
        eigenvalues = np.linalg.eigvalsh(A_signed)
        radius = np.max(np.abs(eigenvalues))
        
        if radius < best_radius:
            best_radius = radius
            signs = np.ones(n_edges, dtype=int)
            for j, idx in enumerate(non_tree_indices):
                if (mask >> j) & 1:
                    signs[idx] = -1
            best_signs = signs
    
    if best_signs is None:
        # Fallback
        signs = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if abs(int(u) - int(v)) == m:
                top = int(u) if int(u) < m else int(v)
                if top % 2 == 0:
                    signs[i] = -1
        return signs
    
    # Phase 2: Local search - try flipping each edge and see if it improves
    current_signs = best_signs.copy()
    current_radius = best_radius
    improved = True
    max_iterations = 100
    
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n_edges):
            if time.time() > deadline:
                break
            
            # Flip edge idx
            old_sign = current_signs[idx]
            current_signs[idx] = -old_sign
            
            # Build signed adjacency matrix
            A_signed = np.zeros((n, n), dtype=float)
            for i, (u, v) in enumerate(edges):
                s = float(current_signs[i])
                A_signed[int(u), int(v)] = s
                A_signed[int(v), int(u)] = s
            
            # Check connectivity using the sign product method
            # For the lift to be connected, we need at least one cycle with product -1
            # This is equivalent to: the signed graph is not "balanced"
            # A signed graph is balanced if every cycle has product +1
            # For 2-lift connectivity: the lift is connected iff the graph is not balanced
            # i.e., there exists a cycle with product -1
            
            # Check if balanced: assign potentials
            # c[v] = product of signs along path from root to v
            # If for every edge (u,v), c[u]*c[v]*s(u,v) = 1, then balanced
            c = np.zeros(n, dtype=int)
            c[0] = 1
            visited = [False] * n
            visited[0] = True
            queue = deque([0])
            balanced = True
            while queue and balanced:
                u = queue.popleft()
                for v, eidx in adj[u]:
                    s = int(current_signs[eidx])
                    if not visited[v]:
                        c[v] = c[u] * s
                        visited[v] = True
                        queue.append(v)
                    else:
                        # Check consistency
                        if c[u] * c[v] * s != 1:
                            balanced = False
                            break
            
            if balanced:
                # Disconnected lift, revert
                current_signs[idx] = old_sign
                continue
            
            # Compute spectral radius
            eigenvalues = np.linalg.eigvalsh(A_signed)
            radius = np.max(np.abs(eigenvalues))
            
            if radius < current_radius:
                current_radius = radius
                best_signs = current_signs.copy()
                improved = True
            else:
                # Revert
                current_signs[idx] = old_sign
    
    return best_signs
