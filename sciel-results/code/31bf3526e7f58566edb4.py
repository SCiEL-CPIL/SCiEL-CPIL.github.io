"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    # Identify edge indices for top cycle, bottom cycle, and matching
    # Map edge index to (type, pos)
    edge_map = {}
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Top cycle
            if v == (u + 1) % m:
                pos = u
            else:
                pos = v
            edge_map[idx] = ('top', pos)
        elif u >= m and v >= m:
            # Bottom cycle
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                pos = u_local
            else:
                pos = v_local
            edge_map[idx] = ('bottom', pos)
        else:
            # Matching
            if u < m:
                pos = u
            else:
                pos = v - m
            edge_map[idx] = ('match', pos)

    # Identify Free Edges
    # Tree edges:
    # - Match edge for pos 0
    # - Top edges for pos 0 to m-2
    # - Bottom edges for pos 0 to m-2
    
    free_indices = []
    free_types = []
    free_pos = []
    
    for idx in range(n_edges):
        etype, pos = edge_map[idx]
        is_in_tree = False
        if etype == 'match' and pos == 0:
            is_in_tree = True
        elif etype == 'top' and pos < m - 1:
            is_in_tree = True
        elif etype == 'bottom' and pos < m - 1:
            is_in_tree = True
            
        if not is_in_tree:
            free_indices.append(idx)
            free_types.append(etype)
            free_pos.append(pos)
            
    n_free = len(free_indices)
    # n_free should be m + 1

    # Pre-allocate adjacency matrix structure
    # We will modify the signs in a pre-allocated matrix
    A_base = np.zeros((n_vertices, n_vertices))
    # Store the base adjacency without signs, just to know where edges are
    # Actually, we just need to set A[u,v] = sign and A[v,u] = sign.
    # We can pre-store the pairs (u, v) for each edge index.
    edge_u = np.array([u for u, v in edges])
    edge_v = np.array([v for u, v in edges])

    best_signs = None
    best_radius = float('inf')

    num_classes = 1 << n_free
    
    # Helper to check if unbalanced
    # Balanced if:
    # 1. Top closing sign (+1)
    # 2. Bottom closing sign (+1)
    # 3. All matching signs are equal (since product of adjacent matches must be +1)
    
    # Identify which free indices correspond to what
    top_close_idx = -1
    bot_close_idx = -1
    match_free_indices = {} # pos -> index in free_indices
    
    for i, idx in enumerate(free_indices):
        etype, pos = edge_map[idx]
        if etype == 'top' and pos == m - 1:
            top_close_idx = i
        elif etype == 'bottom' and pos == m - 1:
            bot_close_idx = i
        elif etype == 'match':
            match_free_indices[pos] = i

    for mask in range(num_classes):
        if time.time() > DEADLINE:
            break
            
        # Extract signs for free edges
        # We need to check balance condition quickly
        is_balanced = True
        
        # 1. Check top closing
        if top_close_idx != -1:
            if mask & (1 << top_close_idx):
                is_balanced = False # Sign is -1
            # else sign is +1
        
        # 2. Check bottom closing
        if is_balanced and bot_close_idx != -1:
            if mask & (1 << bot_close_idx):
                is_balanced = False # Sign is -1
        
        # 3. Check matching signs
        if is_balanced:
            # All matching free edges must have the same sign
            # Get the sign of the first matching free edge (pos 1)
            if 1 in match_free_indices:
                first_match_sign = -1 if (mask & (1 << match_free_indices[1])) else 1
                for pos in range(2, m):
                    if pos in match_free_indices:
                        curr_match_sign = -1 if (mask & (1 << match_free_indices[pos])) else 1
                        if curr_match_sign != first_match_sign:
                            is_balanced = False
                            break
                            
        if is_balanced:
            continue
            
        # Construct signs array
        signs = np.ones(n_edges, dtype=np.int8)
        for i, idx in enumerate(free_indices):
            if mask & (1 << i):
                signs[idx] = -1
        
        # Construct adjacency matrix
        A = np.zeros((n_vertices, n_vertices))
        # Vectorized assignment is faster
        # A[edge_u, edge_v] = signs
        # A[edge_v, edge_u] = signs
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        
        try:
            vals = np.linalg.eigvalsh(A)
            r = np.max(np.abs(vals))
        except:
            continue

        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
            
    # Fallback if no valid unbalanced graph found (should not happen)
    if best_signs is None:
        # Try a known unbalanced pattern: one -1 on a matching edge
        # This makes the graph unbalanced
        signs = np.ones(n_edges, dtype=np.int8)
        # Find a matching edge
        for idx in range(n_edges):
            if edge_map[idx][0] == 'match':
                signs[idx] = -1
                break
        A = np.zeros((n_vertices, n_vertices))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        try:
            vals = np.linalg.eigvalsh(A)
            best_radius = np.max(np.abs(vals))
            best_signs = signs
        except:
            best_signs = np.ones(n_edges, dtype=np.int8)
            best_signs[0] = -1

    return best_signs
# EVOLVE-BLOCK-END
