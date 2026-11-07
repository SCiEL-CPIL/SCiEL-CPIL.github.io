"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    m = graph['n'] // 2
    edges = graph['edges']
    
    # Classify edges
    # Vertices 0..m-1 are Top, m..2m-1 are Bottom
    # Vertical: |u - v| == m
    # Top: u < m and v < m
    # Bottom: u >= m and v >= m
    
    # We need to identify the index of each edge type to assign signs
    # Let's create arrays to hold the sign for each edge in the input list
    signs = np.ones(len(edges), dtype=np.int8)
    
    # We will construct the signs based on the "optimal" cycle product strategy
    # Strategy:
    # v_i (vertical) = 1
    # t_i (top) = -1 if i==0 else 1
    # b_i (bottom) depends on m parity to ensure P_bot = -1 and maximize anti-balanced squares
    
    # First, identify which edge corresponds to which structural edge
    # Structural edges:
    # Top edges: (i, i+1) for i in 0..m-1 (indices mod m)
    # Bottom edges: (m+i, m+i+1) for i in 0..m-1
    # Vertical edges: (i, m+i) for i in 0..m-1
    
    # Since the input edges are an ordered list, we don't know the order.
    # We need to map each input edge to its structural identity.
    
    top_edge_index = {} # (u, v) or (min, max) -> index in 'edges'
    bottom_edge_index = {}
    vertical_edge_index = {}
    
    for idx, (u, v) in enumerate(edges):
        u, v = min(u, v), max(u, v)
        if u < m and v < m:
            top_edge_index[(u, v)] = idx
        elif u >= m and v >= m:
            # Normalize to 0..m-1 range for key
            u_norm = u - m
            v_norm = v - m
            bottom_edge_index[(min(u_norm, v_norm), max(u_norm, v_norm))] = idx
        elif abs(u - v) == m:
            # Vertical edge
            # Key by the top vertex
            top_v = u if u < m else v
            vertical_edge_index[top_v] = idx
        else:
            # Should not happen for prism graph
            pass

    # Assign signs
    
    # 1. Vertical edges: all +1
    for v_idx in vertical_edge_index.values():
        signs[v_idx] = 1
        
    # 2. Top edges: t_0 = -1, others +1
    # Edge (0, 1) is t_0? No, t_i connects i and i+1.
    # t_0 connects 0 and 1.
    for i in range(m):
        u1 = i
        u2 = (i + 1) % m
        key = (min(u1, u2), max(u1, u2))
        if key in top_edge_index:
            if i == 0:
                signs[top_edge_index[key]] = -1
            else:
                signs[top_edge_index[key]] = 1
                
    # 3. Bottom edges
    # We determined:
    # If m even: b_0 = 1, b_i = -1 for i > 0.
    # If m odd: b_0 = 1, b_1 = 1, b_i = -1 for i > 1.
    
    if m % 2 == 0:
        # b_0 = 1, others -1
        for i in range(m):
            u1 = i
            u2 = (i + 1) % m
            key = (min(u1, u2), max(u1, u2))
            if key in bottom_edge_index:
                if i == 0:
                    signs[bottom_edge_index[key]] = 1
                else:
                    signs[bottom_edge_index[key]] = -1
    else:
        # b_0 = 1, b_1 = 1, others -1
        for i in range(m):
            u1 = i
            u2 = (i + 1) % m
            key = (min(u1, u2), max(u1, u2))
            if key in bottom_edge_index:
                if i == 0 or i == 1:
                    signs[bottom_edge_index[key]] = 1
                else:
                    signs[bottom_edge_index[key]] = -1

    return signs
# EVOLVE-BLOCK-END
