import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n = graph['n']
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    # Identify edge categories
    bottom_cycle_edges = []
    top_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_cycle_edges.append(i)
        elif u >= m and v >= m:
            top_cycle_edges.append(i)
        else:
            matching_edges.append(i)
            
    # Sort them to ensure consistent ordering if needed, though indices are what matters
    # The problem doesn't guarantee specific order, so we rely on the list indices.
    
    def build_matrix(signs):
        A = np.zeros((n, n))
        for i in range(n_edges):
            s = signs[i]
            A[u_arr[i], v_arr[i]] = s
            A[v_arr[i], u_arr[i]] = s
        return A
            
    def spectral_radius(signs):
        A = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
        
    def is_valid(signs):
        try:
            check_lift(graph, signs)
            return True
        except Exception:
            return False
            
    best_signs = None
    best_radius = float('inf')
    
    # Generate candidates based on structural insights
    # We assume the optimal solution has minimal frustration:
    # Product of signs on bottom cycle = -1
    # Product of signs on top cycle = -1
    # This guarantees connectivity.
    
    candidates = []
    
    # Case 1: One -1 in bottom, one -1 in top, all matching +1
    for i in range(len(bottom_cycle_edges)):
        for j in range(len(top_cycle_edges)):
            signs = np.ones(n_edges, dtype=int)
            signs[bottom_cycle_edges[i]] = -1
            signs[top_cycle_edges[j]] = -1
            candidates.append(signs)
            
    # Case 2: One -1 in bottom, one -1 in top, one -1 in matching
    for i in range(len(bottom_cycle_edges)):
        for j in range(len(top_cycle_edges)):
            for k in range(len(matching_edges)):
                signs = np.ones(n_edges, dtype=int)
                signs[bottom_cycle_edges[i]] = -1
                signs[top_cycle_edges[j]] = -1
                signs[matching_edges[k]] = -1
                candidates.append(signs)
                
    # Case 3: All +1 except one -1 in bottom and one in matching (Top all +1)
    # This might not guarantee connectivity if the top cycle product is +1 and no other cycle has -1.
    # But the "ladder" cycles will have product -1 if the matching edge is -1 and bottom edge is -1?
    # Cycle: bottom edge i, matching i, matching j, top edge j?
    # If we only have one -1 in bottom and one in matching, the product on the bottom cycle is -1.
    # So connectivity is guaranteed.
    for i in range(len(bottom_cycle_edges)):
        for k in range(len(matching_edges)):
            signs = np.ones(n_edges, dtype=int)
            signs[bottom_cycle_edges[i]] = -1
            signs[matching_edges[k]] = -1
            candidates.append(signs)
            
    for j in range(len(top_cycle_edges)):
        for k in range(len(matching_edges)):
            signs = np.ones(n_edges, dtype=int)
            signs[top_cycle_edges[j]] = -1
            signs[matching_edges[k]] = -1
            candidates.append(signs)

    # Evaluate candidates
    for signs in candidates:
        if not is_valid(signs):
            continue
        radius = spectral_radius(signs)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    # Fallback if no valid candidate found (should not happen)
    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=int)
        # Force connectivity by flipping one edge in a cycle
        if bottom_cycle_edges:
             best_signs[bottom_cycle_edges[0]] = -1
             
    return best_signs
