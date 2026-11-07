# EVOLVE-BLOCK-START
import numpy as np
import time

def _spectral_radius_signed(m, bottom_signs, top_signs, matching_signs):
    """Compute spectral radius of signed adjacency matrix of prism graph."""
    n = 2 * m
    A = np.zeros((n, n), dtype=float)
    
    # Bottom cycle: vertices 0 to m-1
    for i in range(m):
        j = (i + 1) % m
        s = float(bottom_signs[i])
        A[i, j] = s
        A[j, i] = s
    
    # Top cycle: vertices m to 2m-1
    for i in range(m):
        j = m + (i + 1) % m
        s = float(top_signs[i])
        A[m + i, j] = s
        A[j, m + i] = s
    
    # Matching: vertex i to vertex i+m
    for i in range(m):
        s = float(matching_signs[i])
        A[i, m + i] = s
        A[m + i, i] = s
    
    eigvals = np.linalg.eigvalsh(A)
    return max(abs(eigvals[0]), abs(eigvals[-1]))

def _is_valid_lift_fast(m, bottom_signs, top_signs, matching_signs):
    """
    Fast validity check for the 2-lift of a prism graph.
    The lift is disconnected iff the signed graph is gauge-equivalent to the all-+1 graph.
    This happens iff the product of signs around every cycle is +1.
    Cycles in prism:
    1. Bottom cycle
    2. Top cycle
    3. Ladder cycles (for each i, cycle i-bottom -> i+1-bottom -> i+1-top -> i-top -> i-bottom)
    """
    prod_b = 1
    for s in bottom_signs:
        prod_b *= s
    if prod_b == -1:
        return True
        
    prod_t = 1
    for s in top_signs:
        prod_t *= s
    if prod_t == -1:
        return True
        
    # If both main cycles are +1, check ladders
    for i in range(m):
        # Ladder cycle involves:
        # Bottom edge i (connecting i and i+1)
        # Matching edge i+1 (connecting i+1 and i+1+m)
        # Top edge i (connecting i+m and i+1+m) -- Note: Top edge i connects m+i and m+i+1
        # Matching edge i (connecting i and i+m)
        
        # Signs:
        s_b = bottom_signs[i]
        s_m_next = matching_signs[(i + 1) % m]
        s_t = top_signs[i]
        s_m_cur = matching_signs[i]
        
        prod_l = s_b * s_m_next * s_t * s_m_cur
        if prod_l == -1:
            return True
            
    return False

def _extract_cycle_signs(graph, m):
    """Extract edge maps for bottom cycle, top cycle, and matching."""
    edges = graph['edges']
    bottom_edge_map = {}
    top_edge_map = {}
    matching_edge_map = {}
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            a, b = min(u, v), max(u, v)
            if b == (a + 1) % m:
                pos = a
            else:
                pos = b
            bottom_edge_map[i] = pos
        elif u >= m and v >= m:
            a, b = min(u - m, v - m), max(u - m, v - m)
            if b == (a + 1) % m:
                pos = a
            else:
                pos = b
            top_edge_map[i] = pos
        else:
            pos = min(u, v) % m
            matching_edge_map[i] = pos
    
    return bottom_edge_map, top_edge_map, matching_edge_map

def _signs_to_components(signs, bottom_edge_map, top_edge_map, matching_edge_map, m):
    """Convert full signs array to (bottom_signs, top_signs, matching_signs)."""
    bottom_signs = np.ones(m, dtype=int)
    top_signs = np.ones(m, dtype=int)
    matching_signs = np.ones(m, dtype=int)
    
    for edge_idx, pos in bottom_edge_map.items():
        bottom_signs[pos] = signs[edge_idx]
    for edge_idx, pos in top_edge_map.items():
        top_signs[pos] = signs[edge_idx]
    for edge_idx, pos in matching_edge_map.items():
        matching_signs[pos] = signs[edge_idx]
    
    return bottom_signs, top_signs, matching_signs

def _local_search(graph, signs, deadline, max_rounds=100):
    """Local search: flip one edge at a time to improve spectral radius."""
    m = graph['n'] // 2
    bottom_edge_map, top_edge_map, matching_edge_map = _extract_cycle_signs(graph, m)
    
    current_signs = signs.copy()
    b_s, t_s, m_s = _signs_to_components(current_signs, bottom_edge_map, top_edge_map, matching_edge_map, m)
    current_radius = _spectral_radius_signed(m, b_s, t_s, m_s)
    num_edges = len(current_signs)
    
    # Precompute edge indices for each component type for faster access?
    # We already have maps from edge_idx to pos.
    # We need to flip signs in the array and update components.
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        
        # Single flip
        for i in range(num_edges):
            if time.time() > deadline:
                break
            
            trial = current_signs.copy()
            trial[i] = -trial[i]
            
            # Fast validity check
            b_t, t_t, m_t = _signs_to_components(trial, bottom_edge_map, top_edge_map, matching_edge_map, m)
            if not _is_valid_lift_fast(m, b_t, t_t, m_t):
                continue
                
            r = _spectral_radius_signed(m, b_t, t_t, m_t)
            if r < current_radius - 1e-12:
                current_signs = trial
                b_s, t_s, m_s = b_t, t_t, m_t
                current_radius = r
                improved = True
        
        if not improved:
            break
    
    return current_signs, current_radius

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    bottom_edge_map, top_edge_map, matching_edge_map = _extract_cycle_signs(graph, m)
    
    best_signs = None
    best_radius = np.inf
    
    # Strategy:
    # 1. Generate a pool of valid candidates using random restarts and heuristic patterns.
    # 2. Evaluate them.
    # 3. Run local search on the top ones.
    
    candidates = []
    
    # Heuristic: Frustrate the bottom cycle
    for b_pos in range(m):
        s = np.ones(num_edges, dtype=int)
        for edge_idx, pos in bottom_edge_map.items():
            if pos == b_pos:
                s[edge_idx] = -1
                break
        candidates.append(s)
        
    # Heuristic: Frustrate the top cycle
    for t_pos in range(m):
        s = np.ones(num_edges, dtype=int)
        for edge_idx, pos in top_edge_map.items():
            if pos == t_pos:
                s[edge_idx] = -1
                break
        candidates.append(s)
        
    # Heuristic: Frustrate a ladder
    for i in range(m):
        s = np.ones(num_edges, dtype=int)
        # Flip matching edge i
        for edge_idx, pos in matching_edge_map.items():
            if pos == i:
                s[edge_idx] = -1
                break
        candidates.append(s)
        
    # Random candidates
    rng = np.random.default_rng(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        s = rng.choice([-1, 1], size=num_edges).astype(int)
        candidates.append(s)
        
    # Filter valid candidates and compute radius
    valid_candidates = []
    for s in candidates:
        if time.time() > deadline:
            break
        b_s, t_s, m_s = _signs_to_components(s, bottom_edge_map, top_edge_map, matching_edge_map, m)
        if _is_valid_lift_fast(m, b_s, t_s, m_s):
            r = _spectral_radius_signed(m, b_s, t_s, m_s)
            valid_candidates.append((r, s))
            
    valid_candidates.sort(key=lambda x: x[0])
    
    if valid_candidates:
        best_radius, best_signs = valid_candidates[0]
        
        # Local search on top 10 candidates
        for r, s in valid_candidates[:10]:
            if time.time() > deadline - 5.0:
                break
            refined_signs, refined_radius = _local_search(graph, s, deadline, max_rounds=50)
            if refined_radius < best_radius:
                best_radius = refined_radius
                best_signs = refined_signs.copy()
    else:
        # Fallback: Find any valid sign
        # If no candidate was valid, it's unlikely but possible.
        # Try all +1? No, all +1 is invalid (disconnected).
        # Try flipping one edge.
        for i in range(num_edges):
            s = np.ones(num_edges, dtype=int)
            s[i] = -1
            b_s, t_s, m_s = _signs_to_components(s, bottom_edge_map, top_edge_map, matching_edge_map, m)
            if _is_valid_lift_fast(m, b_s, t_s, m_s):
                r = _spectral_radius_signed(m, b_s, t_s, m_s)
                best_radius = r
                best_signs = s
                break
                
        if best_signs is None:
            # Emergency fallback: random search
            for _ in range(1000):
                s = rng.choice([-1, 1], size=num_edges).astype(int)
                b_s, t_s, m_s = _signs_to_components(s, bottom_edge_map, top_edge_map, matching_edge_map, m)
                if _is_valid_lift_fast(m, b_s, t_s, m_s):
                    r = _spectral_radius_signed(m, b_s, t_s, m_s)
                    best_radius = r
                    best_signs = s
                    break
                    
    if best_signs is None:
        # Should not happen, but provide a default
        best_signs = np.ones(num_edges, dtype=int)
        best_signs[0] = -1

    return best_signs.astype(int)
# EVOLVE-BLOCK-END
