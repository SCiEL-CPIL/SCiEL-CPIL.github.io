import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Identify edge indices for top cycle, bottom cycle, and matching
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
            
    # Sort matching edges by vertex index to ensure consistent ordering for cycle products
    # matching edge i connects i and i+m
    # We need to map edge index to the 'i' in (i, i+m)
    match_edge_to_i = {}
    for i in matching_edges:
        u, v = edges[i]
        if u < m:
            match_edge_to_i[i] = u
        else:
            match_edge_to_i[i] = v - m
            
    # Sort matching edges by their vertex index
    matching_edges_sorted = sorted(matching_edges, key=lambda idx: match_edge_to_i[idx])
    
    # Precompute edge indices for top and bottom cycles in order
    # Top cycle: 0-1, 1-2, ..., (m-1)-0
    # Bottom cycle: m-(m+1), ..., (2m-1)-m
    # We need to map vertex to edge index in the cycle
    # For simplicity, we will just compute products by iterating edges
    
    def is_connected_lift_fast(signs):
        """Check connectivity using cycle products."""
        # Product of top cycle
        p_top = 1
        for idx in top_cycle_edges:
            p_top *= signs[idx]
            
        # Product of bottom cycle
        p_bot = 1
        for idx in bottom_cycle_edges:
            p_bot *= signs[idx]
            
        if p_top == -1 or p_bot == -1:
            return True
            
        # Check if any "rung" cycle has product -1
        # Rung cycle i: i -> i+1 (top), i+1 -> i+1+m (match), i+1+m -> i+m (bottom), i+m -> i (match)
        # Edges:
        # Top edge (i, i+1)
        # Match edge (i+1, i+1+m)
        # Bottom edge (i+1+m, i+m) which is (i+1, i) in bottom cycle? No, bottom cycle is m..2m-1.
        # Bottom edge connects (i+m, (i+1)+m).
        # Match edge (i, i+m)
        
        # We need to find the edge indices for these specific connections.
        # This requires a lookup table.
        # Since m is small, we can build a map (u, v) -> edge_index once.
        
        # However, doing this inside the loop is slow.
        # Let's precompute the specific edge indices for the rung cycles.
        
        return False # Placeholder, will be replaced with precomputed logic
        
    # Precompute edge indices for rung cycles
    # We need:
    # top_edge[i]: index of edge (i, i+1) in top cycle
    # bottom_edge[i]: index of edge (i+m, i+1+m) in bottom cycle
    # match_edge[i]: index of edge (i, i+m)
    
    top_edge_idx = [-1] * m
    bottom_edge_idx = [-1] * m
    match_edge_idx = [-1] * m
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Top cycle edge
            if v == (u + 1) % m:
                top_edge_idx[u] = i
            else:
                top_edge_idx[v] = i
        elif u >= m and v >= m:
            # Bottom cycle edge
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                bottom_edge_idx[u_local] = i
            else:
                bottom_edge_idx[v_local] = i
        else:
            # Matching edge
            if u < m:
                match_edge_idx[u] = i
            else:
                match_edge_idx[v - m] = i
                
    def is_connected_lift_fast(signs):
        p_top = 1
        for i in range(m):
            p_top *= signs[top_edge_idx[i]]
            
        p_bot = 1
        for i in range(m):
            p_bot *= signs[bottom_edge_idx[i]]
            
        if p_top == -1 or p_bot == -1:
            return True
            
        for i in range(m):
            # Cycle: i -> i+1 (top), i+1 -> i+1+m (match), i+1+m -> i+m (bottom), i+m -> i (match)
            # Note: bottom edge is (i+m, i+1+m). In our bottom_edge_idx, it's stored at i or i+1?
            # bottom_edge_idx[k] is edge (k+m, k+1+m).
            # So bottom edge for rung i is bottom_edge_idx[i].
            # Top edge for rung i is top_edge_idx[i].
            # Match edges are match_edge_idx[i] and match_edge_idx[(i+1)%m].
            
            q_i = signs[top_edge_idx[i]] * signs[match_edge_idx[(i+1)%m]] * signs[bottom_edge_idx[i]] * signs[match_edge_idx[i]]
            if q_i == -1:
                return True
                
        return False

    def spectral_radius(signs):
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Initializations
    # 1. All +1
    signs = np.ones(num_edges, dtype=int)
    if is_connected_lift_fast(signs):
        r = spectral_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
            
    # 2. Alternating matching
    signs = np.ones(num_edges, dtype=int)
    for i in range(m):
        if i % 2 == 0:
            signs[match_edge_idx[i]] = -1
    if is_connected_lift_fast(signs):
        r = spectral_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
            
    # 3. Random
    for _ in range(100):
        signs = rng.choice([-1, 1], size=num_edges)
        if not is_connected_lift_fast(signs):
            continue
        r = spectral_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback
        best_signs = np.ones(num_edges, dtype=int)
        best_signs[match_edge_idx[0]] = -1
        best_radius = spectral_radius(best_signs)
        
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    temperature = 1.0
    cooling_rate = 0.999
    min_temperature = 0.01
    
    while time.time() < deadline and temperature > min_temperature:
        # Single flip
        edge_idx = rng.integers(0, num_edges)
        current_signs[edge_idx] = -current_signs[edge_idx]
        
        if not is_connected_lift_fast(current_signs):
            current_signs[edge_idx] = -current_signs[edge_idx]
            temperature *= cooling_rate
            continue
            
        new_radius = spectral_radius(current_signs)
        
        if new_radius < current_radius:
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_signs = current_signs.copy()
        else:
            delta = new_radius - current_radius
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_radius = new_radius
            else:
                current_signs[edge_idx] = -current_signs[edge_idx]
                
        temperature *= cooling_rate
        
    # Final polish
    if time.time() < deadline:
        polish_signs = best_signs.copy()
        polish_radius = best_radius
        for edge_idx in range(num_edges):
            if time.time() > deadline:
                break
            polish_signs[edge_idx] = -polish_signs[edge_idx]
            if not is_connected_lift_fast(polish_signs):
                polish_signs[edge_idx] = -polish_signs[edge_idx]
                continue
            r = spectral_radius(polish_signs)
            if r < polish_radius:
                polish_radius = r
            else:
                polish_signs[edge_idx] = -polish_signs[edge_idx]
                
        if polish_radius < best_radius:
            best_radius = polish_radius
            best_signs = polish_signs.copy()
            
    return best_signs
