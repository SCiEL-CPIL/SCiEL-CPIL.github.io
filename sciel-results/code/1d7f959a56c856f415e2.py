import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 60s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Identify edge types
    bottom_edges = []  # edges in bottom cycle (indices into edges list)
    top_edges = []     # edges in top cycle
    match_edges = []   # matching edges
    bottom_order = {}  # position in bottom cycle
    top_order = {}     # position in top cycle
    match_pos = {}     # position in matching

    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_edges.append(i)
            # Determine position in cycle
            a, b = min(u, v), max(u, v)
            if b == a + 1 or (a == 0 and b == m - 1):
                bottom_order[i] = a
            else:
                bottom_order[i] = a
        elif u >= m and v >= m:
            top_edges.append(i)
            a, b = min(u, v), max(u, v)
            a_local, b_local = a - m, b - m
            if b_local == a_local + 1 or (a_local == 0 and b_local == m - 1):
                top_order[i] = a_local
            else:
                top_order[i] = a_local
        else:
            match_edges.append(i)
            pos = min(u, v) if min(u, v) < m else min(u, v) - m
            match_pos[i] = pos

    def compute_radius_fast(bottom_signs, top_signs, match_signs):
        """Compute spectral radius using block structure."""
        # Build the signed adjacency matrix
        A = np.zeros((n, n), dtype=np.float64)
        for i in bottom_edges:
            u, v = edges[i]
            A[u, v] = bottom_signs[i]
            A[v, u] = bottom_signs[i]
        for i in top_edges:
            u, v = edges[i]
            A[u, v] = top_signs[i]
            A[v, u] = top_signs[i]
        for i in match_edges:
            u, v = edges[i]
            A[u, v] = match_signs[i]
            A[v, u] = match_signs[i]
        
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_connected_lift(bottom_signs, top_signs, match_signs):
        """Check if the 2-lift is connected."""
        # The lift is connected iff the holonomy is non-trivial
        # Product around bottom cycle
        prod_bottom = 1
        for i in bottom_edges:
            prod_bottom *= bottom_signs[i]
        
        # Product around top cycle
        prod_top = 1
        for i in top_edges:
            prod_top *= top_signs[i]
        
        # For the lift to be connected, at least one cycle must have product -1
        # Actually, we need to check the fundamental group generators
        # The prism graph has rank 2 (two independent cycles)
        # The lift is connected iff the map from pi_1 to Z_2 is surjective
        # This happens iff at least one of the two fundamental cycles has product -1
        
        return prod_bottom == -1 or prod_top == -1

    def get_signs_array(bottom_signs, top_signs, match_signs):
        """Convert separate sign arrays to the full edge sign array."""
        signs = np.zeros(E, dtype=np.int8)
        for i in bottom_edges:
            signs[i] = bottom_signs[i]
        for i in top_edges:
            signs[i] = top_signs[i]
        for i in match_edges:
            signs[i] = match_signs[i]
        return signs

    # Strategy: enumerate all combinations of cycle products
    # The spectral radius depends on the products, not individual signs (by gauge invariance)
    # Fix the gauge: set all but one edge in each cycle to +1
    # Then the product is determined by the last edge's sign
    
    # For each combination of (prod_bottom, prod_top, and the "twist" pattern),
    # compute the spectral radius
    
    # Actually, let's just do a smarter search:
    # The key parameters are the products around the two cycles
    # And the relative signs between matching, bottom, and top edges
    
    # Let me use the fact that we can fix a spanning tree
    # Spanning tree: all matching edges + first m-1 bottom edges
    # Free edges: last bottom edge + all m top edges = m+1 free edges
    
    # But a better approach: exploit the structure
    # The signed adjacency matrix of the prism has eigenvalues that come from
    # the eigenvalues of a 2x2 block matrix for each "mode"
    
    # For each k = 0, ..., m-1, the eigenvalues satisfy:
    # lambda^2 - (w_k * (s_k + t_k)) * lambda + (s_k * t_k * w_k^2 - 1) = 0
    # where w_k, s_k, t_k are related to the signs
    
    # Actually, let me just do an efficient exhaustive search with early termination
    
    best_radius = np.inf
    best_signs = None
    
    # Use the structure: fix gauge by setting matching edges to +1
    # Then the free variables are the bottom and top cycle signs
    # But we need to ensure connectivity
    
    # Simpler: enumerate all 2^(2m) possibilities for bottom and top cycle signs
    # with matching fixed to +1, but that's too many for m=9
    
    # Better: the spectral radius depends only on the cycle products and the 
    # "relative" signs. Let me parameterize by:
    # - prod_bottom = product of bottom signs
    # - prod_top = product of top signs  
    # - For each i, the "twist" w_i = match_sign[i] * bottom_sign[i] * top_sign[i]
    #   (but these aren't independent)
    
    # Let me just do a random + local search approach
    
    # Initialize with random valid signs
    rng = np.random.default_rng(3)
    
    def random_valid_signs():
        """Generate random signs that give a connected lift."""
        while True:
            b_signs = rng.choice([-1, 1], size=len(bottom_edges))
            t_signs = rng.choice([-1, 1], size=len(top_edges))
            m_signs = rng.choice([-1, 1], size=len(match_edges))
            
            # Check connectivity
            prod_b = np.prod(b_signs)
            prod_t = np.prod(t_signs)
            
            if prod_b == -1 or prod_t == -1:
                return b_signs, t_signs, m_signs
    
    # Local search: flip one edge at a time
    def local_search(b_signs, t_signs, m_signs, max_iter=10000):
        """Local search to minimize spectral radius."""
        current_b = b_signs.copy()
        current_t = t_signs.copy()
        current_m = m_signs.copy()
        
        current_radius = compute_radius_fast(current_b, current_t, current_m)
        
        all_edge_indices = bottom_edges + top_edges + match_edges
        all_signs = {}
        for i in bottom_edges:
            all_signs[i] = current_b[i]
        for i in top_edges:
            all_signs[i] = current_t[i]
        for i in match_edges:
            all_signs[i] = current_m[i]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try flipping each edge
            for edge_idx in all_edge_indices:
                old_sign = all_signs[edge_idx]
                new_sign = -old_sign
                all_signs[edge_idx] = new_sign
                
                # Update the sign arrays
                if edge_idx in bottom_edges:
                    current_b[edge_idx] = new_sign
                elif edge_idx in top_edges:
                    current_t[edge_idx] = new_sign
                else:
                    current_m[edge_idx] = new_sign
                
                # Check connectivity
                prod_b = np.prod(current_b)
                prod_t = np.prod(current_t)
                
                if prod_b == -1 or prod_t == -1:
                    new_radius = compute_radius_fast(current_b, current_t, current_m)
                    if new_radius < current_radius:
                        current_radius = new_radius
                        improved = True
                    else:
                        # Revert
                        all_signs[edge_idx] = old_sign
                        if edge_idx in bottom_edges:
                            current_b[edge_idx] = old_sign
                        elif edge_idx in top_edges:
                            current_t[edge_idx] = old_sign
                        else:
                            current_m[edge_idx] = old_sign
                else:
                    # Revert (disconnected)
                    all_signs[edge_idx] = old_sign
                    if edge_idx in bottom_edges:
                        current_b[edge_idx] = old_sign
                    elif edge_idx in top_edges:
                        current_t[edge_idx] = old_sign
                    else:
                        current_m[edge_idx] = old_sign
            
            if not improved:
                break
        
        return current_b, current_t, current_m, current_radius
    
    # Multiple restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        b_signs, t_signs, m_signs = random_valid_signs()
        b_opt, t_opt, m_opt, radius = local_search(b_signs, t_signs, m_signs, max_iter=200)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = get_signs_array(b_opt, t_opt, m_opt)
    
    # If no valid signs found, use all +1 (but this gives disconnected lift for even m)
    # For safety, use a known valid configuration
    if best_signs is None:
        # Set all bottom to +1 except last to -1 (gives prod_bottom = -1)
        b_signs = np.ones(len(bottom_edges), dtype=np.int8)
        b_signs[-1] = -1
        t_signs = np.ones(len(top_edges), dtype=np.int8)
        m_signs = np.ones(len(match_edges), dtype=np.int8)
        best_signs = get_signs_array(b_signs, t_signs, m_signs)
    
    return best_signs
