"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    """
    Assign signs to edges of a prism graph to minimize the spectral radius
    of the signed adjacency matrix, while ensuring the 2-lift is connected.
    
    Strategy: For each cycle in the prism (top m-cycle, bottom m-cycle, and
    the 'ladder' cycles formed by matching edges), we want the product of signs
    around the cycle to be -1 if possible to minimize spectral radius.
    
    For a prism graph with m-cycles:
    - Top cycle: edges (i, i+1) for i=0..m-1 (mod m)
    - Bottom cycle: edges (m+i, m+i+1) for i=0..m-1 (mod m)
    - Matching: edges (i, m+i) for i=0..m-1
    
    The signed adjacency matrix of the base graph has eigenvalues that are
    related to the lift. We want to minimize the max absolute eigenvalue.
    
    Key insight: For a cycle of length m with all +1 signs, the spectral radius
    is 2. If we flip signs to make the product -1 around the cycle, we can
    potentially reduce the spectral radius further or keep it optimal.
    
    For the prism graph, we have two m-cycles and m "ladder" cycles of length 2
    (well, actually the ladder cycles are formed by pairs of matching edges and
    corresponding cycle edges).
    
    Let's think about this differently. The signed adjacency matrix A of the
    prism graph has entries:
    - A[i][i+1] = s_top[i] for the top cycle
    - A[m+i][m+i+1] = s_bottom[i] for the bottom cycle  
    - A[i][m+i] = s_match[i] for the matching
    
    The eigenvalues of A determine the spectral radius.
    
    For a 2-lift to be connected, the product of signs around any fundamental
    cycle must be -1 (i.e., the sign vector must be "frustrated" in the right
    way).
    
    Actually, for the 2-lift to be connected, we need that there exists a path
    in the base graph such that the product of signs along the path is -1.
    This is equivalent to saying that the sign assignment is not "bipartite"
    in a certain sense.
    
    Let's use a more systematic approach:
    1. Enumerate possible sign patterns for the matching edges (2^m possibilities,
       but m is at most 9, so 512 max).
    2. For each matching sign pattern, determine optimal signs for the top and
       bottom cycles to minimize spectral radius while ensuring connectivity.
    
    Actually, let's think about the structure more carefully.
    
    The prism graph P_m has vertices 0..m-1 (top) and m..2m-1 (bottom).
    Edges:
    - Top cycle: (i, (i+1)%m) for i=0..m-1
    - Bottom cycle: (m+i, m+(i+1)%m) for i=0..m-1
    - Matching: (i, m+i) for i=0..m-1
    
    For the 2-lift to be connected, we need the sign vector to define a
    connected 2-lift. A sufficient condition is that the product of signs
    around some cycle is -1.
    
    Let's try a direct optimization approach:
    - For small m (4, 5, 7, 9), we can afford to search over sign patterns.
    - The total number of edges is 3m (m top + m bottom + m matching).
    - For m=9, that's 27 edges, so 2^27 is too many to enumerate.
    
    Better approach: Use the structure of the prism graph.
    
    Observation: The signed adjacency matrix of the prism graph can be analyzed
    by considering the symmetry. If we set all top cycle signs equal, all bottom
    cycle signs equal, and all matching signs equal, we get a highly symmetric
    case.
    
    Let's try:
    - s_top = t for all top edges
    - s_bottom = b for all bottom edges
    - s_match = r for all matching edges
    
    Then the signed adjacency matrix has a block structure. The eigenvalues
    can be computed analytically.
    
    Actually, let's just do a local search / simulated annealing style approach
    with a good initialization.
    
    Here's my plan:
    1. Initialize with a pattern that makes the 2-lift connected.
    2. Use local search: flip individual edge signs and accept if spectral radius
       decreases and the lift remains connected.
    3. Run for a fixed number of iterations or until time limit.
    """
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge index for quick lookup
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    def build_signed_adjacency(signs):
        """Build the signed adjacency matrix from edge signs."""
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        """Compute spectral radius of the signed adjacency matrix."""
        A = build_signed_adjacency(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        """Check if the 2-lift is connected."""
        try:
            radius, details = check_lift(graph, signs)
            return True
        except Exception:
            return False
    
    # Initial sign assignment: make it connected
    # Strategy: set all signs to +1 except one edge to make the product -1
    # around a fundamental cycle.
    
    # Let's try a structured initialization:
    # Set matching signs to +1, and alternate signs on one of the cycles
    # to create frustration.
    
    def initial_signs():
        signs = np.ones(num_edges, dtype=int)
        # Find the top cycle edges and alternate their signs
        # Top cycle: (i, (i+1)%m) for i=0..m-1
        top_edge_indices = []
        for i in range(m):
            u = i
            v = (i + 1) % m
            idx = edge_index.get((u, v)) or edge_index.get((v, u))
            if idx is not None:
                top_edge_indices.append(idx)
        
        # Alternate signs on top cycle to create frustration
        for j, idx in enumerate(top_edge_indices):
            if j % 2 == 1:
                signs[idx] = -1
        
        return signs
    
    def local_search(signs, max_iters=5000):
        """Local search to minimize spectral radius while maintaining connectivity."""
        signs = signs.copy()
        current_radius = spectral_radius(signs)
        
        if not is_connected_lift(signs):
            return None
        
        for it in range(max_iters):
            # Try flipping each edge
            best_idx = -1
            best_radius = current_radius
            best_signs = None
            
            for idx in range(num_edges):
                new_signs = signs.copy()
                new_signs[idx] = -new_signs[idx]
                
                if not is_connected_lift(new_signs):
                    continue
                
                r = spectral_radius(new_signs)
                if r < best_radius:
                    best_radius = r
                    best_idx = idx
                    best_signs = new_signs
            
            if best_idx >= 0:
                signs = best_signs
                current_radius = best_radius
            else:
                # Stuck in local minimum, try random restarts
                break
        
        return signs, current_radius
    
    # Run local search from multiple initializations
    best_signs = None
    best_radius = float('inf')
    
    # Try different initializations
    initializations = []
    
    # Init 1: Alternate signs on top cycle
    s1 = initial_signs()
    initializations.append(s1)
    
    # Init 2: Alternate signs on bottom cycle
    s2 = np.ones(num_edges, dtype=int)
    bottom_edge_indices = []
    for i in range(m):
        u = m + i
        v = m + (i + 1) % m
        idx = edge_index.get((u, v)) or edge_index.get((v, u))
        if idx is not None:
            bottom_edge_indices.append(idx)
    for j, idx in enumerate(bottom_edge_indices):
        if j % 2 == 1:
            s2[idx] = -1
    initializations.append(s2)
    
    # Init 3: All +1 except one matching edge
    s3 = np.ones(num_edges, dtype=int)
    match_edge_indices = []
    for i in range(m):
        u = i
        v = m + i
        idx = edge_index.get((u, v)) or edge_index.get((v, u))
        if idx is not None:
            match_edge_indices.append(idx)
    if match_edge_indices:
        s3[match_edge_indices[0]] = -1
    initializations.append(s3)
    
    # Init 4: Random with constraint
    rng = np.random.default_rng(42)
    s4 = rng.choice([-1, 1], size=num_edges).astype(int)
    # Ensure connectivity by checking and fixing if needed
    if not is_connected_lift(s4):
        # Try to fix by flipping one edge
        for idx in range(num_edges):
            s4_try = s4.copy()
            s4_try[idx] = -s4_try[idx]
            if is_connected_lift(s4_try):
                s4 = s4_try
                break
    initializations.append(s4)
    
    for init_signs in initializations:
        result = local_search(init_signs, max_iters=2000)
        if result is not None:
            signs, radius = result
            if radius < best_radius:
                best_radius = radius
                best_signs = signs
    
    # If no valid solution found, fall back to a simple valid one
    if best_signs is None:
        # Use the seed's approach
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)
    
    return best_signs
# EVOLVE-BLOCK-END
