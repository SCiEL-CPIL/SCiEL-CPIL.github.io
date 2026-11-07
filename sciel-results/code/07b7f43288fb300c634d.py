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
    # Bottom cycle: u < m, v < m
    # Top cycle: u >= m, v >= m
    # Matching: one < m, one >= m
    
    # Build adjacency structure for fast spectral radius computation
    # The signed adjacency matrix A has A[u,v] = sign of edge (u,v)
    # Spectral radius = max |eigenvalue|
    
    # Pre-compute edge indices by type for faster manipulation
    bottom_edges = []  # edges in bottom cycle
    top_edges = []     # edges in top cycle
    match_edges = []   # matching edges
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_edges.append(i)
        elif u >= m and v >= m:
            top_edges.append(i)
        else:
            match_edges.append(i)
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        try:
            eigvals = np.linalg.eigvalsh(A)
            return np.max(np.abs(eigvals))
        except:
            return np.inf
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            return True, radius
        except Exception:
            return False, np.inf
    
    # Start with a reasonable initial assignment
    # Try to minimize spectral radius
    # Strategy: random sampling + local search (flip one edge at a time)
    
    rng = np.random.default_rng(42)
    
    # Initial signs: try a few patterns
    # Pattern 1: all +1
    # Pattern 2: bottom +1, top -1, match +1
    # Pattern 3: alternating patterns
    
    best_signs = None
    best_radius = np.inf
    
    # Try some structured initial assignments
    initial_patterns = []
    
    # All +1
    initial_patterns.append(np.ones(E, dtype=np.int8))
    
    # Bottom +1, top -1, match +1
    p1 = np.ones(E, dtype=np.int8)
    for i in top_edges:
        p1[i] = -1
    initial_patterns.append(p1)
    
    # Bottom -1, top +1, match +1
    p2 = np.ones(E, dtype=np.int8)
    for i in bottom_edges:
        p2[i] = -1
    initial_patterns.append(p2)
    
    # Alternating on cycles
    for phase in [0, 1]:
        p3 = np.ones(E, dtype=np.int8)
        for idx, i in enumerate(bottom_edges):
            if idx % 2 != phase:
                p3[i] = -1
        for idx, i in enumerate(top_edges):
            if idx % 2 != phase:
                p3[i] = -1
        initial_patterns.append(p3)
    
    # Random initializations
    for _ in range(20):
        initial_patterns.append(rng.choice([-1, 1], size=E).astype(np.int8))
    
    # Evaluate initial patterns
    for signs in initial_patterns:
        if time.time() > deadline:
            break
        valid, radius = is_valid_lift(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # Local search: flip one edge at a time
    if best_signs is None:
        # Fallback: use the first valid pattern or all +1
        best_signs = initial_patterns[0].copy()
        best_radius = np.inf
        valid, radius = is_valid_lift(best_signs)
        if valid:
            best_radius = radius
    
    if time.time() > deadline:
        return best_signs
    
    # Local search with time guard
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Pre-compute which edges can be flipped (all of them, but we need to check validity)
    # To speed up, we can cache some computations
    
    max_iterations = 50000
    iteration = 0
    
    while time.time() < deadline - 2.0 and iteration < max_iterations:
        iteration += 1
        
        # Try flipping a random edge
        edge_idx = rng.integers(0, E)
        
        # Skip if this edge is the same as before (to avoid oscillation)
        trial_signs = current_signs.copy()
        trial_signs[edge_idx] = -trial_signs[edge_idx]
        
        valid, radius = is_valid_lift(trial_signs)
        if valid and radius < current_radius:
            current_signs = trial_signs
            current_radius = radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_signs = current_signs.copy()
        elif time.time() > deadline - 1.0:
            break
    
    # If we have time, try a second pass with more random restarts
    if time.time() < deadline - 5.0:
        for restart in range(10):
            if time.time() > deadline - 3.0:
                break
            
            # Random restart
            restart_signs = rng.choice([-1, 1], size=E).astype(np.int8)
            valid, radius = is_valid_lift(restart_signs)
            if not valid:
                continue
            
            local_signs = restart_signs.copy()
            local_radius = radius
            
            # Local search from this restart
            for _ in range(5000):
                if time.time() > deadline - 2.0:
                    break
                
                edge_idx = rng.integers(0, E)
                trial_signs = local_signs.copy()
                trial_signs[edge_idx] = -trial_signs[edge_idx]
                
                valid, radius = is_valid_lift(trial_signs)
                if valid and radius < local_radius:
                    local_signs = trial_signs
                    local_radius = radius
                    if local_radius < best_radius:
                        best_radius = local_radius
                        best_signs = local_signs.copy()
    
    return best_signs
