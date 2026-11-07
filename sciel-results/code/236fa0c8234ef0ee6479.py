<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    """
    Assign signs (-1 or +1) to edges of a prism graph to minimize the
    spectral radius of the signed adjacency matrix, subject to the
    2-lift being connected and 3-regular.
    
    Strategy: Exploit the circulant structure of prism graphs.
    A prism graph with parameter m has:
      - Bottom cycle: vertices 0..m-1, edges (i, i+1 mod m)
      - Top cycle: vertices m..2m-1, edges (m+i, m+i+1 mod m)
      - Matching: edges (i, m+i) for i=0..m-1
    
    The signed adjacency matrix A has entries A[u,v] = sign for edge (u,v).
    The spectral radius is max |eigenvalue| of A.
    
    For a connected 3-regular graph with 2m vertices, we want to minimize
    the spectral radius. The theoretical minimum for a 3-regular graph on
    2m vertices is related to the Alon-Boppana bound, but for small m we
    can search more directly.
    
    Key insight: The prism graph is a circulant-like graph. We can parametrize
    the sign assignment and use continuous optimization over phases, then
    round to the nearest sign.
    
    However, a simpler and effective approach for these small graphs (m=4,5,7,9)
    is to use a combination of:
    1. Random restarts with local search (flipping individual signs)
    2. Exploiting symmetry: the graph has a Z_m symmetry, so we can fix
       some signs and search over the rest
    3. Gradient-based continuous relaxation
    
    Let's implement a robust local search with multiple restarts.
    """
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency structure for fast evaluation
    # adj[u] = list of (v, edge_index)
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            return True
        except Exception:
            return False
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds safety margin
    
    best_signs = None
    best_radius = float('inf')
    
    # Strategy 1: Random initialization + local search
    rng = np.random.default_rng(3)
    
    def local_search(initial_signs, max_iter=5000):
        """Local search: flip individual signs to reduce spectral radius."""
        signs = initial_signs.copy()
        current_radius = compute_radius(signs)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            best_flip = None
            best_new_radius = current_radius
            
            # Try flipping each edge
            for idx in range(num_edges):
                old_sign = signs[idx]
                signs[idx] = -old_sign
                new_radius = compute_radius(signs)
                signs[idx] = old_sign
                
                if new_radius < best_new_radius - 1e-12:
                    best_new_radius = new_radius
                    best_flip = idx
            
            if best_flip is None:
                break
            
            signs[best_flip] = -signs[best_flip]
            current_radius = best_new_radius
        
        return signs, current_radius
    
    # Strategy 2: Structured initialization based on circulant structure
    # For a prism graph, we can try sign patterns that respect the Z_m symmetry
    # Bottom cycle signs: s0, s1, ..., s_{m-1} (for edges (i, i+1))
    # Top cycle signs: t0, t1, ..., t_{m-1} (for edges (m+i, m+i+1))
    # Matching signs: r0, r1, ..., r_{m-1} (for edges (i, m+i))
    
    # Let's identify which edges are which type
    bottom_cycle_edges = []
    top_cycle_edges = []
    matching_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_cycle_edges.append(idx)
        elif u >= m and v >= m:
            top_cycle_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # Generate candidate sign patterns
    candidates = []
    
    # Pattern 1: All matching signs +1, alternating cycle signs
    for cycle_pattern in [
        lambda i: 1 if i % 2 == 0 else -1,
        lambda i: -1 if i % 2 == 0 else 1,
        lambda i: 1,
        lambda i: -1,
    ]:
        for match_sign in [1, -1]:
            signs = np.ones(num_edges, dtype=int)
            for i, idx in enumerate(bottom_cycle_edges):
                signs[idx] = cycle_pattern(i)
            for i, idx in enumerate(top_cycle_edges):
                signs[idx] = cycle_pattern(i)
            for i, idx in enumerate(matching_edges):
                signs[idx] = match_sign
            candidates.append(signs)
    
    # Pattern 2: Random with constraint that lift is connected
    # The lift is connected iff the product of signs around certain cycles is +1
    # For a prism, the key cycles are:
    # - Each "square" formed by two matching edges and two cycle edges
    # - The full cycles
    
    # Let's just do random initialization with validity check
    num_random_starts = 200
    for start_idx in range(num_random_starts):
        if time.time() > deadline:
            break
        
        # Random signs
        signs = rng.choice([-1, 1], size=num_edges)
        
        # Check validity
        if not is_valid_lift(signs):
            continue
        
        # Local search
        improved_signs, radius = local_search(signs, max_iter=2000)
        
        # Verify validity after local search
        if is_valid_lift(improved_signs):
            if radius < best_radius:
                best_radius = radius
                best_signs = improved_signs.copy()
    
    # Strategy 3: More systematic search over structured patterns
    # Try all combinations of a few key parameters
    # For small m, we can be more exhaustive
    
    if m <= 7:
        # Try all 2^m patterns for matching signs, with fixed cycle patterns
        for cycle_pattern_idx in range(4):
            if time.time() > deadline:
                break
            
            if cycle_pattern_idx == 0:
                cycle_fn = lambda i: 1 if i % 2 == 0 else -1
            elif cycle_pattern_idx == 1:
                cycle_fn = lambda i: -1 if i % 2 == 0 else 1
            elif cycle_pattern_idx == 2:
                cycle_fn = lambda i: 1
            else:
                cycle_fn = lambda i: -1
            
            # Try subsets of matching sign patterns
            for match_mask in range(min(2**m, 100)):
                if time.time() > deadline:
                    break
                
                signs = np.ones(num_edges, dtype=int)
                for i, idx in enumerate(bottom_cycle_edges):
                    signs[idx] = cycle_fn(i)
                for i, idx in enumerate(top_cycle_edges):
                    signs[idx] = cycle_fn(i)
                for i, idx in enumerate(matching_edges):
                    signs[idx] = 1 if (match_mask >> i) & 1 else -1
                
                if not is_valid_lift(signs):
                    continue
                
                improved_signs, radius = local_search(signs, max_iter=1000)
                if is_valid_lift(improved_signs):
                    if radius < best_radius:
                        best_radius = radius
                        best_signs = improved_signs.copy()
    
    # Strategy 4: Continuous relaxation with gradient descent
    # Parametrize signs as cos(theta_i) where theta_i in [0, pi]
    # Minimize spectral radius of A(theta) where A[u,v] = cos(theta_idx)
    # Then round to nearest sign
    
    if time.time() < deadline - 10:
        def spectral_radius_continuous(thetas):
            A = np.zeros((n, n))
            for idx, (u, v) in enumerate(edges):
                s = np.cos(thetas[idx])
                A[u, v] = s
                A[v, u] = s
            eigvals = np.linalg.eigvalsh(A)
            return np.max(np.abs(eigvals))
        
        # Gradient descent
        def gradient_descent(init_thetas, lr=0.1, max_iter=1000):
            thetas = init_thetas.copy()
            for i in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute gradient via finite differences
                grad = np.zeros_like(thetas)
                eps = 1e-4
                for j in range(num_edges):
                    thetas_plus = thetas.copy()
                    thetas_plus[j] += eps
                    thetas_minus = thetas.copy()
                    thetas_minus[j] -= eps
                    grad[j] = (spectral_radius_continuous(thetas_plus) - 
                              spectral_radius_continuous(thetas_minus)) / (2 * eps)
                
                thetas -= lr * grad
                thetas = np.clip(thetas, 0, np.pi)
            
            return thetas
        
        # Try a few initializations
        for init_idx in range(20):
            if time.time() > deadline:
                break
            
            init_thetas = rng.uniform(0, np.pi, size=num_edges)
            thetas = gradient_descent(init_thetas, lr=0.05, max_iter=500)
            
            # Round to signs
            signs = np.where(np.cos(thetas) >= 0, 1, -1).astype(int)
            
            if not is_valid_lift(signs):
                continue
            
            # Local search from this point
            improved_signs, radius = local_search(signs, max_iter=1000)
            if is_valid_lift(improved_signs):
                if radius < best_radius:
                    best_radius = radius
                    best_signs = improved_signs.copy()
    
    # Final fallback: if no valid solution found, use the seed's approach
    if best_signs is None:
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)
    
    return best_signs
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    """
    Assign signs (-1 or +1) to edges of a prism graph to minimize the
    spectral radius of the signed adjacency matrix, subject to the
    2-lift being connected and 3-regular.
    
    Strategy: Exploit the circulant structure of prism graphs using continuous
    phase-space optimization with momentum, followed by discrete local search.
    """
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency structure for fast evaluation
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            return True
        except Exception:
            return False
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_signs = None
    best_radius = float('inf')
    
    # Identify edge types for structured initialization
    bottom_cycle_edges = []
    top_cycle_edges = []
    matching_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_cycle_edges.append(idx)
        elif u >= m and v >= m:
            top_cycle_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    rng = np.random.default_rng(3)
    
    def local_search(initial_signs, max_iter=3000):
        """Local search: flip individual signs to reduce spectral radius."""
        signs = initial_signs.copy()
        current_radius = compute_radius(signs)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            best_flip = None
            best_new_radius = current_radius
            
            for idx in range(num_edges):
                old_sign = signs[idx]
                signs[idx] = -old_sign
                new_radius = compute_radius(signs)
                signs[idx] = old_sign
                
                if new_radius < best_new_radius - 1e-12:
                    best_new_radius = new_radius
                    best_flip = idx
            
            if best_flip is None:
                break
            
            signs[best_flip] = -signs[best_flip]
            current_radius = best_new_radius
        
        return signs, current_radius
    
    # Strategy 1: Phase-space continuous optimization with momentum
    # Parametrize each sign as cos(theta_i) where theta_i in [0, pi]
    # Minimize spectral radius via gradient descent with momentum
    
    def spectral_radius_continuous(thetas):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = np.cos(thetas[idx])
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def gradient_descent_momentum(init_thetas, lr=0.1, momentum=0.9, max_iter=2000):
        """Gradient descent with momentum for continuous relaxation."""
        thetas = init_thetas.copy()
        velocity = np.zeros_like(thetas)
        current_val = spectral_radius_continuous(thetas)
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradient via finite differences
            grad = np.zeros_like(thetas)
            eps = 1e-4
            for j in range(num_edges):
                thetas_plus = thetas.copy()
                thetas_plus[j] += eps
                thetas_minus = thetas.copy()
                thetas_minus[j] -= eps
                grad[j] = (spectral_radius_continuous(thetas_plus) - 
                          spectral_radius_continuous(thetas_minus)) / (2 * eps)
            
            # Update with momentum
            velocity = momentum * velocity - lr * grad
            thetas = thetas + velocity
            thetas = np.clip(thetas, 0, np.pi)
            
            new_val = spectral_radius_continuous(thetas)
            if new_val < current_val - 1e-10:
                current_val = new_val
            else:
                # Revert if no improvement
                thetas = thetas - velocity
                velocity = 0.5 * velocity
        
        return thetas
    
    # Run phase-space optimization from multiple initializations
    num_phase_inits = 30
    for init_idx in range(num_phase_inits):
        if time.time() > deadline - 5:
            break
        
        # Different initialization strategies
        if init_idx < 10:
            # Uniform random
            init_thetas = rng.uniform(0, np.pi, size=num_edges)
        elif init_idx < 20:
            # Structured: alternating phases
            init_thetas = np.zeros(num_edges)
            for i, idx in enumerate(bottom_cycle_edges):
                init_thetas[idx] = i * np.pi / m
            for i, idx in enumerate(top_cycle_edges):
                init_thetas[idx] = i * np.pi / m
            for i, idx in enumerate(matching_edges):
                init_thetas[idx] = np.pi / 2
        else:
            # Near-extremal: mostly 0 or pi with some random
            init_thetas = rng.choice([0, np.pi], size=num_edges)
            for _ in range(num_edges // 4):
                init_thetas[rng.integers(0, num_edges)] = rng.uniform(0, np.pi)
        
        thetas = gradient_descent_momentum(init_thetas, lr=0.05, momentum=0.9, max_iter=1500)
        
        # Round to signs
        signs = np.where(np.cos(thetas) >= 0, 1, -1).astype(int)
        
        if is_valid_lift(signs):
            improved_signs, radius = local_search(signs, max_iter=2000)
            if is_valid_lift(improved_signs):
                if radius < best_radius:
                    best_radius = radius
                    best_signs = improved_signs.copy()
    
    # Strategy 2: Random restarts with local search
    num_random_starts = 100
    for start_idx in range(num_random_starts):
        if time.time() > deadline - 3:
            break
        
        signs = rng.choice([-1, 1], size=num_edges)
        
        if not is_valid_lift(signs):
            continue
        
        improved_signs, radius = local_search(signs, max_iter=1500)
        
        if is_valid_lift(improved_signs):
            if radius < best_radius:
                best_radius = radius
                best_signs = improved_signs.copy()
    
    # Strategy 3: Structured patterns
    for cycle_pattern in [
        lambda i: 1 if i % 2 == 0 else -1,
        lambda i: -1 if i % 2 == 0 else 1,
        lambda i: 1,
        lambda i: -1,
    ]:
        for match_sign in [1, -1]:
            if time.time() > deadline - 2:
                break
            
            signs = np.ones(num_edges, dtype=int)
            for i, idx in enumerate(bottom_cycle_edges):
                signs[idx] = cycle_pattern(i)
            for i, idx in enumerate(top_cycle_edges):
                signs[idx] = cycle_pattern(i)
            for i, idx in enumerate(matching_edges):
                signs[idx] = match_sign
            
            if is_valid_lift(signs):
                improved_signs, radius = local_search(signs, max_iter=1000)
                if is_valid_lift(improved_signs):
                    if radius < best_radius:
                        best_radius = radius
                        best_signs = improved_signs.copy()
    
    # Final fallback
    if best_signs is None:
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)
    
    return best_signs
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
