"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_signed_adjacency(graph, signs):
    """Build the signed adjacency matrix from graph and signs."""
    n = graph['n']
    A = np.zeros((n, n))
    for edge, s in zip(graph['edges'], signs):
        u, v = edge
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(A):
    """Compute spectral radius of symmetric matrix."""
    eigvals = np.linalg.eigvalsh(A)
    return np.max(np.abs(eigvals))

def _is_valid_lift(graph, signs):
    """Check if the lift is connected and 3-regular using check_lift."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, radius
    except Exception:
        return False, None

def _simulated_annealing(graph, signs, max_iters=2000, start_temp=1.0, end_temp=0.01):
    """Simulated Annealing to minimize spectral radius."""
    n_edges = len(signs)
    signs = signs.copy()
    A = _build_signed_adjacency(graph, signs)
    current_radius = _spectral_radius(A)
    
    # Verify validity
    valid, _ = _is_valid_lift(graph, signs)
    if not valid:
        return signs, current_radius
        
    best_signs = signs.copy()
    best_radius = current_radius
    
    # Temperature schedule
    temp = start_temp
    cooling_rate = (end_temp / start_temp) ** (1.0 / max_iters)
    
    for it in range(max_iters):
        # Pick a random edge to flip
        i = np.random.randint(n_edges)
        old_sign = signs[i]
        signs[i] = -old_sign
        
        A = _build_signed_adjacency(graph, signs)
        new_radius = _spectral_radius(A)
        
        # Check validity
        valid, _ = _is_valid_lift(graph, signs)
        
        if valid:
            delta = new_radius - current_radius
            # Accept if better or with probability exp(-delta/T)
            if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-9)):
                current_radius = new_radius
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_signs = signs.copy()
            else:
                # Reject: revert
                signs[i] = old_sign
        else:
            # Reject invalid: revert
            signs[i] = old_sign
            
        temp *= cooling_rate
        
    return best_signs, best_radius

def sign_edges(graph, budget):
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    rng = np.random.default_rng(42)
    np.random.seed(42) # Seed global random for SA
    
    best_signs = None
    best_radius = float('inf')
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    # Initial candidates
    candidates = []
    
    # 1. Alternating on cycles
    alt_signs = []
    for (u, v) in graph['edges']:
        if u < m and v < m:
            alt_signs.append(1 if (u % 2 == 0) else -1)
        elif u >= m and v >= m:
            alt_signs.append(1 if ((u - m) % 2 == 0) else -1)
        else:
            alt_signs.append(1)
    candidates.append(np.array(alt_signs, dtype=int))
    
    # 2. All +1
    candidates.append(np.ones(n_edges, dtype=int))
    
    # 3. All -1 on matching
    signs3 = candidates[0].copy()
    for i, (u, v) in enumerate(graph['edges']):
        if not (u < m and v < m) and not (u >= m and v >= m):
            signs3[i] = -1
    candidates.append(signs3)
    
    # 4-10. Random
    for _ in range(7):
        candidates.append(rng.choice([-1, 1], size=n_edges))
        
    # Evaluate candidates and run SA on the best few
    scored_candidates = []
    for signs in candidates:
        valid, radius = _is_valid_lift(graph, signs)
        if valid:
            scored_candidates.append((radius, signs))
            
    # Sort by radius
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run SA on top 3 candidates
    for radius, signs in scored_candidates[:3]:
        if time.time() - start_time > time_limit:
            break
            
        # Run SA
        sa_signs, sa_radius = _simulated_annealing(graph, signs, max_iters=1000)
        
        if sa_radius < best_radius:
            best_radius = sa_radius
            best_signs = sa_signs.copy()
            
    # If no valid candidate was found or SA didn't run, try random search fallback
    if best_signs is None:
        # Try to find any valid solution
        for _ in range(100):
            signs = rng.choice([-1, 1], size=n_edges)
            valid, radius = _is_valid_lift(graph, signs)
            if valid:
                best_signs = signs
                best_radius = radius
                break
        if best_signs is None:
            from autoresearch_suite.science_tcs_expansion import lift_baseline
            return lift_baseline(graph, budget)
            
    return best_signs
# EVOLVE-BLOCK-END
