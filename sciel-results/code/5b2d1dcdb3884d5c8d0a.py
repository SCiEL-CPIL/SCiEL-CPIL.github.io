"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

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
    return float(np.max(np.abs(eigvals)))

def _check_validity(graph, signs):
    """Check if the lift is connected and 3-regular using check_lift.
    Returns (is_valid, radius)."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, radius
    except Exception:
        return False, None

def _flip_sign(signs, idx):
    """Create a new signs array with one sign flipped."""
    new_signs = signs.copy()
    new_signs[idx] = -new_signs[idx]
    return new_signs

def _local_search(graph, signs, max_iters=50000, time_limit=30.0):
    """Local search: flip individual signs to reduce spectral radius while maintaining validity."""
    n_edges = len(graph['edges'])
    best_signs = signs.copy()
    best_radius = _check_validity(graph, best_signs)[1]
    if best_radius is None:
        return best_signs, None
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try flipping each sign
        for i in range(n_edges):
            if (time.time() - start_time) >= time_limit:
                break
            new_signs = _flip_sign(best_signs, i)
            valid, radius = _check_validity(graph, new_signs)
            if valid and radius is not None and radius < best_radius - 1e-10:
                best_signs = new_signs
                best_radius = radius
                improved = True
    return best_signs, best_radius

def _random_signs(n_edges, rng):
    """Generate random signs."""
    return rng.choice([-1, 1], size=n_edges)

def sign_edges(graph, budget):
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = float('inf')
    
    # Phase 1: Random sampling to find a valid starting point
    # Use the seed's approach first
    for _ in range(budget):
        signs = _random_signs(n_edges, rng)
        valid, radius = _check_validity(graph, signs)
        if valid and radius is not None:
            if best_signs is None or radius < best_radius:
                best_signs = signs
                best_radius = radius
            if (time.time() - start_time) > 2.0:
                break
    
    if best_signs is None:
        # Try more random samples
        for _ in range(500):
            signs = _random_signs(n_edges, rng)
            valid, radius = _check_validity(graph, signs)
            if valid and radius is not None:
                if best_signs is None or radius < best_radius:
                    best_signs = signs
                    best_radius = radius
            if (time.time() - start_time) > 5.0:
                break
    
    if best_signs is None:
        # Last resort: all +1
        best_signs = np.ones(n_edges, dtype=int)
        valid, radius = _check_validity(graph, best_signs)
        if valid:
            best_radius = radius
    
    if best_signs is None:
        # Should not happen, but return something
        return np.ones(n_edges, dtype=int)
    
    # Phase 2: Local search from the best starting point
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        best_signs, best_radius = _local_search(
            graph, best_signs, 
            max_iters=100000,
            time_limit=max(0, time_remaining - 5.0)
        )
    
    # Phase 3: Try multiple random restarts with local search if time permits
    time_remaining = deadline - time.time()
    restart_count = 0
    while time_remaining > 10.0 and restart_count < 20:
        signs = _random_signs(n_edges, rng)
        valid, radius = _check_validity(graph, signs)
        if valid and radius is not None:
            # Quick local search with short time limit
            short_time = min(3.0, time_remaining - 10.0)
            if short_time > 0.5:
                new_signs, new_radius = _local_search(
                    graph, signs,
                    max_iters=10000,
                    time_limit=short_time
                )
                if new_radius is not None and new_radius < best_radius:
                    best_signs = new_signs
                    best_radius = new_radius
        restart_count += 1
        time_remaining = deadline - time.time()
    
    # Final: one more local search polish if time permits
    time_remaining = deadline - time.time()
    if time_remaining > 3.0:
        best_signs, best_radius = _local_search(
            graph, best_signs,
            max_iters=50000,
            time_limit=max(0, time_remaining - 2.0)
        )
    
    return best_signs
# EVOLVE-BLOCK-END
