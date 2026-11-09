"""Spectral graph lifts: minimize spectral radius via structured search + local refinement."""
import time
import numpy as np

def _build_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        A[u, v] = float(s)
        A[v, u] = float(s)
    return A

def _spectral_radius(A):
    try:
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    except Exception:
        return float('inf')

def _check_valid(graph, signs):
    """Return (valid, radius) using the provided helper."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, float(radius)
    except Exception:
        return False, float('inf')

def _evaluate(graph, signs):
    """Evaluate a sign vector: return (valid, radius)."""
    valid, radius = _check_valid(graph, signs)
    if not valid:
        return False, float('inf')
    return True, radius

def _flip_sign(signs, idx):
    """Return a copy of signs with sign at idx flipped."""
    new_signs = signs.copy()
    new_signs[idx] = -new_signs[idx]
    return new_signs

def _local_search(graph, signs, time_limit, deadline):
    """
    Local search: try flipping each edge, keep if it improves spectral radius.
    Repeat until no improvement or time runs out.
    """
    current_signs = signs.copy()
    current_radius = _spectral_radius(_build_signed_adj(graph, current_signs))
    n_edges = len(graph['edges'])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_edges):
            if time.time() >= deadline:
                break
            candidate = _flip_sign(current_signs, i)
            valid, radius = _check_valid(graph, candidate)
            if valid and radius < current_radius - 1e-12:
                current_signs = candidate
                current_radius = radius
                improved = True
    return current_signs, current_radius

def _random_search(graph, n_iter, deadline, rng):
    """
    Random search with local refinement.
    Returns best (signs, radius) found.
    """
    best_signs = None
    best_radius = float('inf')
    n_edges = len(graph['edges'])
    
    for _ in range(n_iter):
        if time.time() >= deadline:
            break
        signs = rng.choice([-1, 1], size=n_edges)
        valid, radius = _check_valid(graph, signs)
        if valid and radius < best_radius:
            best_signs = signs.copy()
            best_radius = radius
            # Refine with local search
            refined_signs, refined_radius = _local_search(graph, best_signs, time_limit, deadline)
            if refined_radius < best_radius:
                best_signs = refined_signs
                best_radius = refined_radius
    
    return best_signs, best_radius

def _structured_patterns(graph, m, deadline, rng):
    """
    Generate structured sign patterns likely to give low spectral radius.
    For prism graphs, we can exploit the symmetry.
    
    The prism graph has 3m edges:
    - m edges in bottom cycle: (i, i+1 mod m)
    - m edges in top cycle: (m+i, m+i+1 mod m)  
    - m matching edges: (i, m+i)
    
    We try various structured patterns.
    """
    n_edges = len(graph['edges'])
    edges = graph['edges']
    candidates = []
    
    # Pattern 1: All +1
    candidates.append(np.ones(n_edges, dtype=np.int8))
    
    # Pattern 2: All -1
    candidates.append(-np.ones(n_edges, dtype=np.int8))
    
    # Pattern 3: Bottom cycle +1, top cycle -1, matching +1
    signs = np.ones(n_edges, dtype=np.int8)
    for idx, (u, v) in enumerate(edges):
        if u >= m and v >= m:  # top cycle
            signs[idx] = -1
    candidates.append(signs)
    
    # Pattern 4: Bottom cycle -1, top cycle +1, matching -1
    signs = np.ones(n_edges, dtype=np.int8)
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:  # bottom cycle
            signs[idx] = -1
        elif u < m and v >= m or u >= m and v < m:  # matching
            signs[idx] = -1
    candidates.append(signs)
    
    # Pattern 5: Alternating signs on cycles
    signs = np.ones(n_edges, dtype=np.int8)
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:  # bottom cycle
            # Determine position in cycle
            if u == m - 1 and v == 0:
                signs[idx] = -1 if (m % 2 == 0) else 1
            else:
                signs[idx] = 1 if (u % 2 == 0) else -1
        elif u >= m and v >= m:  # top cycle
            if u == 2*m - 1 and v == m:
                signs[idx] = -1 if (m % 2 == 0) else 1
            else:
                signs[idx] = 1 if ((u - m) % 2 == 0) else -1
    candidates.append(signs)
    
    # Pattern 6: Matching all -1, cycles all +1
    signs = np.ones(n_edges, dtype=np.int8)
    for idx, (u, v) in enumerate(edges):
        if (u < m and v >= m) or (u >= m and v < m):
            signs[idx] = -1
    candidates.append(signs)
    
    # Pattern 7: Matching all +1, cycles all -1
    signs = np.ones(n_edges, dtype=np.int8)
    for idx, (u, v) in enumerate(edges):
        if (u < m and v < m) or (u >= m and v >= m):
            signs[idx] = -1
    candidates.append(signs)
    
    # Pattern 8: Random structured - half matching -1
    signs = np.ones(n_edges, dtype=np.int8)
    for idx, (u, v) in enumerate(edges):
        if (u < m and v >= m) or (u >= m and v < m):
            if rng.random() < 0.5:
                signs[idx] = -1
    candidates.append(signs)
    
    # Pattern 9: Checkerboard on all edges based on vertex parity
    signs = np.ones(n_edges, dtype=np.int8)
    for idx, (u, v) in enumerate(edges):
        if (u + v) % 2 == 1:
            signs[idx] = -1
    candidates.append(signs)
    
    # Pattern 10: Another structured pattern
    signs = np.ones(n_edges, dtype=np.int8)
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:  # bottom
            signs[idx] = 1 if u % 3 == 0 else -1
        elif u >= m and v >= m:  # top
            signs[idx] = 1 if (u - m) % 3 == 0 else -1
    candidates.append(signs)
    
    best_signs = None
    best_radius = float('inf')
    
    for signs in candidates:
        if time.time() >= deadline:
            break
        valid, radius = _check_valid(graph, signs)
        if valid and radius < best_radius:
            best_signs = signs.copy()
            best_radius = radius
    
    return best_signs, best_radius

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # 55 second safety margin
    
    m = graph['n'] // 2
    rng = np.random.default_rng(3)
    
    # First, try structured patterns
    best_signs, best_radius = _structured_patterns(graph, m, deadline, rng)
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(len(graph['edges']), dtype=np.int8)
        best_radius = _spectral_radius(_build_signed_adj(graph, best_signs))
    
    # Local search from the best structured pattern
    if time.time() < deadline:
        refined_signs, refined_radius = _local_search(graph, best_signs, budget, deadline)
        if refined_radius < best_radius:
            best_signs = refined_signs
            best_radius = refined_radius
    
    # Random search with local refinement
    if time.time() < deadline:
        n_iter = 5000  # Adjust based on time
        rand_signs, rand_radius = _random_search(graph, n_iter, deadline, rng)
        if rand_signs is not None and rand_radius < best_radius:
            best_signs = rand_signs
            best_radius = rand_radius
    
    # Final local search pass on the best found
    if time.time() < deadline - 5.0:
        final_signs, final_radius = _local_search(graph, best_signs, budget, deadline)
        if final_radius < best_radius:
            best_signs = final_signs
            best_radius = final_radius
    
    return best_signs.astype(np.int8)
