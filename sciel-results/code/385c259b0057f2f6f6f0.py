"""Sign-edges solver for prism-graph spectral lifts: analytical + local search."""
import time
import numpy as np

# Hard deadline guard: evaluator kills at 90s; return well before.
_DEADLINE = 82.0  # seconds from process start as a safety margin


def _build_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v), s in zip(edges, signs):
        sv = float(s)
        A[u, v] = sv
        A[v, u] = sv
    return A


def _lift_radius(graph, signs):
    """Return spectral radius of the lift's signed adjacency, or inf if invalid."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, _ = check_lift(graph, signs)
        return float(radius)
    except Exception:
        return float('inf')


def _analytical_radius(graph, signs):
    """
    Fast analytical evaluation of spectral radius for prism graph signed adjacency.
    Prism: vertices 0..m-1 (top cycle), m..2m-1 (bottom cycle), matching (i, i+m).
    Edges: top cycle (0,1),(1,2),...,(m-1,0); bottom cycle (m,m+1),...,(2m-1,m); matching.
    
    The signed adjacency matrix has a block structure. For a sign pattern, we can
    compute eigenvalues analytically using the winding number of negative edges.
    
    For the top cycle with k_top negative edges, eigenvalues of that cycle's signed
    adjacency are 2*cos(2*pi*j/m + pi*k_top/m) for j=0..m-1.
    Similarly for bottom cycle with k_bot negative edges.
    The matching edges have signs s_match_i for i=0..m-1.
    
    The full matrix is:
    [ A_top   M ]
    [ M'      A_bot ]
    where M is the matching diagonal with signs.
    
    We use the fact that if both cycles have the same "twist" parity structure,
    we can decouple. But in general, we need to compute the eigenvalues of this
    2m x 2m matrix. For small m (4,5,7,9), this is fast enough.
    """
    m = graph['n'] // 2
    edges = graph['edges']
    
    # Parse edge structure: identify which edges are top cycle, bottom cycle, matching
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for idx, (u, v) in enumerate(edges):
        s = int(signs[idx])
        if u < m and v < m:
            top_cycle_edges.append((u, v, s))
        elif u >= m and v >= m:
            bottom_cycle_edges.append((u - m, v - m, s))
        else:
            matching_edges.append((min(u, v) % m, s))
    
    # Sort matching edges by vertex
    matching_edges.sort()
    match_signs = np.array([s for _, s in matching_edges], dtype=np.float64)
    
    # Build top cycle signed adjacency (m x m)
    A_top = np.zeros((m, m), dtype=np.float64)
    for u, v, s in top_cycle_edges:
        A_top[u, v] = s
        A_top[v, u] = s
    
    # Build bottom cycle signed adjacency (m x m)
    A_bot = np.zeros((m, m), dtype=np.float64)
    for u, v, s in bottom_cycle_edges:
        A_bot[u, v] = s
        A_bot[v, u] = s
    
    # Full matrix
    A = np.zeros((2 * m, 2 * m), dtype=np.float64)
    A[:m, :m] = A_top
    A[m:, m:] = A_bot
    A[:m, m:] = np.diag(match_signs)
    A[m:, :m] = np.diag(match_signs)
    
    # Check connectivity of the 2-lift
    # The lift is connected iff the sign pattern has "odd winding" around the
    # fundamental cycle. For a prism, the fundamental cycle goes around one
    # cycle and back via matching. The lift is connected iff the product of
    # signs around any cycle that's not a boundary is -1 (i.e., odd number of
    # negative edges in the "non-trivial" cycle).
    # 
    # For the prism graph, the cycle space has dimension 2 (two independent cycles:
    # top cycle and bottom cycle, or top cycle + matching path). The lift is
    # connected iff the sign assignment is "non-bounding", i.e., the product of
    # signs around the top cycle times the product around the bottom cycle (with
    # appropriate matching) is -1.
    #
    # More precisely: the 2-lift is connected iff there exists a cycle in the base
    # graph whose signed product is -1. For the prism, take the cycle: go around
    # top cycle (product = P_top), then down via matching (sign = s_0), then back
    # around bottom cycle in reverse (product = P_bot), then up via matching (sign = s_0).
    # Total product = P_top * s_0 * P_bot * s_0 = P_top * P_bot.
    # So the lift is connected iff P_top * P_bot = -1, i.e., P_top != P_bot.
    
    P_top = np.prod([s for _, _, s in top_cycle_edges])
    P_bot = np.prod([s for _, _, s in bottom_cycle_edges])
    
    if P_top * P_bot == 1:
        # Lift is disconnected
        return float('inf')
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigvals)))


def _check_valid_lift(graph, signs):
    """Check if the lift is connected and 3-regular."""
    m = graph['n'] // 2
    edges = graph['edges']
    
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for idx, (u, v) in enumerate(edges):
        s = int(signs[idx])
        if u < m and v < m:
            top_cycle_edges.append((u, v, s))
        elif u >= m and v >= m:
            bottom_cycle_edges.append((u - m, v - m, s))
        else:
            matching_edges.append((min(u, v) % m, s))
    
    P_top = np.prod([s for _, _, s in top_cycle_edges])
    P_bot = np.prod([s for _, _, s in bottom_cycle_edges])
    
    return P_top * P_bot == -1


def _systematic_search(graph, deadline):
    """
    Systematically search over sign patterns using the analytical evaluator.
    For prism graphs with m=4,5,7,9, the number of edges is 3m.
    We enumerate patterns by the number of negative edges in top and bottom cycles.
    """
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Identify edge indices for top cycle, bottom cycle, matching
    top_indices = []
    bot_indices = []
    match_indices = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_indices.append(idx)
        elif u >= m and v >= m:
            bot_indices.append(idx)
        else:
            match_indices.append(idx)
    
    best_signs = None
    best_radius = float('inf')
    
    # Try all combinations of negative edge counts in top and bottom cycles
    # For each count, try a few representative patterns
    for k_top in range(m + 1):
        for k_bot in range(m + 1):
            if time.time() >= deadline:
                break
            # P_top = (-1)^k_top, P_bot = (-1)^k_bot
            # Connected iff k_top + k_bot is odd
            if (k_top + k_bot) % 2 == 0:
                continue
            
            # Try a few patterns: spread negatives evenly
            for spread in range(min(3, m)):
                # Pattern: place k_top negatives at positions spread by 'spread'
                top_signs = np.ones(m, dtype=np.int64)
                if k_top > 0:
                    positions = (np.arange(k_top) * (m // k_top + 1) + spread) % m
                    top_signs[positions] = -1
                
                bot_signs = np.ones(m, dtype=np.int64)
                if k_bot > 0:
                    positions = (np.arange(k_bot) * (m // k_bot + 1) + spread) % m
                    bot_signs[positions] = -1
                
                # Try a few matching patterns
                for match_pattern in range(2):
                    if time.time() >= deadline:
                        break
                    signs = np.ones(n_edges, dtype=np.int64)
                    
                    # Set top cycle signs
                    for i, idx in enumerate(top_indices):
                        signs[idx] = top_signs[i]
                    
                    # Set bottom cycle signs
                    for i, idx in enumerate(bot_indices):
                        signs[idx] = bot_signs[i]
                    
                    # Set matching signs
                    if match_pattern == 0:
                        # All positive matching
                        for idx in match_indices:
                            signs[idx] = 1
                    else:
                        # Alternating matching
                        for i, idx in enumerate(match_indices):
                            signs[idx] = 1 if i % 2 == 0 else -1
                    
                    r = _analytical_radius(graph, signs)
                    if r < best_radius:
                        best_radius = r
                        best_signs = signs.copy()
        
        if time.time() >= deadline:
            break
    
    return best_signs, best_radius


def _local_search(graph, signs, radius, deadline):
    """Flip-based local search using analytical evaluator."""
    n_edges = len(graph['edges'])
    signs = signs.copy()
    best_radius = radius
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try single flips
        for i in range(n_edges):
            if time.time() >= deadline:
                break
            signs[i] = -signs[i]
            r = _analytical_radius(graph, signs)
            if r < best_radius - 1e-12:
                best_radius = r
                improved = True
            else:
                signs[i] = -signs[i]  # revert
        
        # Try pairwise flips (escape local minima)
        if not improved and time.time() < deadline:
            for i in range(n_edges):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n_edges):
                    if time.time() >= deadline:
                        break
                    signs[i] = -signs[i]
                    signs[j] = -signs[j]
                    r = _analytical_radius(graph, signs)
                    if r < best_radius - 1e-12:
                        best_radius = r
                        improved = True
                    else:
                        signs[i] = -signs[i]
                        signs[j] = -signs[j]
                    if improved:
                        break
                if improved:
                    break
    
    return signs, best_radius


def sign_edges(graph, budget):
    start = time.time()
    deadline = start + _DEADLINE

    n_edges = len(graph['edges'])
    if n_edges == 0:
        return np.array([], dtype=np.int64)

    # Phase 1: Systematic search using analytical evaluator
    signs, radius = _systematic_search(graph, deadline)

    if signs is None:
        # Fallback: random search
        rng = np.random.default_rng(3)
        best_signs = None
        best_radius = float('inf')
        for _ in range(budget):
            s = rng.choice([-1, 1], size=n_edges)
            r = _analytical_radius(graph, s)
            if r < best_radius:
                best_radius = r
                best_signs = s.copy()
        signs = best_signs if best_signs is not None else np.ones(n_edges, dtype=np.int64)
        radius = best_radius if best_radius < float('inf') else _analytical_radius(graph, signs)

    # Phase 2: local search to refine
    if time.time() < deadline:
        signs, radius = _local_search(graph, signs, radius, deadline)

    # Verify with check_lift and fallback if invalid
    final_radius = _lift_radius(graph, signs)
    if final_radius == float('inf'):
        # Try random fallback
        rng = np.random.default_rng(42)
        for _ in range(100):
            s = rng.choice([-1, 1], size=n_edges)
            r = _lift_radius(graph, s)
            if r < float('inf'):
                signs = s
                break

    # Ensure the return is a numpy array of int signs
    return np.asarray(signs, dtype=np.int64)
