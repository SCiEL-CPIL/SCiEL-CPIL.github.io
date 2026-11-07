"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _spectral_radius_signed_fast(graph, signs):
    """Compute spectral radius of signed adjacency matrix efficiently."""
    n = graph['n']
    edges = graph['edges']
    # Build signed adjacency matrix
    A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        s = float(signs[i])
        A[u, v] = s
        A[v, u] = s
    try:
        eigvals = np.linalg.eigvalsh(A)
        return max(abs(eigvals[0]), abs(eigvals[-1]))
    except:
        return np.inf

def _is_valid_lift_fast(graph, signs):
    """Check if the lift is valid (connected, 3-regular) using BFS."""
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    # Build adjacency for the lift
    # Vertices: (v, side) where v in 0..n-1, side in {0, 1}
    # Total 2n vertices in the lift
    lift_n = 2 * n
    adj = [[] for _ in range(lift_n)]
    
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            # +1: joins equal sides
            adj[u * 2].append(v * 2)
            adj[v * 2].append(u * 2)
            adj[u * 2 + 1].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2 + 1)
        else:
            # -1: joins opposite sides
            adj[u * 2].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2)
            adj[u * 2 + 1].append(v * 2)
            adj[v * 2].append(u * 2 + 1)
    
    # Check 3-regular: each vertex should have degree 3
    for v in range(lift_n):
        if len(adj[v]) != 3:
            return False
    
    # Check connectedness using BFS
    visited = [False] * lift_n
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                queue.append(neighbor)
    
    return count == lift_n

def _get_cycle_structure(graph):
    """Extract cycle edge indices and matching edge indices."""
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    bottom_cycle_edges = []
    top_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_cycle_edges.append(i)
        elif u >= m and v >= m:
            top_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    return bottom_cycle_edges, top_cycle_edges, matching_edges

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    bottom_cycle_edges, top_cycle_edges, matching_edges = _get_cycle_structure(graph)
    
    best_signs = None
    best_radius = np.inf
    
    def try_signs(s):
        nonlocal best_signs, best_radius
        if time.time() > deadline:
            return
        if not _is_valid_lift_fast(graph, s):
            return
        r = _spectral_radius_signed_fast(graph, s)
        if r < best_radius:
            best_radius = r
            best_signs = s.copy()
    
    # Strategy: The key insight for prism graphs is that the spectral radius
    # is minimized when we create "frustration" in the cycles.
    # For a cycle with product of signs = -1 (odd number of -1s), the eigenvalues
    # are shifted. Let's systematically search over sign patterns.
    
    # First, try structured patterns based on frustration levels
    # For each cycle, the product of signs matters.
    # We want to explore configurations with different frustration patterns.
    
    # Pattern families:
    # 1. Vary the number of -1s in bottom cycle (0 to m)
    # 2. Vary the number of -1s in top cycle (0 to m)
    # 3. Vary matching signs
    
    # Since m can be up to 9, we can afford to be systematic for small m
    # but need to be smart for larger m.
    
    # Let's first try: all matching +1, then vary cycle frustrations
    # Key: for a cycle of length m with k negative signs, the eigenvalues depend on k
    
    # Try all combinations for small cases, random for large
    if m <= 7:
        # Systematic search over frustration patterns
        # For each possible number of -1s in bottom and top cycles
        for num_neg_bottom in range(0, m + 1):
            if time.time() > deadline:
                break
            for num_neg_top in range(0, m + 1):
                if time.time() > deadline:
                    break
                # For matching, try all +1 and alternating
                for match_pattern in [0, 1]:  # 0: all +1, 1: alternating
                    if time.time() > deadline:
                        break
                    
                    s = np.ones(num_edges, dtype=int)
                    
                    # Place num_neg_bottom -1s in bottom cycle
                    if num_neg_bottom > 0:
                        # Distribute evenly or at start
                        for j in range(num_neg_bottom):
                            idx = bottom_cycle_edges[j % len(bottom_cycle_edges)]
                            s[idx] = -1
                    
                    # Place num_neg_top -1s in top cycle
                    if num_neg_top > 0:
                        for j in range(num_neg_top):
                            idx = top_cycle_edges[j % len(top_cycle_edges)]
                            s[idx] = -1
                    
                    # Set matching signs
                    if match_pattern == 1:
                        for j, idx in enumerate(matching_edges):
                            if j % 2 == 1:
                                s[idx] = -1
                    
                    try_signs(s)
        
        # Also try all matching -1
        for num_neg_bottom in range(0, m + 1):
            if time.time() > deadline:
                break
            for num_neg_top in range(0, m + 1):
                if time.time() > deadline:
                    break
                s = np.ones(num_edges, dtype=int)
                
                if num_neg_bottom > 0:
                    for j in range(num_neg_bottom):
                        idx = bottom_cycle_edges[j % len(bottom_cycle_edges)]
                        s[idx] = -1
                
                if num_neg_top > 0:
                    for j in range(num_neg_top):
                        idx = top_cycle_edges[j % len(top_cycle_edges)]
                        s[idx] = -1
                
                # All matching -1
                for idx in matching_edges:
                    s[idx] = -1
                
                try_signs(s)
    else:
        # For larger m, use random search with structured bias
        rng = np.random.default_rng(3)
        
        # Try random patterns with bias toward frustration
        for trial in range(2000):
            if time.time() > deadline:
                break
            s = rng.choice([-1, 1], size=num_edges).astype(int)
            try_signs(s)
    
    # If no valid signs found yet, try more aggressively
    if best_signs is None:
        rng = np.random.default_rng(3)
        for trial in range(5000):
            if time.time() > deadline:
                break
            s = rng.choice([-1, 1], size=num_edges).astype(int)
            try_signs(s)
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
        if _is_valid_lift_fast(graph, best_signs):
            best_radius = _spectral_radius_signed_fast(graph, best_signs)
    
    # Local search: flip one edge at a time
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    improved = True
    max_local_iterations = 20
    for _ in range(max_local_iterations):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            trial = current_signs.copy()
            trial[i] = -trial[i]
            
            if _is_valid_lift_fast(graph, trial):
                r = _spectral_radius_signed_fast(graph, trial)
                if r < current_radius - 1e-10:
                    current_signs = trial
                    current_radius = r
                    improved = True
                    best_radius = r
                    best_signs = trial.copy()
                    break  # Restart from new best
    
    # Multi-start local search if time permits
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            s = rng.choice([-1, 1], size=num_edges).astype(int)
            if not _is_valid_lift_fast(graph, s):
                continue
            r = _spectral_radius_signed_fast(graph, s)
            
            # Local search from this start
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                trial = s.copy()
                trial[i] = -trial[i]
                if _is_valid_lift_fast(graph, trial):
                    r_trial = _spectral_radius_signed_fast(graph, trial)
                    if r_trial < r - 1e-10:
                        s = trial
                        r = r_trial
            
            if r < best_radius:
                best_radius = r
                best_signs = s.copy()
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
