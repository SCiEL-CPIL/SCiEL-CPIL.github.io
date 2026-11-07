"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _spectral_radius_signed(m, bottom_signs, top_signs, matching_signs):
    """Compute spectral radius of signed adjacency matrix of prism graph using block structure."""
    n = 2 * m
    # Build the signed adjacency matrix
    A = np.zeros((n, n), dtype=float)
    
    for i in range(m):
        j = (i + 1) % m
        s = float(bottom_signs[i])
        A[i, j] = s
        A[j, i] = s
    
    for i in range(m):
        j = m + (i + 1) % m
        s = float(top_signs[i])
        A[m + i, j] = s
        A[j, m + i] = s
    
    for i in range(m):
        s = float(matching_signs[i])
        A[i, m + i] = s
        A[m + i, i] = s
    
    eigvals = np.linalg.eigvalsh(A)
    return max(abs(eigvals[0]), abs(eigvals[-1]))

def _check_lift_connected_fast(n, edges, signs):
    """Fast connectivity check for the 2-lift without importing check_lift."""
    # Build adjacency for the lift graph
    # Vertices: (v, 0) and (v, 1) -> index 2*v and 2*v+1
    lift_adj = [[] for _ in range(2 * n)]
    
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            # +1: joins equal sides
            lift_adj[2*u].append(2*v)
            lift_adj[2*v].append(2*u)
            lift_adj[2*u+1].append(2*v+1)
            lift_adj[2*v+1].append(2*u+1)
        else:
            # -1: joins opposite sides
            lift_adj[2*u].append(2*v+1)
            lift_adj[2*v+1].append(2*u)
            lift_adj[2*u+1].append(2*v)
            lift_adj[2*v].append(2*u+1)
    
    # BFS/DFS to check connectivity
    visited = [False] * (2 * n)
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        node = stack.pop()
        for neighbor in lift_adj[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                count += 1
                stack.append(neighbor)
    
    return count == 2 * n

def _is_valid_lift(graph, signs):
    """Check if the lift is valid (connected, 3-regular)."""
    n = graph['n']
    edges = graph['edges']
    if len(signs) != len(edges):
        return False
    # Check all signs are ±1
    if not all(s in (-1, 1) for s in signs):
        return False
    # Check connectivity of the lift
    return _check_lift_connected_fast(n, edges, signs)

def _extract_cycle_signs(graph, m):
    """Extract edge maps for bottom cycle, top cycle, and matching."""
    edges = graph['edges']
    bottom_edge_map = {}
    top_edge_map = {}
    matching_edge_map = {}
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            a, b = min(u, v), max(u, v)
            if b == (a + 1) % m:
                pos = a
            else:
                pos = b
            bottom_edge_map[i] = pos
        elif u >= m and v >= m:
            a, b = min(u - m, v - m), max(u - m, v - m)
            if b == (a + 1) % m:
                pos = a
            else:
                pos = b
            top_edge_map[i] = pos
        else:
            pos = min(u, v) % m
            matching_edge_map[i] = pos
    
    return bottom_edge_map, top_edge_map, matching_edge_map

def _signs_to_components(signs, bottom_edge_map, top_edge_map, matching_edge_map, m):
    """Convert full signs array to (bottom_signs, top_signs, matching_signs)."""
    bottom_signs = np.ones(m, dtype=int)
    top_signs = np.ones(m, dtype=int)
    matching_signs = np.ones(m, dtype=int)
    
    for edge_idx, pos in bottom_edge_map.items():
        bottom_signs[pos] = signs[edge_idx]
    for edge_idx, pos in top_edge_map.items():
        top_signs[pos] = signs[edge_idx]
    for edge_idx, pos in matching_edge_map.items():
        matching_signs[pos] = signs[edge_idx]
    
    return bottom_signs, top_signs, matching_signs

def _make_frustrated_signs(m, bottom_edge_map, top_edge_map, matching_edge_map, num_edges, 
                           bottom_frustrate_pos, top_frustrate_pos, matching_pattern):
    """Create sign pattern with frustrated cycles (product = -1)."""
    signs = np.ones(num_edges, dtype=int)
    
    # Frustrate bottom cycle: flip one edge
    if bottom_frustrate_pos is not None:
        for edge_idx, pos in bottom_edge_map.items():
            if pos == bottom_frustrate_pos:
                signs[edge_idx] = -1
                break
    
    # Frustrate top cycle: flip one edge
    if top_frustrate_pos is not None:
        for edge_idx, pos in top_edge_map.items():
            if pos == top_frustrate_pos:
                signs[edge_idx] = -1
                break
    
    # Apply matching pattern
    for i in range(m):
        if matching_pattern[i] == -1:
            for edge_idx, pos in matching_edge_map.items():
                if pos == i:
                    signs[edge_idx] = -1
                    break
    
    return signs

def _local_search(graph, signs, deadline, max_rounds=20):
    """Local search: flip one or two edges at a time to improve spectral radius."""
    m = graph['n'] // 2
    n = graph['n']
    edges = graph['edges']
    bottom_edge_map, top_edge_map, matching_edge_map = _extract_cycle_signs(graph, m)
    
    current_signs = signs.copy()
    b_s, t_s, m_s = _signs_to_components(current_signs, bottom_edge_map, top_edge_map, matching_edge_map, m)
    current_radius = _spectral_radius_signed(m, b_s, t_s, m_s)
    num_edges = len(current_signs)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        
        # Single flip
        for i in range(num_edges):
            if time.time() > deadline:
                break
            trial = current_signs.copy()
            trial[i] = -trial[i]
            
            if _is_valid_lift(graph, trial):
                b_t, t_t, m_t = _signs_to_components(trial, bottom_edge_map, top_edge_map, matching_edge_map, m)
                r = _spectral_radius_signed(m, b_t, t_t, m_t)
                if r < current_radius - 1e-12:
                    current_signs = trial
                    b_s, t_s, m_s = b_t, t_t, m_t
                    current_radius = r
                    improved = True
        
        # Double flip (pairwise) - only if no single flip improved
        if not improved:
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                for j in range(i+1, num_edges):
                    if time.time() > deadline:
                        break
                    trial = current_signs.copy()
                    trial[i] = -trial[i]
                    trial[j] = -trial[j]
                    
                    if _is_valid_lift(graph, trial):
                        b_t, t_t, m_t = _signs_to_components(trial, bottom_edge_map, top_edge_map, matching_edge_map, m)
                        r = _spectral_radius_signed(m, b_t, t_t, m_t)
                        if r < current_radius - 1e-12:
                            current_signs = trial
                            b_s, t_s, m_s = b_t, t_t, m_t
                            current_radius = r
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return current_signs, current_radius

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    bottom_edge_map, top_edge_map, matching_edge_map = _extract_cycle_signs(graph, m)
    
    best_signs = None
    best_radius = np.inf
    
    def evaluate_signs(s):
        """Evaluate a sign vector, return (radius, valid) or (inf, False)."""
        if not _is_valid_lift(graph, s):
            return np.inf, False
        b_s, t_s, m_s = _signs_to_components(s, bottom_edge_map, top_edge_map, matching_edge_map, m)
        r = _spectral_radius_signed(m, b_s, t_s, m_s)
        return r, True
    
    # Generate candidates with frustrated cycles
    candidates = []
    
    # For each combination of frustration positions and matching patterns
    # Use more matching patterns for larger m
    num_matching_patterns = min(2 ** m, max(16, 4 * m))
    for b_pos in range(m + 1):  # +1 means no frustration
        if time.time() > deadline - 30:
            break
        for t_pos in range(m + 1):
            for mask in range(num_matching_patterns):
                matching_pattern = np.ones(m, dtype=int)
                for i in range(m):
                    if mask & (1 << i):
                        matching_pattern[i] = -1
                
                b_frustrate = b_pos if b_pos < m else None
                t_frustrate = t_pos if t_pos < m else None
                
                s = _make_frustrated_signs(m, bottom_edge_map, top_edge_map, matching_edge_map, 
                                          num_edges, b_frustrate, t_frustrate, matching_pattern)
                candidates.append(s)
    
    # Add random candidates
    rng = np.random.default_rng(42)
    for _ in range(300):
        if time.time() > deadline - 30:
            break
        s = rng.choice([-1, 1], size=num_edges).astype(int)
        candidates.append(s)
    
    # Evaluate candidates
    candidate_results = []
    for s in candidates:
        if time.time() > deadline - 30:
            break
        r, valid = evaluate_signs(s)
        if valid:
            candidate_results.append((r, s.copy()))
    
    candidate_results.sort(key=lambda x: x[0])
    
    for r, s in candidate_results[:30]:
        if r < best_radius:
            best_radius = r
            best_signs = s.copy()
    
    # Local search on top candidates
    for r, s in candidate_results[:20]:
        if time.time() > deadline - 10:
            break
        refined_signs, refined_radius = _local_search(graph, s, deadline, max_rounds=20)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_signs = refined_signs.copy()
    
    # Simulated annealing for better exploration
    if best_signs is not None and time.time() < deadline - 15:
        current_signs = best_signs.copy()
        current_radius = best_radius
        sa_start_temp = 0.5
        sa_end_temp = 0.01
        num_sa_steps = 500
        rng_sa = np.random.default_rng(123)
        
        for step in range(num_sa_steps):
            if time.time() > deadline - 5:
                break
            temp = sa_start_temp * (sa_end_temp / sa_start_temp) ** (step / num_sa_steps)
            
            # Flip 1-3 random edges
            num_flips = rng_sa.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            flip_indices = rng_sa.choice(num_edges, size=num_flips, replace=False)
            
            trial = current_signs.copy()
            for idx in flip_indices:
                trial[idx] = -trial[idx]
            
            r, valid = evaluate_signs(trial)
            if valid:
                delta = r - current_radius
                if delta < 0 or rng_sa.random() < np.exp(-delta / max(temp, 1e-10)):
                    current_signs = trial
                    current_radius = r
                    if r < best_radius:
                        best_radius = r
                        best_signs = trial.copy()
    
    # If no valid candidate found, try random search
    if best_signs is None:
        rng = np.random.default_rng(0)
        for _ in range(500):
            if time.time() > deadline:
                break
            s = rng.choice([-1, 1], size=num_edges).astype(int)
            if _is_valid_lift(graph, s):
                b_s, t_s, m_s = _signs_to_components(s, bottom_edge_map, top_edge_map, matching_edge_map, m)
                r = _spectral_radius_signed(m, b_s, t_s, m_s)
                if r < best_radius:
                    best_radius = r
                    best_signs = s.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
    
    # Final local search polish
    if time.time() < deadline - 3.0:
        refined_signs, refined_radius = _local_search(graph, best_signs, deadline, max_rounds=25)
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_signs = refined_signs.copy()
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
