"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Build edge index mapping
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    def build_signed_adj(signs):
        """Build signed adjacency matrix from signs."""
        n = graph['n']
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        n = graph['n']
        # Build the lift graph
        # Vertices: (v, 0) and (v, 1) for v in 0..n-1
        # Total 2n vertices in lift
        lift_n = 2 * n
        
        # Build adjacency for lift
        lift_adj = [[] for _ in range(lift_n)]
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                # +1: joins equal sides
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # -1: joins opposite sides
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        # Check 3-regular
        for i in range(lift_n):
            if len(lift_adj[i]) != 3:
                return False
        
        # Check connectedness using BFS
        visited = [False] * lift_n
        queue = [0]
        visited[0] = True
        count = 1
        
        while queue:
            node = queue.pop(0)
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == lift_n
    
    def try_signs(signs):
        """Try a sign configuration, return (valid, radius) or (False, inf)."""
        if not is_valid_lift(signs):
            return False, float('inf')
        r = spectral_radius(signs)
        return True, r
    
    # Strategy: Use spectral theory insights
    # For prism graphs, we want to minimize the spectral radius
    # Key insight: alternating signs on cycles can reduce spectral radius
    
    best_signs = None
    best_radius = float('inf')
    
    # Try structured patterns first
    patterns = []
    
    # Pattern 1: All +1
    patterns.append(np.ones(n_edges, dtype=np.int8))
    
    # Pattern 2: All -1
    patterns.append(-np.ones(n_edges, dtype=np.int8))
    
    # Pattern 3: Alternate on top cycle, +1 on matching, alternate on bottom
    for phase in [0, 1]:
        signs = np.ones(n_edges, dtype=np.int8)
        # Need to figure out edge ordering
        # Assume: top cycle edges first, then matching, then bottom cycle?
        # Or some other ordering - let's try to infer from the graph structure
        
        # For now, try various systematic patterns
        for i in range(n_edges):
            signs[i] = 1 if (i % 2 == phase) else -1
        patterns.append(signs)
    
    # Pattern 4: Top cycle alternating, matching +1, bottom cycle alternating
    # We need to identify which edges belong to which part
    # Let's try to infer edge types
    
    # Identify edge types based on vertex ranges
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    # Pattern: Alternate on cycles, +1 on matching
    for top_phase in [0, 1]:
        for bottom_phase in [0, 1]:
            signs = np.ones(n_edges, dtype=np.int8)
            for j, idx in enumerate(top_cycle_edges):
                signs[idx] = 1 if (j % 2 == top_phase) else -1
            for j, idx in enumerate(bottom_cycle_edges):
                signs[idx] = 1 if (j % 2 == bottom_phase) else -1
            # matching stays +1
            patterns.append(signs)
    
    # Pattern: Alternate on cycles, -1 on matching
    for top_phase in [0, 1]:
        for bottom_phase in [0, 1]:
            signs = np.ones(n_edges, dtype=np.int8)
            for j, idx in enumerate(top_cycle_edges):
                signs[idx] = 1 if (j % 2 == top_phase) else -1
            for j, idx in enumerate(bottom_cycle_edges):
                signs[idx] = 1 if (j % 2 == bottom_phase) else -1
            for idx in matching_edges:
                signs[idx] = -1
            patterns.append(signs)
    
    # Evaluate all patterns
    for signs in patterns:
        if time.time() > deadline:
            break
        valid, radius = try_signs(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # If no valid pattern found, use random search
    if best_signs is None:
        rng = np.random.default_rng(3)
        for _ in range(budget):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=n_edges).astype(np.int8)
            valid, radius = try_signs(signs)
            if valid and radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    
    # Local search to improve
    if best_signs is not None and time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(n_edges):
                if time.time() > deadline - 5.0:
                    break
                trial = best_signs.copy()
                trial[i] = -trial[i]
                valid, radius = try_signs(trial)
                if valid and radius < best_radius:
                    best_radius = radius
                    best_signs = trial
                    improved = True
    
    # If still no valid solution, return a default (shouldn't happen)
    if best_signs is None:
        best_signs = np.ones(n_edges, dtype=np.int8)
    
    return best_signs
# EVOLVE-BLOCK-END
