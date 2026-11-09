"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit (killed at 90s wall-clock)
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build signed adjacency matrix for a given sign vector
    def build_signed_adj(signs):
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] += s
            A[v, u] += s
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """
        Check if the 2-lift is connected and 3-regular.
        The lift has 2n vertices: (v, 0) and (v, 1) for v in 0..n-1.
        For edge (u, v) with sign s:
          - if s = +1: connects (u,0)-(v,0) and (u,1)-(v,1)
          - if s = -1: connects (u,0)-(v,1) and (u,1)-(v,0)
        """
        # Build adjacency for the lift graph
        lift_adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        # Check 3-regular
        for v in range(2 * n):
            if len(lift_adj[v]) != 3:
                return False
        
        # Check connectedness via BFS
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            cur = queue.pop()
            for nb in lift_adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        return count == 2 * n
    
    # Generate a random sign vector
    rng = np.random.default_rng(3)
    
    # Try to find a good solution
    best_signs = None
    best_radius = float('inf')
    
    # First, try some structured patterns
    # All +1
    signs_all_plus = np.ones(num_edges, dtype=int)
    if is_valid_lift(signs_all_plus):
        r = spectral_radius(signs_all_plus)
        if r < best_radius:
            best_radius = r
            best_signs = signs_all_plus.copy()
    
    # Alternating patterns on cycles
    # For prism graph: two m-cycles and matching edges
    # Edges are: cycle1 (m edges), cycle2 (m edges), matching (m edges)
    # Assuming edges are ordered: first m are cycle 0, next m are cycle m, last m are matching
    
    # Try various sign patterns
    candidates = []
    
    # Pattern 1: all +1
    candidates.append(np.ones(num_edges, dtype=int))
    
    # Pattern 2: alternating on each cycle, +1 on matching
    for phase in range(2):
        signs = np.ones(num_edges, dtype=int)
        for i in range(m):
            signs[i] = 1 if i % 2 == phase else -1
            signs[m + i] = 1 if i % 2 == phase else -1
        candidates.append(signs)
    
    # Pattern 3: alternating on matching
    for phase in range(2):
        signs = np.ones(num_edges, dtype=int)
        for i in range(m):
            signs[2*m + i] = 1 if i % 2 == phase else -1
        candidates.append(signs)
    
    # Pattern 4: mix
    for p1 in range(2):
        for p2 in range(2):
            signs = np.ones(num_edges, dtype=int)
            for i in range(m):
                signs[i] = 1 if i % 2 == p1 else -1
                signs[m + i] = 1 if i % 2 == p2 else -1
            candidates.append(signs)
    
    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if is_valid_lift(cand):
            r = spectral_radius(cand)
            if r < best_radius:
                best_radius = r
                best_signs = cand.copy()
    
    # Now do random search
    num_tries = 0
    while time.time() < deadline:
        signs = rng.choice([-1, 1], size=num_edges)
        if is_valid_lift(signs):
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
        num_tries += 1
        if num_tries % 1000 == 0 and time.time() > deadline:
            break
    
    # Also try local search from best solution
    if best_signs is not None and time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                new_signs = best_signs.copy()
                new_signs[i] *= -1
                if is_valid_lift(new_signs):
                    r = spectral_radius(new_signs)
                    if r < best_radius - 1e-12:
                        best_radius = r
                        best_signs = new_signs
                        improved = True
    
    if best_signs is None:
        # Fallback: all +1 should work for prism graphs
        return np.ones(num_edges, dtype=int)
    
    return best_signs
# EVOLVE-BLOCK-END
