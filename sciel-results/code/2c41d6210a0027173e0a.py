"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def sign_edges(graph, budget):
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Precompute adjacency structure for fast signed matrix construction
    # We need to map edge index to (u, v)
    edge_list = edges
    
    def get_signed_matrix(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edge_list):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        return A

    def spectral_radius(signs):
        A = get_signed_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_connected_lift(signs):
        """
        The 2-lift is connected iff the system x_u + x_v = 1 - s_uv (mod 2) is inconsistent.
        s_uv = 1 -> 0, s_uv = -1 -> 1.
        Let b_e = 0 if s_e=1, 1 if s_e=-1.
        System: x_u + x_v = b_e for all edges (u,v).
        Consistent iff sum(b_e) over any cycle is 0 mod 2.
        """
        # Build b vector
        b = np.zeros(num_edges, dtype=int)
        for i, s in enumerate(signs):
            if s == 1:
                b[i] = 0
            else:
                b[i] = 1
        
        # Use Gaussian elimination or simple cycle check?
        # For small graphs, we can check consistency by solving the linear system over GF(2).
        # Variables: x_0, ..., x_{n-1}
        # Equations: x_u + x_v = b_e
        
        # We can pick a spanning tree and express all x in terms of x_0.
        # Then check non-tree edges.
        
        # Build adjacency list with edge indices
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edge_list):
            adj[u].append((v, i))
            adj[v].append((u, i))
            
        # BFS to find spanning tree
        visited = [False] * n
        x = [0] * n # x_0 = 0
        x[0] = 0
        visited[0] = True
        queue = [0]
        tree_edges = [False] * num_edges
        
        while queue:
            u = queue.pop(0)
            for v, idx in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    # x_u + x_v = b_idx
                    x[v] = x[u] ^ b[idx]
                    tree_edges[idx] = True
                    queue.append(v)
        
        # Check non-tree edges
        for i, (u, v) in enumerate(edge_list):
            if not tree_edges[i]:
                # Check if x_u + x_v == b_i
                if (x[u] ^ x[v]) != b[i]:
                    return True # Inconsistent -> Connected
        
        return False # Consistent -> Disconnected

    # Initial candidates
    rng = np.random.default_rng(42)
    
    # Heuristic: Try to make signs alternate to minimize spectral radius
    # For bipartite graphs, this might work well.
    candidates = []
    
    # Random starts
    for _ in range(50):
        candidates.append(rng.choice([-1, 1], size=num_edges))
        
    # Structured starts
    # All 1
    candidates.append(np.ones(num_edges, dtype=int))
    # All -1
    candidates.append(np.ones(num_edges, dtype=int) * -1)
    
    # Alternating on cycles
    for start_sign in [-1, 1]:
        signs = np.ones(num_edges, dtype=int)
        for i, (u, v) in enumerate(edge_list):
            if abs(u - v) == m:
                signs[i] = start_sign
            else:
                # Cycle edge
                # Determine position
                if u < m and v < m:
                    pos = min(u, v)
                else:
                    pos = min(u, v) - m
                signs[i] = start_sign * ((-1) ** pos)
        candidates.append(signs)
        
    best_signs = None
    best_radius = float('inf')
    
    # Evaluate initial candidates
    for signs in candidates:
        if is_connected_lift(signs):
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
                
    if best_signs is None:
        # Fallback: find any connected lift
        # Flip random edges until connected
        signs = np.ones(num_edges, dtype=int)
        while not is_connected_lift(signs):
            idx = rng.integers(0, num_edges)
            signs[idx] *= -1
        best_signs = signs.copy()
        best_radius = spectral_radius(best_signs)
        
    # Local Search
    # We want to minimize spectral radius.
    # If current is connected, try flips that keep it connected and reduce radius.
    # If current is disconnected (should not happen if we maintain invariant), fix it.
    
    max_iters = 100
    for _ in range(max_iters):
        improved = False
        for i in range(num_edges):
            trial = best_signs.copy()
            trial[i] *= -1
            
            if not is_connected_lift(trial):
                continue
                
            r = spectral_radius(trial)
            if r < best_radius:
                best_radius = r
                best_signs = trial
                improved = True
                
        if not improved:
            break
            
    return best_signs
# EVOLVE-BLOCK-END
