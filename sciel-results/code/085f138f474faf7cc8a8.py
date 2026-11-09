"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build edge index lookup for cycle edges
    # For prism graphs, we expect:
    # Top cycle edges: (i, i+1) for i=0..m-2, and (m-1, 0)
    # Bottom cycle edges: (m+i, m+i+1) for i=0..m-2, and (m+m-1, m)
    # Matching edges: (i, m+i) for i=0..m-1
    
    top_cycle_edge_indices = []
    bottom_cycle_edge_indices = []
    matching_edge_indices = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edge_indices.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edge_indices.append(i)
        else:
            matching_edge_indices.append(i)
            
    # Map vertex to edge indices for top and bottom cycles
    top_adj = [[] for _ in range(m)]
    for idx in top_cycle_edge_indices:
        u, v = edges[idx]
        top_adj[u].append((v, idx))
        top_adj[v].append((u, idx))
        
    bottom_adj = [[] for _ in range(m)]
    for idx in bottom_cycle_edge_indices:
        u, v = edges[idx]
        bottom_adj[u-m].append((v-m, idx))
        bottom_adj[v-m].append((u-m, idx))
    
    # Build signed adjacency matrix quickly
    # Pre-compute edge endpoints as arrays for speed
    edge_u = np.array([u for u, v in edges])
    edge_v = np.array([v for u, v in edges])
    
    def get_signed_adj(signs):
        A = np.zeros((n, n))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        return A

    def spectral_radius(signs):
        A = get_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_connected_lift(signs):
        # Build graph of the lift
        # Vertices 0..2n-1. (v,0) is v, (v,1) is v+n.
        # Edge (u,v) with sign s:
        # if s=1: (u,0)-(v,0) and (u,1)-(v,1)
        # if s=-1: (u,0)-(v,1) and (u,1)-(v,0)
        
        adj_lift = [[] for _ in range(2*n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                adj_lift[u].append(v)
                adj_lift[v].append(u)
                adj_lift[u+n].append(v+n)
                adj_lift[v+n].append(u+n)
            else:
                adj_lift[u].append(v+n)
                adj_lift[v+n].append(u)
                adj_lift[u+n].append(v)
                adj_lift[v].append(u+n)
                
        # BFS from 0
        visited = [False] * (2*n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for nei in adj_lift[node]:
                if not visited[nei]:
                    visited[nei] = True
                    count += 1
                    queue.append(nei)
        return count == 2*n

    # Structural approach: 
    # The signed adjacency matrix A_s = A * S where S is the sign matrix.
    # For a 2-lift, the eigenvalues of the lift are the eigenvalues of A_s.
    # We want to minimize the spectral radius of A_s.
    
    # Key insight: The prism graph has a symmetric structure.
    # We can parameterize signs by:
    # 1. Signs on top cycle edges (m edges)
    # 2. Signs on bottom cycle edges (m edges)  
    # 3. Signs on matching edges (m edges)
    
    # Due to symmetry, we can try to find patterns that minimize the spectral radius.
    # One approach: use the fact that for a circulant-like structure, 
    # certain sign patterns (e.g., alternating) can reduce the spectral radius.
    
    # Let's try a structured search:
    # 1. Start with all -1 (known to be connected for m=5,7,9)
    # 2. For m=4, start with a pattern that ensures connectivity
    # 3. Use local search but with a better initialization and more systematic flips
    
    best_radius = float('inf')
    best_signs = None
    
    # Try several initial patterns
    initial_patterns = []
    
    # All -1
    initial_patterns.append(np.full(num_edges, -1))
    
    # All +1
    initial_patterns.append(np.full(num_edges, 1))
    
    # Alternating on cycles
    for pattern_type in ['top_alt', 'bottom_alt', 'both_alt']:
        signs = np.full(num_edges, 1)
        if pattern_type in ['top_alt', 'both_alt']:
            for i, idx in enumerate(top_cycle_edge_indices):
                signs[idx] = -1 if i % 2 == 0 else 1
        if pattern_type in ['bottom_alt', 'both_alt']:
            for i, idx in enumerate(bottom_cycle_edge_indices):
                signs[idx] = -1 if i % 2 == 0 else 1
        initial_patterns.append(signs)
    
    # Matching edges all -1, cycles all +1
    signs = np.full(num_edges, 1)
    for idx in matching_edge_indices:
        signs[idx] = -1
    initial_patterns.append(signs)
    
    # Matching edges all +1, cycles all -1
    signs = np.full(num_edges, -1)
    for idx in matching_edge_indices:
        signs[idx] = 1
    initial_patterns.append(signs)
    
    # Random patterns
    rng = np.random.default_rng(3)
    for _ in range(10):
        initial_patterns.append(rng.choice([-1, 1], size=num_edges))
    
    # Local search with simulated annealing
    for init_signs in initial_patterns:
        if time.time() > deadline:
            break
            
        curr = init_signs.copy()
        
        # Check connectivity
        if not is_connected_lift(curr):
            # Try to fix connectivity
            fixed = False
            for _ in range(200):
                if is_connected_lift(curr):
                    fixed = True
                    break
                idx = rng.integers(0, num_edges)
                curr[idx] *= -1
            if not fixed:
                continue
                
        curr_rad = spectral_radius(curr)
        
        # Simulated annealing
        T = 0.5
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T and time.time() < deadline:
            # Pick a random edge to flip
            idx = rng.integers(0, num_edges)
            new_signs = curr.copy()
            new_signs[idx] *= -1
            
            # Check connectivity
            if not is_connected_lift(new_signs):
                continue
                
            new_rad = spectral_radius(new_signs)
            delta = new_rad - curr_rad
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or rng.random() < np.exp(-delta / T):
                curr = new_signs
                curr_rad = new_rad
                
            T *= cooling
            
            # Update best
            if curr_rad < best_radius:
                best_radius = curr_rad
                best_signs = curr.copy()
                
    # If no valid solution found, use fallback
    if best_signs is None:
        # Try to find any valid solution
        for i in range(num_edges):
            signs = np.full(num_edges, -1)
            signs[i] = 1
            if is_connected_lift(signs):
                best_signs = signs
                break
        if best_signs is None:
            # Last resort: all +1 (may be disconnected but return something)
            best_signs = np.full(num_edges, 1)
            
    return best_signs
# EVOLVE-BLOCK-END
