import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Classify edges: cycle edges (within top or bottom) vs matching edges
    # Top vertices: 0..m-1, Bottom vertices: m..2m-1
    # Matching edges connect i and i+m
    # Cycle edges connect i and i+1 (mod m) within the same half
    
    matching_indices = []
    cycle_indices = []
    matching_top_vertex = []  # The top vertex (0..m-1) for each matching edge
    
    for i, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            matching_indices.append(i)
            top = u if u < m else v
            matching_top_vertex.append(top)
        else:
            cycle_indices.append(i)
    
    n_match = len(matching_indices)
    n_cycle = len(cycle_indices)
    
    # Build the adjacency structure for spectral computation
    # We need the signed adjacency matrix of the base graph
    # The eigenvalues of the lift are the eigenvalues of the signed adjacency matrix of the base
    
    # For a prism graph with signs:
    # - Top cycle edges: sign t_i for edge (i, i+1 mod m)
    # - Bottom cycle edges: sign b_i for edge (m+i, m+i+1 mod m)  
    # - Matching edges: sign s_i for edge (i, m+i)
    
    # The signed adjacency matrix A has:
    # A[i, j] = sign of edge (i,j) if edge exists, 0 otherwise
    # A is symmetric for undirected signed graphs
    
    # Strategy: Use the fact that the spectrum depends only on the "twist" pattern
    # For prism graphs, we can parameterize by:
    # - The product of signs around the top cycle: P_top
    # - The product of signs around the bottom cycle: P_bottom
    # - The pattern of matching signs
    
    # Key insight: The signed adjacency matrix can be written in block form
    # A = [[C_top, M], [M^T, C_bottom]]
    # where C_top and C_bottom are the signed cycle adjacency matrices
    # and M is the diagonal matrix of matching signs
    
    # For a cycle of length m with signs s_0, s_1, ..., s_{m-1} (edge i connects i to i+1 mod m):
    # The eigenvalues are 2*Re(s_k * exp(2*pi*i*k/m)) for k=0,...,m-1
    # where s_k is the product of signs along the path... actually this is more complex
    
    # Let me think differently. The signed adjacency matrix of a cycle with edge signs e_0, ..., e_{m-1}
    # (edge k connects k to (k+1)%m with sign e_k) has eigenvalues:
    # lambda_k = e_k * exp(2*pi*i*k/m) + e_{k-1} * exp(-2*pi*i*k/m) ... no this isn't right either
    
    # Actually, for a signed cycle, the eigenvalues can be computed directly.
    # Let me just build the matrix and compute eigvalsh.
    
    def compute_radius(signs):
        """Compute spectral radius of the signed adjacency matrix."""
        n = 2 * m
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_connected_lift(signs):
        """Check if the 2-lift is connected.
        The lift is connected iff the product of signs around some cycle is -1."""
        # For prism graphs, we can check a few key cycles
        # Cycle 1: top cycle (0, 1, 2, ..., m-1, 0)
        # Cycle 2: bottom cycle (m, m+1, ..., 2m-1, m)
        # Cycle 3: square cycles (i, i+1, m+i+1, m+i, i) for each i
        
        # Find the top cycle edges
        top_cycle_product = 1
        bottom_cycle_product = 1
        
        for i, (u, v) in enumerate(edges):
            if abs(u - v) == m:
                continue
            if u < m and v < m:
                top_cycle_product *= signs[i]
            elif u >= m and v >= m:
                bottom_cycle_product *= signs[i]
        
        if top_cycle_product == -1 or bottom_cycle_product == -1:
            return True
        
        # Check square cycles
        # Square cycle: (i, i+1 mod m, m+(i+1 mod m), m+i, i)
        # Edges: (i, i+1 mod m) [top], (i+1 mod m, m+i+1 mod m) [matching], 
        #         (m+i+1 mod m, m+i) [bottom], (m+i, i) [matching]
        
        # Build lookup for edge signs
        edge_sign = {}
        for i, (u, v) in enumerate(edges):
            edge_sign[(u, v)] = signs[i]
            edge_sign[(v, u)] = signs[i]
        
        for i in range(m):
            j = (i + 1) % m
            # Check if all four edges exist
            e1 = edge_sign.get((i, j))
            e2 = edge_sign.get((j, m + j))
            e3 = edge_sign.get((m + j, m + i))
            e4 = edge_sign.get((m + i, i))
            if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                prod = e1 * e2 * e3 * e4
                if prod == -1:
                    return True
        
        return False
    
    # Initial best: try a few heuristic patterns
    best_radius = float('inf')
    best_signs = None
    
    # Pattern 1: All cycle edges +1, matching edges alternating
    signs1 = np.ones(n_edges, dtype=int)
    for idx, top in zip(matching_indices, matching_top_vertex):
        signs1[idx] = -1 if top % 2 == 0 else 1
    if is_connected_lift(signs1):
        r = compute_radius(signs1)
        if r < best_radius:
            best_radius = r
            best_signs = signs1.copy()
    
    # Pattern 2: All cycle edges +1, all matching edges -1 (may not be connected for even m)
    signs2 = np.ones(n_edges, dtype=int)
    for idx in matching_indices:
        signs2[idx] = -1
    if is_connected_lift(signs2):
        r = compute_radius(signs2)
        if r < best_radius:
            best_radius = r
            best_signs = signs2.copy()
    
    # Pattern 3: Top cycle all -1, bottom all +1, matching all +1
    signs3 = np.ones(n_edges, dtype=int)
    for idx in cycle_indices:
        u, v = edges[idx]
        if u < m and v < m:
            signs3[idx] = -1
    if is_connected_lift(signs3):
        r = compute_radius(signs3)
        if r < best_radius:
            best_radius = r
            best_signs = signs3.copy()
    
    # Pattern 4: Top cycle alternating, bottom all +1, matching all +1
    signs4 = np.ones(n_edges, dtype=int)
    top_cycle_edges = []
    for idx in cycle_indices:
        u, v = edges[idx]
        if u < m and v < m:
            top_cycle_edges.append((idx, u))
    top_cycle_edges.sort(key=lambda x: x[1])
    for k, (idx, u) in enumerate(top_cycle_edges):
        if k % 2 == 0:
            signs4[idx] = -1
    if is_connected_lift(signs4):
        r = compute_radius(signs4)
        if r < best_radius:
            best_radius = r
            best_signs = signs4.copy()
    
    # Now do local search / simulated annealing
    if best_signs is None:
        # Fallback: use pattern 1
        best_signs = signs1.copy()
        best_radius = compute_radius(signs1)
    
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Simulated annealing
    if time.time() < deadline - 5.0:
        T_start = 0.5
        T_end = 0.01
        n_iter = 50000
        elapsed = time.time() - start_time
        max_iter = int(n_iter * (deadline - 5.0 - elapsed) / (deadline - 5.0 - start_time))
        max_iter = max(max_iter, 1000)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            T = T_start * (T_end / T_start) ** (it / max_iter)
            
            # Pick a random edge to flip
            idx = np.random.randint(n_edges)
            old_sign = current_signs[idx]
            current_signs[idx] = -old_sign
            
            new_radius = compute_radius(current_signs)
            
            if new_radius < current_radius or np.random.random() < np.exp(-(new_radius - current_radius) / T):
                current_radius = new_radius
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_signs = current_signs.copy()
            else:
                current_signs[idx] = old_sign
        
        # Verify best is connected
        if not is_connected_lift(best_signs):
            # Fallback to a known connected pattern
            best_signs = signs1.copy()
            best_radius = compute_radius(signs1)
    
    return best_signs
