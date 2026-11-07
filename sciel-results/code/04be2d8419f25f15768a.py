import numpy as np
import time

def sign_edges(graph, budget):
    """
    Minimizes the spectral radius of the signed adjacency matrix of the prism graph.
    """
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Pre-compute edge endpoints for speed
    u_list = np.array([e[0] for e in edges])
    v_list = np.array([e[1] for e in edges])
    
    def build_adjacency(signs):
        """Build signed adjacency matrix."""
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(num_edges):
            s = signs[i]
            u = u_list[i]
            v = v_list[i]
            A[u, v] = s
            A[v, u] = s
        return A

    def is_lift_connected(signs):
        """
        Check if the 2-lift is connected.
        Lift vertices: (v, 0) and (v, 1) for v in 0..n-1.
        Edge (u, v) with sign s:
        if s == 1: (u,0)-(v,0) and (u,1)-(v,1)
        if s == -1: (u,0)-(v,1) and (u,1)-(v,0)
        """
        # BFS on the lift graph
        # Node index in lift: v*2 + side
        visited = [False] * (2 * n)
        queue = [0]  # Start at (0, 0)
        visited[0] = True
        count = 1
        
        # Pre-compute adjacency list for the lift? 
        # Since n is small, we can just iterate edges for each node in queue.
        # Or build a small adjacency list for the lift.
        lift_adj = [[] for _ in range(2 * n)]
        for i in range(num_edges):
            s = signs[i]
            u = u_list[i]
            v = v_list[i]
            
            u0, u1 = u * 2, u * 2 + 1
            v0, v1 = v * 2, v * 2 + 1
            
            if s == 1:
                lift_adj[u0].append(v0)
                lift_adj[v0].append(u0)
                lift_adj[u1].append(v1)
                lift_adj[v1].append(u1)
            else:
                lift_adj[u0].append(v1)
                lift_adj[v1].append(u0)
                lift_adj[u1].append(v0)
                lift_adj[v0].append(u1)
        
        while queue:
            curr = queue.pop(0)
            for nei in lift_adj[curr]:
                if not visited[nei]:
                    visited[nei] = True
                    count += 1
                    queue.append(nei)
                    
        return count == 2 * n

    def get_radius_and_validity(signs):
        if not is_lift_connected(signs):
            return float('inf'), False
        A = build_adjacency(signs)
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        return radius, True

    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    best_radius = float('inf')
    best_signs = None
    
    # Heuristic initial guess: Alternating signs on cycles, +1 on matching
    # This often provides a good starting point.
    # Top cycle edges: indices 0 to m-1? 
    # The problem says "ordered list". 
    # Structure: two m-cycles and matching.
    # Usually: 
    # 0..m-1: top cycle edges (i, i+1)
    # m..2m-1: bottom cycle edges (m+i, m+i+1)
    # 2m..3m-1: matching edges (i, m+i)
    # This is a common convention. Let's assume this or just use random.
    # Random is safer if order is not guaranteed, but "prism graphs... joined by matching" implies a specific structure.
    # Let's try a few structured inits and random ones.
    
    inits = []
    
    # Init 1: All +1
    inits.append(np.ones(num_edges, dtype=int))
    
    # Init 2: Alternating on all edges (might not be valid or optimal)
    alt = np.array([1 if i % 2 == 0 else -1 for i in range(num_edges)], dtype=int)
    inits.append(alt)
    
    # Init 3: Top +, Bottom -, Match +
    # Assuming order: top, bottom, match
    init3 = np.ones(num_edges, dtype=int)
    if num_edges == 3 * m:
        init3[m:2*m] = -1
    inits.append(init3)
    
    # Random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(rng.choice([-1, 1], size=num_edges).astype(int))
        
    for init_signs in inits:
        if time.time() > deadline:
            break
            
        current_signs = init_signs.copy()
        current_radius, valid = get_radius_and_validity(current_signs)
        
        if not valid:
            # Try to fix validity or just skip if it stays invalid?
            # Local search might fix it.
            pass
            
        improved = True
        while improved:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                    
                # Try flipping sign i
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                new_radius, new_valid = get_radius_and_validity(current_signs)
                
                if new_valid and new_radius < current_radius - 1e-9:
                    current_radius = new_radius
                    improved = True
                else:
                    # Revert
                    current_signs[i] = old_sign
                    # If we were valid and became invalid, we revert.
                    # If we were invalid and became valid but radius is worse? 
                    # We only accept if valid AND better.
                    # If current was invalid, current_radius is inf. 
                    # If new is valid, new_radius < inf, so we accept.
                    # So this logic handles invalid->valid transitions.
                    
            if time.time() > deadline:
                break
                
        if valid and current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()
            
    # If no valid solution found (unlikely), return a default valid one
    # A known valid configuration: All +1 is always valid?
    # Lift of all +1: Two disjoint copies of the graph. NOT CONNECTED.
    # So All +1 is INVALID.
    # We need at least one -1 to connect the two sides?
    # Actually, if there is any path with product of signs -1, the lift connects the two sides?
    # Yes. If the signed graph is not "balanced" (i.e., not all cycles have product +1), the lift is connected.
    # For a prism, if we set one edge to -1, the lift becomes connected.
    
    if best_signs is None:
        # Fallback: Set first edge to -1, rest +1
        fallback = np.ones(num_edges, dtype=int)
        fallback[0] = -1
        # Verify
        r, v = get_radius_and_validity(fallback)
        if v:
            best_signs = fallback
        else:
            # Try random until valid
            while True:
                cand = rng.choice([-1, 1], size=num_edges).astype(int)
                r, v = get_radius_and_validity(cand)
                if v:
                    best_signs = cand
                    break
    
    return best_signs
