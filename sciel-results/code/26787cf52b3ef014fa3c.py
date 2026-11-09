import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build adjacency list for connectivity checks
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Precompute edge index lookup for fast sign assignment
    # For each edge, store its index
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
    
    def is_connected_lift(signs):
        """Check if the 2-lift is connected."""
        # Build the lift graph
        # Vertices: (v, 0) and (v, 1) for v in 0..n-1
        # Edge (u,v) with sign s: if s=+1, (u,0)-(v,0) and (u,1)-(v,1)
        # if s=-1, (u,0)-(v,1) and (u,1)-(v,0)
        
        lift_adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                lift_adj[2*u].append(2*v)
                lift_adj[2*v].append(2*u)
                lift_adj[2*u+1].append(2*v+1)
                lift_adj[2*v+1].append(2*u+1)
            else:
                lift_adj[2*u].append(2*v+1)
                lift_adj[2*v+1].append(2*u)
                lift_adj[2*u+1].append(2*v)
                lift_adj[2*v].append(2*u+1)
        
        # BFS from (0, 0)
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            u = queue.pop(0)
            for v in lift_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        
        return count == 2 * n
    
    def spectral_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    # Initial solution: random valid signs
    rng = np.random.default_rng(42)
    
    # Try to find a good initial solution
    best_signs = None
    best_radius = float('inf')
    
    # Start with a heuristic: alternate signs on the matching edges
    # This often helps with connectivity
    for init_attempt in range(100):
        signs = np.ones(num_edges, dtype=int)
        
        # Randomly flip some edges to -1
        # Ensure at least one -1 for connectivity
        num_neg = rng.integers(1, num_edges + 1)
        neg_indices = rng.choice(num_edges, size=num_neg, replace=False)
        signs[neg_indices] = -1
        
        if not is_connected_lift(signs):
            continue
        
        radius = spectral_radius(signs)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: simple valid pattern
        best_signs = np.ones(num_edges, dtype=int)
        best_signs[0] = -1
        if not is_connected_lift(best_signs):
            best_signs[1] = -1
        best_radius = spectral_radius(best_signs)
    
    # Local search / simulated annealing
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Precompute which edges form the two cycles and the matching
    # Top cycle: 0-1-2-...-(m-1)-0
    # Bottom cycle: m-(m+1)-...-(2m-1)-m
    # Matching: i-(i+m) for i in 0..m-1
    
    # Identify edge types
    top_cycle_edges = set()
    bottom_cycle_edges = set()
    matching_edges = set()
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.add(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.add(i)
        else:
            matching_edges.add(i)
    
    # Simulated annealing
    temperature = 1.0
    cooling_rate = 0.999
    min_temperature = 0.01
    
    step_count = 0
    while time.time() < deadline and temperature > min_temperature:
        # Choose a random edge to flip
        edge_idx = rng.integers(0, num_edges)
        
        # Flip the sign
        current_signs[edge_idx] = -current_signs[edge_idx]
        
        # Check connectivity
        if not is_connected_lift(current_signs):
            # Revert
            current_signs[edge_idx] = -current_signs[edge_idx]
            temperature *= cooling_rate
            continue
        
        new_radius = spectral_radius(current_signs)
        
        # Accept if better or with probability exp(-(new - old) / T)
        if new_radius < current_radius:
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_signs = current_signs.copy()
        else:
            # Metropolis criterion
            delta = new_radius - current_radius
            if delta < 0 or rng.random() < np.exp(-delta / temperature):
                current_radius = new_radius
            else:
                # Revert
                current_signs[edge_idx] = -current_signs[edge_idx]
        
        temperature *= cooling_rate
        step_count += 1
    
    return best_signs
