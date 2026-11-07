import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin for 60s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Precompute adjacency list for fast BFS
    # adj[u] = list of (v, edge_index)
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    def is_connected_lift(signs):
        """
        Check if the 2-lift is connected.
        The lift is disconnected iff the signed graph is balanced.
        Balanced iff there exists x in {-1, 1}^n such that x_u * x_v = s_uv.
        We check this by BFS assigning values to x.
        """
        x = [0] * n
        x[0] = 1
        visited = [False] * n
        visited[0] = True
        stack = [0]
        
        while stack:
            u = stack.pop()
            for v, i in adj[u]:
                s = signs[i]
                if not visited[v]:
                    x[v] = x[u] * s
                    visited[v] = True
                    stack.append(v)
                else:
                    # Check consistency
                    if x[u] * s != x[v]:
                        return True # Inconsistent, so not balanced, so connected
        
        # If we reach here, it was consistent (balanced)
        # But we must ensure all nodes were visited (graph is connected in base)
        # Prism graphs are connected, so if base is connected, visited should be all True
        if not all(visited):
             # Should not happen
             return True 
        
        return False # Balanced, so lift is disconnected

    def compute_radius(signs):
        """Compute spectral radius of the signed adjacency matrix."""
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    # Initial best
    best_signs = None
    best_radius = np.inf
    
    rng = np.random.default_rng(42)

    # Generate initial candidates
    # 1. Random
    # 2. Alternating patterns
    
    def evaluate(signs):
        if not is_connected_lift(signs):
            return np.inf
        return compute_radius(signs)

    # Simulated Annealing
    def simulated_annealing(initial_signs, time_limit):
        current_signs = initial_signs.copy()
        current_radius = evaluate(current_signs)
        
        if np.isinf(current_radius):
            # Try to find a valid one
            for _ in range(1000):
                trial = rng.choice([-1, 1], size=E).astype(np.int8)
                if is_connected_lift(trial):
                    current_signs = trial
                    current_radius = compute_radius(trial)
                    break
            if np.isinf(current_radius):
                return current_signs, np.inf

        best_signs_local = current_signs.copy()
        best_radius_local = current_radius
        
        T = 1.0
        cooling = 0.999
        start_sa = time.time()
        
        while time.time() < start_sa + time_limit:
            # Pick a random edge to flip
            edge_idx = int(rng.integers(0, E))
            trial_signs = current_signs.copy()
            trial_signs[edge_idx] = -trial_signs[edge_idx]
            
            if not is_connected_lift(trial_signs):
                continue # Reject invalid moves immediately
            
            trial_radius = compute_radius(trial_signs)
            delta = trial_radius - current_radius
            
            if delta < 0 or (rng.random() < np.exp(-delta / T) if T > 1e-6 else False):
                current_signs = trial_signs
                current_radius = trial_radius
                
                if current_radius < best_radius_local:
                    best_radius_local = current_radius
                    best_signs_local = current_signs.copy()
                    
            T *= cooling
            
        return best_signs_local, best_radius_local

    # Run multiple restarts
    num_restarts = 0
    max_restarts = 100
    
    while time.time() < deadline - 5.0 and num_restarts < max_restarts:
        num_restarts += 1
        
        # Generate a diverse initial state
        # Strategy: Start with random, or specific patterns
        if num_restarts % 3 == 0:
            # Alternating pattern
            init = np.ones(E, dtype=np.int8)
            # Flip every other edge in bottom cycle
            bottom_idx = 0
            for i, (u, v) in enumerate(edges):
                if u < m and v < m:
                    if bottom_idx % 2 == 0:
                        init[i] = -1
                    bottom_idx += 1
        else:
            init = rng.choice([-1, 1], size=E).astype(np.int8)
            
        # Ensure initial state is valid, if not, fix it
        if not is_connected_lift(init):
             # Try to find a valid neighbor
             found = False
             for _ in range(100):
                 trial = init.copy()
                 e_idx = int(rng.integers(0, E))
                 trial[e_idx] = -trial[e_idx]
                 if is_connected_lift(trial):
                     init = trial
                     found = True
                     break
             if not found:
                 # Just use random until valid
                 while not is_connected_lift(init):
                     init = rng.choice([-1, 1], size=E).astype(np.int8)
        
        # Run SA
        # Adaptive time allocation: use remaining time / remaining restarts
        remaining_time = deadline - 5.0 - time.time()
        time_per_restart = remaining_time / max(1, max_restarts - num_restarts)
        
        # We need to adjust max_iters based on speed. 
        # Let's just run for a fixed number of iterations that fits in time_per_restart
        # We can't easily estimate iterations without running. 
        # So we will run a fixed number of iterations, e.g., 5000, and check time.
        
        # Better: Run SA in a loop that checks time.
        local_signs, local_radius = simulated_annealing(init, time_per_restart * 0.9)
        
        if local_radius < best_radius:
            best_radius = local_radius
            best_signs = local_signs.copy()

    if best_signs is None:
        # Fallback
        best_signs = rng.choice([-1, 1], size=E).astype(np.int8)
        while not is_connected_lift(best_signs):
            best_signs = rng.choice([-1, 1], size=E).astype(np.int8)

    return best_signs
