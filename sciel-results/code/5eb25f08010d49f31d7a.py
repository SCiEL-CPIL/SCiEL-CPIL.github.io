import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit (5s buffer)
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    e_count = len(edges)
    
    # Build adjacency for spanning tree
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Find spanning tree using BFS
    visited = [False] * n
    tree_edge_indices = set()
    queue = [0]
    visited[0] = True
    
    while queue:
        u = queue.pop(0)
        for v, edge_idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edge_indices.add(edge_idx)
                queue.append(v)
                
    # Chords are edges not in the spanning tree
    chords = [i for i in range(e_count) if i not in tree_edge_indices]
    num_chords = len(chords)
    
    best_radius = float('inf')
    best_signs = np.ones(e_count, dtype=int)
    
    # Phase synchronization approach: parameterize vertices with phases,
    # optimize continuous phases to minimize spectral radius, then discretize
    
    def signs_to_radius(signs):
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    # Precompute edge endpoints as arrays for speed
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)
    
    # Phase optimization: minimize spectral radius over continuous phases
    # z_v = exp(i * theta_v), edge sign = sign(Re(z_u * conj(z_v))) = sign(cos(theta_u - theta_v))
    
    def phase_to_signs(theta):
        # theta: array of n phases
        diffs = theta[edge_u] - theta[edge_v]
        cos_vals = np.cos(diffs)
        # sign: +1 if cos > 0, -1 if cos < 0
        # For ties (cos = 0), use +1
        return np.where(cos_vals >= 0, 1, -1).astype(int)
    
    def spectral_radius_from_phases(theta):
        signs = phase_to_signs(theta)
        A = np.zeros((n, n), dtype=float)
        for i in range(e_count):
            s = signs[i]
            A[edge_u[i], edge_v[i]] = s
            A[edge_v[i], edge_u[i]] = s
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    # Gradient descent on phases (smooth surrogate: use soft-max of eigenvalue magnitudes)
    def soft_spectral_radius(theta, temperature=0.1):
        signs = phase_to_signs(theta)
        A = np.zeros((n, n), dtype=float)
        for i in range(e_count):
            s = signs[i]
            A[edge_u[i], edge_v[i]] = s
            A[edge_v[i], edge_u[i]] = s
        eigvals = np.linalg.eigvalsh(A)
        abs_eigs = np.abs(eigvals)
        # Log-sum-exp for smooth max
        max_eig = np.max(abs_eigs)
        log_sum_exp = max_eig + np.log(np.sum(np.exp((abs_eigs - max_eig) / temperature))) * temperature
        return log_sum_exp
    
    def soft_spectral_radius_grad(theta, temperature=0.1):
        # Numerical gradient for soft spectral radius
        eps = 1e-4
        grad = np.zeros(n)
        base_val = soft_spectral_radius(theta, temperature)
        for j in range(n):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            val_plus = soft_spectral_radius(theta_plus, temperature)
            grad[j] = (val_plus - base_val) / eps
        return grad
    
    # Multiple random restarts for phase optimization
    rng = np.random.default_rng(42)
    
    # First, try exhaustive chord enumeration (fast for small chord counts)
    if num_chords <= 10:
        for mask in range(1 << num_chords):
            if time.time() > deadline:
                break
                
            signs = np.ones(e_count, dtype=int)
            
            for k, edge_idx in enumerate(chords):
                if mask & (1 << k):
                    signs[edge_idx] = -1
            
            radius = signs_to_radius(signs)
            
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    
    # Phase-based optimization: multiple restarts
    num_restarts = 20
    steps_per_restart = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize phases
        if restart == 0:
            theta = np.zeros(n)
        else:
            theta = rng.uniform(0, 2 * np.pi, size=n)
        
        current_radius = spectral_radius_from_phases(theta)
        
        # Gradient descent with adaptive step size
        step_size = 0.5
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            grad = soft_spectral_radius_grad(theta, temperature=0.1)
            
            # Update phases
            theta_new = theta - step_size * grad
            # Wrap to [0, 2*pi)
            theta_new = np.mod(theta_new, 2 * np.pi)
            
            new_radius = spectral_radius_from_phases(theta_new)
            
            if new_radius < current_radius:
                theta = theta_new
                current_radius = new_radius
                step_size = min(step_size * 1.05, 2.0)  # Increase step slightly
            else:
                step_size *= 0.9  # Decrease step
            
            # Check convergence
            if step_size < 0.01:
                break
        
        # Get signs from optimized phases
        signs = phase_to_signs(theta)
        radius = signs_to_radius(signs)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # Local search polish: flip individual edges and check if it improves
    if time.time() < deadline:
        improved = True
        max_polish_iters = 100
        for _ in range(max_polish_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(e_count):
                if time.time() > deadline:
                    break
                # Try flipping edge i
                old_sign = best_signs[i]
                best_signs[i] = -old_sign
                new_radius = signs_to_radius(best_signs)
                if new_radius < best_radius - 1e-10:
                    best_radius = new_radius
                    improved = True
                else:
                    best_signs[i] = old_sign
            if not improved:
                break
    
    return best_signs
