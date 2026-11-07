import time
import numpy as np
from itertools import product as iproduct

def sign_edges(graph, budget):
    deadline = time.time() + 85.0
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    def get_edge_types():
        """Categorize edges into top cycle, bottom cycle, and matching"""
        top_cycle = []
        bottom_cycle = []
        matching = []
        
        for idx, (u, v) in enumerate(edges):
            if u < m and v < m:
                top_cycle.append(idx)
            elif u >= m and v >= m:
                bottom_cycle.append(idx)
            else:
                matching.append(idx)
        
        return top_cycle, bottom_cycle, matching
    
    top_cycle, bottom_cycle, matching = get_edge_types()
    
    def build_signed_matrix(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(signs):
        A = build_signed_matrix(signs)
        evals = np.linalg.eigvalsh(A)
        return np.max(np.abs(evals))
    
    def is_connected_lift(signs):
        """
        A 2-lift of a prism graph is connected iff the product of signs
        around any cycle that uses matching edges is -1.
        For prism graphs, check the product around the "ladder" cycles.
        """
        lift_adj = {v: set() for v in range(2*n)}
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                lift_adj[u].add(v)
                lift_adj[v].add(u)
                lift_adj[u + n].add(v + n)
                lift_adj[v + n].add(u + n)
            else:
                lift_adj[u].add(v + n)
                lift_adj[v + n].add(u)
                lift_adj[u + n].add(v)
                lift_adj[v].add(u + n)
        
        visited = set([0])
        queue = [0]
        while queue:
            curr = queue.pop(0)
            for neighbor in lift_adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == 2 * n
    
    def is_valid(signs):
        return is_connected_lift(signs)
    
    best_radius = float('inf')
    best_signs = None
    
    # For m <= 6, exhaustive search
    if m <= 6:
        num_signs = 3 * m
        for trial in range(1 << num_signs):
            if time.time() > deadline - 5:
                break
            signs = np.array([(1 if (trial >> i) & 1 else -1) for i in range(num_signs)])
            if not is_valid(signs):
                continue
            r = spectral_radius(signs)
            if r < best_radius:
                best_radius = r
                best_signs = signs.copy()
    
    # For larger m, use continuous phase-space optimization
    else:
        rng = np.random.default_rng(42)
        
        def random_valid_signing():
            while True:
                signs = rng.choice([-1, 1], size=num_edges)
                if is_valid(signs):
                    return signs
        
        # Initialize with random valid signing
        best_signs = random_valid_signing()
        best_radius = spectral_radius(best_signs)
        
        # Continuous phase-space optimization
        # Represent each sign as cos(theta) where theta in [0, pi]
        # Use gradient descent on the smooth surrogate: max |eigenvalue| of A(theta)
        # where A(theta) has entries cos(theta_i) for edge i
        
        def build_continuous_matrix(thetas):
            A = np.zeros((n, n))
            cos_vals = np.cos(thetas)
            for i, (u, v) in enumerate(edges):
                s = cos_vals[i]
                A[u, v] = s
                A[v, u] = s
            return A
        
        def continuous_spectral_radius(thetas):
            A = build_continuous_matrix(thetas)
            evals = np.linalg.eigvalsh(A)
            return np.max(np.abs(evals))
        
        def gradient_spectral_radius(thetas):
            """Compute gradient of spectral radius w.r.t. thetas using finite differences"""
            eps = 1e-5
            grad = np.zeros_like(thetas)
            f0 = continuous_spectral_radius(thetas)
            for i in range(len(thetas)):
                theta_plus = thetas.copy()
                theta_plus[i] += eps
                theta_minus = thetas.copy()
                theta_minus[i] -= eps
                fp = continuous_spectral_radius(theta_plus)
                fm = continuous_spectral_radius(theta_minus)
                grad[i] = (fp - fm) / (2 * eps)
            return grad
        
        # Multi-start gradient descent in phase space
        n_restarts = 20
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            if restart == 0:
                # Start from current best
                init_thetas = np.arccos(np.clip(best_signs.astype(float), -1, 1))
            else:
                # Random initialization
                init_thetas = rng.uniform(0, np.pi, size=num_edges)
            
            # Gradient descent with momentum
            thetas = init_thetas.copy()
            velocity = np.zeros_like(thetas)
            lr = 0.05
            momentum = 0.9
            max_iter = 500
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                grad = gradient_spectral_radius(thetas)
                velocity = momentum * velocity - lr * grad
                thetas = thetas + velocity
                thetas = np.clip(thetas, 0, np.pi)
            
            # Round to nearest sign
            final_signs = np.where(np.cos(thetas) >= 0, 1, -1)
            
            if is_valid(final_signs):
                r = spectral_radius(final_signs)
                if r < best_radius:
                    best_radius = r
                    best_signs = final_signs.copy()
        
        # Also try simulated annealing on the discrete space
        current_signs = best_signs.copy()
        current_radius = best_radius
        T = 1.0
        cooling = 0.9995
        min_T = 1e-6
        
        while time.time() < deadline - 5 and T > min_T:
            idx = rng.integers(0, num_edges)
            new_signs = current_signs.copy()
            new_signs[idx] *= -1
            
            if not is_valid(new_signs):
                T *= cooling
                continue
            
            new_radius = spectral_radius(new_signs)
            delta = new_radius - current_radius
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_signs = new_signs
                current_radius = new_radius
                
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = current_signs.copy()
            
            T *= cooling
    
    # Final polish: try flipping each edge individually
    if best_signs is not None and time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(num_edges):
                if time.time() > deadline - 2:
                    break
                new_signs = best_signs.copy()
                new_signs[idx] *= -1
                if not is_valid(new_signs):
                    continue
                r = spectral_radius(new_signs)
                if r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = new_signs
                    improved = True
    
    if best_signs is None:
        best_signs = random_valid_signing()
    
    return best_signs
# EVOLVE-BLOCK-END
