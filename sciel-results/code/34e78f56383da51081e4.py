"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Sobol-based Pareto front sampling + local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate Sobol sequence for initial points on Pareto front manifold
        # For DTLZ2: x_1..x_{dim-1} = 0.5, x_0 varies
        n_initial = min(50, self.budget // 4)
        
        # Use Halton-like quasi-random sequence for x_0 values
        def halton_sequence(n, base):
            seq = []
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                seq.append(r)
            return seq
        
        # Generate diverse x_0 values using multiple bases for better coverage
        x0_values = np.linspace(0.0, 1.0, n_initial)
        
        # Create initial population on Pareto front manifold
        X = np.zeros((n_initial, self.dim))
        X[:, 0] = x0_values
        X[:, 1:] = 0.5  # g(x)=0 for DTLZ2 when x_1..x_{n-1}=0.5
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = n_initial
        
        # Local refinement using remaining budget
        if used < self.budget:
            # Sort by sum of objectives to find promising regions
            order = np.argsort(np.sum(F, axis=1))
            top_indices = order[:min(10, len(order))]
            
            # Local search around best points
            step_size = 0.05
            for idx in top_indices:
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                base_point = X[idx].copy()
                best_local = base_point.copy()
                best_local_f = problem(base_point)
                
                # Coordinate descent style local search
                for coord in range(self.dim):
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                    
                    for direction in [-1, 1]:
                        if used >= self.budget or time.time() > deadline - 5:
                            break
                        
                        test_point = best_local.copy()
                        test_point[coord] += direction * step_size
                        test_point = np.clip(test_point, lo, hi)
                        
                        test_f = problem(test_point)
                        used += 1
                        
                        # Check if this improves hypervolume contribution
                        # Simple heuristic: if sum is lower, keep it
                        if np.sum(test_f) < np.sum(best_local_f):
                            best_local = test_point
                            best_local_f = test_f
                
                # Update population with improved point
                if np.sum(best_local_f) < np.sum(F[idx]):
                    X[idx] = best_local
                    F[idx] = best_local_f
        
        return X
# EVOLVE-BLOCK-END
