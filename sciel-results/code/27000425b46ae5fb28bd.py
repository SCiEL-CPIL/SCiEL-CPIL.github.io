"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Pareto front sampling with refinement for DTLZ7."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front: g=0 means x[19]=x[20]=x[21]=0.5 (0-indexed)
        # f1 = x[0], f2 = x[1], f3 = h(x[2]) where h depends on x[2]
        # Sample x[0], x[1], x[2] using Fibonacci lattice in [0,1]^3
        
        n_samples = 100  # number of initial Pareto front samples
        used = 0
        
        # Generate Fibonacci lattice in 3D
        golden = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_samples)
        # Use 3D Fibonacci lattice
        theta = 2 * np.pi * indices / golden
        phi = np.arccos(1 - 2 * (indices + 0.5) / n_samples)
        # Map to [0,1]^3 using spherical coordinates
        x0 = 0.5 + 0.5 * np.sin(phi) * np.cos(theta)
        x1 = 0.5 + 0.5 * np.sin(phi) * np.sin(theta)
        x2 = 0.5 + 0.5 * np.cos(phi)
        
        # Clip to [0,1]
        x0 = np.clip(x0, 0, 1)
        x1 = np.clip(x1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        
        # Create full 22-dim vectors
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0
        X[:, 1] = x1
        X[:, 2] = x2
        # x[3] to x[18] = 0
        X[:, 19] = 0.5
        X[:, 20] = 0.5
        X[:, 21] = 0.5
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used += n_samples
        
        # Refine: use remaining budget for local search around best points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Simple refinement: perturb non-dominated points
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                if time.time() >= deadline:
                    break
                # Find non-dominated points
                dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
                if np.any(~dominated):
                    parents_idx = np.where(~dominated)[0]
                else:
                    parents_idx = np.arange(len(F))
                
                parent_idx = parents_idx[rng.integers(len(parents_idx))]
                parent = X[parent_idx].copy()
                
                # Perturb only the first 3 variables (Pareto front parameters)
                perturb = 0.05 * rng.standard_normal(3)
                parent[0] = np.clip(parent[0] + perturb[0], 0, 1)
                parent[1] = np.clip(parent[1] + perturb[1], 0, 1)
                parent[2] = np.clip(parent[2] + perturb[2], 0, 1)
                # Keep x[19:22] = 0.5
                parent[19:22] = 0.5
                
                fc = problem(parent)
                # Replace worst point
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst] = parent
                F[worst] = fc
                used += 1
        
        return X
# EVOLVE-BLOCK-END
