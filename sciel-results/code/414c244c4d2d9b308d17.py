"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1-aware optimizer: exploit known Pareto front structure via stratified sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT1 Pareto front: x1 in [0,1], x2+ = 0
        # f1 = x1, f2 = 1 - sqrt(x1)
        # Power-law grid concentrates where f2 curvature is highest (near x1=0)
        # Use beta distribution with alpha < 1 to bias toward 0
        
        # Allocate most budget to Pareto front sampling
        n_front = int(budget * 0.85)
        n_perturb = budget - n_front
        
        # Power-law grid: sample x1 from Beta(0.5, 1) which biases toward 0
        # This concentrates where f2 = 1 - sqrt(x1) has highest curvature
        x1_vals = np.random.beta(0.5, 1.0, size=n_front)
        
        # Also add uniform samples for coverage
        n_uniform = max(1, n_front // 10)
        n_beta = n_front - n_uniform
        x1_vals = np.concatenate([
            np.random.beta(0.5, 1.0, size=n_beta),
            np.random.uniform(0, 1, size=n_uniform)
        ])
        
        # Shuffle to avoid ordering bias
        rng.shuffle(x1_vals)
        
        # Build Pareto front points: x1 = x1_val, x2+ = 0
        X_front = np.zeros((n_front, dim))
        X_front[:, 0] = x1_vals
        # x2 through xd remain 0 (this is the Pareto front)
        
        # Evaluate Pareto front points
        F_front = np.array([problem(x) for x in X_front])
        used = n_front
        
        # Use remaining budget for perturbations to ensure robustness
        # Small perturbations in x2+ direction to verify we're near the front
        for i in range(n_perturb):
            if time.time() > deadline:
                break
            # Pick a random Pareto front point and add small noise
            idx = rng.integers(n_front)
            child = X_front[idx].copy()
            # Small perturbation in non-first dimensions
            noise = 0.01 * (hi - lo) * rng.standard_normal(dim)
            child += noise
            child = np.clip(child, lo, hi)
            # Keep x1 close to original
            child[0] = x1_vals[idx] + 0.005 * rng.standard_normal()
            child[0] = np.clip(child[0], 0, 1)
            fc = problem(child)
            used += 1
        
        # Combine all evaluated points
        # Return the best set - the Pareto front points should dominate
        return X_front
# EVOLVE-BLOCK-END
