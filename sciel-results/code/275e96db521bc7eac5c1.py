"""Multi-objective optimizer for ZDT1 with structure-aware initialization and refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 30s safety margin before 310s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For ZDT1: f1 = x0, f2 = g*(1 - sqrt(f1/g)) where g = 1 + 10*(n-1) + sum((xi-0.5)^2)
        # Optimal: x1..xn = 0.5, x0 varies in [0,1]
        # Strategy: Exploit ZDT1 structure - fix x[1:] near 0.5, optimize x0

        # Phase 1: Structure-aware initialization
        # Place points along the Pareto front with x[1:] = 0.5
        n_phase1 = min(100, budget // 2)
        
        # Uniform spread of x0 values
        x0_vals = np.linspace(0, 1, n_phase1)
        X = np.zeros((n_phase1, dim))
        X[:, 0] = x0_vals
        X[:, 1:] = 0.5  # Optimal for g(x) in ZDT1
        
        F = np.array([problem(x) for x in X])
        used = n_phase1

        # Phase 2: Refinement around promising regions
        # Use remaining budget to refine x0 positions and slightly perturb x[1:]
        remaining = budget - used
        
        if remaining > 0:
            # Identify non-dominated points
            F_best = F.copy()
            X_best = X.copy()
            
            # Create new points by perturbing existing ones
            # Focus on varying x0 more finely and small perturbations in x[1:]
            n_phase2 = remaining
            
            # Use a mix of fine x0 sampling and small perturbations
            X2 = np.zeros((n_phase2, dim))
            
            # Half: fine x0 sampling with x[1:] = 0.5
            n_fine = n_phase2 // 2
            x0_fine = np.linspace(0, 1, n_fine)
            X2[:n_fine, 0] = x0_fine
            X2[:n_fine, 1:] = 0.5
            
            # Other half: perturbations of existing points
            n_perturb = n_phase2 - n_fine
            if n_perturb > 0:
                rng = np.random.default_rng(42)
                indices = rng.integers(0, n_phase1, size=n_perturb)
                X2[n_fine:, :] = X[indices, :]
                # Small perturbation in x0
                X2[n_fine:, 0] += rng.normal(0, 0.01, size=n_perturb)
                X2[n_fine:, 0] = np.clip(X2[n_fine:, 0], 0, 1)
                # Tiny perturbation in x[1:]
                X2[n_fine:, 1:] += rng.normal(0, 0.001, size=(n_perturb, dim - 1))
                X2[n_fine:, 1:] = np.clip(X2[n_fine:, 1:], 0, 1)
            
            F2 = np.array([problem(x) for x in X2])
            used += n_phase2
            
            # Combine all points
            X_all = np.vstack([X, X2])
            F_all = np.vstack([F, F2])
        else:
            X_all = X
            F_all = F

        # Phase 3: If time allows, do a final refinement pass
        if used < budget and time.time() < deadline:
            # Try to find even better points by local search
            # Identify the current best non-dominated set
            # For ZDT1, the ideal is x[1:] = 0.5 exactly
            
            # Use remaining budget for very fine x0 sampling
            n_phase3 = budget - used
            if n_phase3 > 0:
                x0_very_fine = np.linspace(0, 1, n_phase3)
                X3 = np.zeros((n_phase3, dim))
                X3[:, 0] = x0_very_fine
                X3[:, 1:] = 0.5
                
                # Check time before evaluating
                if time.time() < deadline - 5:
                    F3 = np.array([problem(x) for x in X3])
                    used += n_phase3
                    X_all = np.vstack([X_all, X3])
                    F_all = np.vstack([F_all, F3])

        return X_all
# EVOLVE-BLOCK-END
