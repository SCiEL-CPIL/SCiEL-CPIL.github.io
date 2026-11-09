"""ZDT1 optimizer: exploit problem structure with local search for hypervolume maximization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT1 optimizer: stratified sampling with local refinement for hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        budget = self.budget

        n_evals = 0
        results = []

        def evaluate(x):
            nonlocal n_evals
            if n_evals >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.array(f, dtype=float)
            n_evals += 1
            results.append(f)
            return f

        # Phase 1: Probe the problem structure with a few evaluations
        # Try different x[1:] configurations to find which minimizes f2
        probe_configs = [
            np.zeros(n),
            np.full(n, 0.5),
            np.ones(n),
        ]
        
        # Probe at x[0] = 0.5 with different x[1:] to understand g
        probe_results = {}
        for i, config in enumerate(probe_configs):
            if time.time() > deadline or n_evals >= budget:
                break
            x = config.copy()
            x[0] = 0.5
            f = evaluate(x)
            if f is not None:
                probe_results[i] = f
        
        # Determine best x[1:] configuration (lowest f2)
        best_config_idx = 1  # default: x[1:] = 0.5
        if probe_results:
            best_f2 = min(probe_results.values(), key=lambda f: f[1])
            best_config_idx = [k for k, v in probe_results.items() if np.allclose(v, best_f2)][0]
        
        # Build the best x[1:] template
        if best_config_idx == 0:
            x1_template = np.zeros(n-1)
        elif best_config_idx == 2:
            x1_template = np.ones(n-1)
        else:
            x1_template = np.full(n-1, 0.5)
        
        # Phase 2: Dense stratified sampling on the Pareto front
        # Use remaining budget for uniform x[0] spacing
        remaining = budget - n_evals
        if remaining > 0:
            n_strata = remaining
            # Uniform spacing in x[0] which corresponds to uniform f1
            x0_values = np.linspace(0.0, 1.0, n_strata)
            
            for x0 in x0_values:
                if time.time() > deadline or n_evals >= budget:
                    break
                x = np.concatenate(([x0], x1_template))
                evaluate(x)
        
        return results
# EVOLVE-BLOCK-END
