"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start weighted-sum hill climbing for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # For ZDT2: f1 = x[0], f2 = (1 + 10*n) * (1 - sqrt(f1/g(x)))
        # where n = dim-1, g(x) = 1 + 10(n-1) + sum_{i=2}^{n} (x_i - 0.5)^2 * (1 + 10*cos(2*pi*(x_i+10*i/n)))
        # Pareto front: x[0] in [0,1], x[1:] = 0.5
        # f1 = x[0], f2 = 1 + 10(n-1) + ... but on PF: f2 = (1+10n)*(1-sqrt(f1))

        # Strategy: sample x[0] values, set x[1:] = 0.5, then refine with local search
        # This exploits the known structure of ZDT2

        dim = self.dim
        budget = self.budget

        # Pre-compute the Pareto front analytically for ZDT2
        # f1 = x0, f2 = (1 + 10*(dim-1)) * (1 - sqrt(x0)) for x0 in [0,1], x[1:] = 0.5
        # But we need to actually evaluate to record points, and the problem might clip

        # Phase 1: Analytical Pareto front sampling (uses evaluations)
        # We know the PF: x[0] = t, x[1:] = 0.5 for t in [0,1]
        # f1 = t, f2 = (1 + 10*(dim-1)) * (1 - sqrt(t))
        # But we must call problem() to record these points

        # Use a portion of budget for analytical PF points
        # Spread points along the PF for good hypervolume coverage
        n_pf = min(budget // 2, 100)  # use half budget for PF sampling
        n_pf = max(n_pf, 10)

        # Log-spaced or uniform? For hypervolume with ref [11,11], 
        # we want good coverage of the front.
        # The front goes from (0, 1+10*(dim-1)) to (1, 0)
        # With dim=30: f2 ranges from 0 to 1+10*29 = 291... wait that's > 11
        # Actually for ZDT2 with dim=30: f2_max = (1+10*29)*(1-0) = 291
        # But reference point is [11, 11], so only points with f2 < 11 count
        # f2 < 11 means (1+290)*(1-sqrt(x0)) < 11 => 1-sqrt(x0) < 11/291 => sqrt(x0) > 1 - 11/291 = 280/291
        # x0 > (280/291)^2 ≈ 0.925
        # So only x0 in [~0.925, 1] gives f2 < 11!
        # And f1 = x0 must be < 11, which is always true since x0 <= 1

        # So the effective PF segment is x0 in [(1-11/291)^2, 1] = [0.925, 1]
        # f1 in [0.925, 1], f2 in [0, 11]

        # This is a very short segment! Let me compute more carefully
        # g(x) on PF (x[1:]=0.5): g = 1 + 10*(dim-2) + sum of 0 = 1 + 10*28 = 281
        # Wait, let me re-read ZDT2 definition:
        # f1 = x_1
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} (x_i - 0.5)^2 * (1 + 10*cos(2*pi*(x_i + 10*i/n)))
        # where n = dim = 30
        # On PF: x_i = 0.5 for i >= 2, so the sum is 0
        # g = 1 + 10*29 = 291
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(f1/g)) = 291 * (1 - sqrt(x0/291))
        # Wait, I had the formula wrong. Let me reconsider.
        # Standard ZDT2: f2 = (1 + 10(n-1)) * (1 - sqrt(f1/g))
        # On PF: f2 = 291 * (1 - sqrt(x0/291))
        # f2 < 11: 291*(1-sqrt(x0/291)) < 11 => 1-sqrt(x0/291) < 11/291
        # sqrt(x0/291) > 1 - 11/291 = 280/291
        # x0/291 > (280/291)^2
        # x0 > 291 * (280/291)^2 = 280^2/291 = 78400/291 ≈ 269.4... 
        # That can't be right since x0 <= 1.
        
        # Let me recompute: x0 > 291*(280/291)^2 = 280^2/291
        # 280^2 = 78400, 78400/291 ≈ 269.4
        # So x0 > 269.4 which is impossible since x0 in [0,1]
        # This means NO points on the PF have f2 < 11?
        
        # Wait, that means the entire PF has f2 >= 11?
        # At x0=1: f2 = 291*(1-sqrt(1/291)) = 291*(1-0.0586) = 291*0.9414 ≈ 273.9
        # At x0=0: f2 = 291*(1-0) = 291
        # So f2 ranges from 273.9 to 291, all > 11!
        
        # This means with reference point [11,11], NO points on the true PF count!
        # The hypervolume would be 0 for the true PF...
        
        # Wait, I must be misunderstanding the ZDT2 formulation or the reference point.
        # Let me re-read the task: "reference point [11.0, 11.0]"
        # And "Only points strictly better than the reference point in every objective count"
        # So we need f1 < 11 AND f2 < 11.
        
        # For standard ZDT2 with dim=30, f1 = x0 in [0,1] < 11 always.
        # f2 = 291*(1-sqrt(x0/291)) which is always > 273 > 11.
        
        # This is strange. Perhaps the ZDT2 used here has a different scaling?
        # Or perhaps I'm wrong about the standard formulation.
        
        # Let me just use the empirical approach: multi-start local search
        # and let the evaluator's problem() define the actual objectives.
        
        # Multi-start weighted sum hill climbing
        # Use all budget for local search from multiple starts
        
        results = []
        evals_used = 0
        
        # Number of restarts
        n_restarts = min(20, budget // 5)
        evals_per_restart = budget // n_restarts
        evals_per_restart = max(evals_per_restart, 3)
        
        # Generate diverse starting points
        starts = []
        for i in range(n_restarts):
            x = lo + (hi - lo) * rng.random(dim)
            starts.append(x)
        
        for x_start in starts:
            if evals_used >= budget:
                break
            if time.time() > deadline - 5:
                break
                
            x = x_start.copy()
            f = np.array(problem(x), dtype=float)
            evals_used += 1
            results.append(f.copy())
            
            # Local search: try small perturbations
            step_size = 0.1
            for step in range(evals_per_restart):
                if evals_used >= budget:
                    break
                if time.time() > deadline - 2:
                    break
                    
                # Try random perturbation
                x_trial = x + step_size * (hi - lo) * rng.standard_normal(dim)
                x_trial = np.clip(x_trial, lo, hi)
                f_trial = np.array(problem(x_trial), dtype=float)
                evals_used += 1
                results.append(f_trial.copy())
                
                # Accept if not dominated by current (simple improvement)
                # Use weighted sum with random weights
                w = rng.dirichlet(np.ones(self.n_obj))
                if np.dot(w, f_trial) < np.dot(w, f):
                    x = x_trial.copy()
                    f = f_trial.copy()
                    step_size *= 0.95  # shrink step
                else:
                    step_size *= 0.9  # shrink on rejection
                
                if step_size < 1e-6:
                    break
        
        return np.array(results)
# EVOLVE-BLOCK-END
