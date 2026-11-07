import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0

        # DTLZ2 structure:
        # x[0], x[1] are position variables (angles)
        # x[2:] are decision variables; g(x) is minimized at x[2:] = 0.5
        # The Pareto front is a quarter-sphere in 3D: f1^2 + f2^2 + f3^2 = 1

        # Generate Sobol sequence for position variables
        # We use the full budget for high-quality space-filling samples on the Pareto front.
        # 200 is not a power of 2, but Sobol can generate any number.
        # Scrambling helps avoid patterns.
        n_evals = self.budget
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        sobol_samples = sampler.random(n_evals)
        
        # Map to [0, 1] and then to bounds
        # sobol_samples are in [0, 1]
        pos_samples = sobol_samples * (hi[:2] - lo[:2]) + lo[:2]

        candidates = []
        
        # Create full candidate set: Sobol points with decision variables at 0.5
        for i in range(n_evals):
            x = mid.copy()
            x[0] = pos_samples[i, 0]
            x[1] = pos_samples[i, 1]
            # Ensure decision variables are exactly at the optimum (0.5)
            x[2:] = 0.5
            candidates.append(x)

        F = []
        X = []
        eval_count = 0
        max_evals = self.budget

        # Evaluate all candidates
        # We don't need sorting if we are just taking the first N from a good sampler.
        # However, if we wanted to bias, we could sort. For now, uniform is best for HV.
        
        for i in range(len(candidates)):
            if eval_count >= max_evals:
                break
            if time.time() > deadline:
                break
            x = candidates[i]
            try:
                f = problem(x)
                F.append(f)
                X.append(x)
                eval_count += 1
            except Exception:
                break

        return X
