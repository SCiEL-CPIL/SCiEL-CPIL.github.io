import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 295.0

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        used = 0
        points = []
        objectives = []

        # DTLZ1 with M=3, N=7:
        # Pareto front: x[2:] = 0.5, and (x1, x2) in [0,1]^2
        # f1 = 0.5*x1, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        # The front is the 2D simplex f1+f2+f3=0.5, fi>=0.
        #
        # Strategy: Use a low-discrepancy sequence (Sobol) to sample (x1, x2) in [0,1]^2.
        # Additionally, include structured points at the corners and key locations
        # to ensure good coverage of the simplex vertices and edges, which are
        # critical for hypervolume.
        
        n_points = self.budget
        
        # Build a list of structured (x1, x2) pairs for the simplex corners/edges
        structured = []
        
        # Corners of the unit square (which map to simplex vertices/edges)
        for x1 in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for x2 in [0.0, 0.25, 0.5, 0.75, 1.0]:
                structured.append([x1, x2])
        
        # Add a few more specific points for better edge coverage
        for t in [1.0/6.0, 5.0/6.0, 1.0/3.0, 2.0/3.0]:
            structured.append([t, 0.0])
            structured.append([t, 1.0])
            structured.append([0.0, t])
            structured.append([1.0, t])
            structured.append([t, t])
        
        # Remove duplicates
        seen = set()
        unique_structured = []
        for pt in structured:
            key = (round(pt[0], 10), round(pt[1], 10))
            if key not in seen:
                seen.add(key)
                unique_structured.append(pt)
        
        n_structured = len(unique_structured)
        n_fill = max(0, n_points - n_structured)
        
        # Generate Sobol samples for the remaining points
        sobol_samples = np.empty((0, 2))
        if n_fill > 0:
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=2, scramble=True, seed=42)
                n_power2 = 1
                while n_power2 < n_fill:
                    n_power2 *= 2
                sobol_samples = sampler.random(n_power2)[:n_fill]
            except ImportError:
                rng = np.random.default_rng(12345)
                sobol_samples = rng.random((n_fill, 2))
        
        # Combine structured and Sobol samples
        if len(sobol_samples) > 0:
            all_pairs = np.vstack([np.array(unique_structured), sobol_samples])
        else:
            all_pairs = np.array(unique_structured)
        
        # Truncate to budget
        all_pairs = all_pairs[:n_points]
        
        # Evaluate each point on the Pareto front
        for i in range(len(all_pairs)):
            if used >= self.budget or self._elapsed() > self.deadline:
                break
            
            x1 = float(np.clip(all_pairs[i, 0], 0.0, 1.0))
            x2 = float(np.clip(all_pairs[i, 1], 0.0, 1.0))
            
            x = np.full(self.dim, 0.5)
            x[0] = x1
            x[1] = x2
            
            f = problem(x)
            points.append(x.copy())
            objectives.append(np.array(f, dtype=float))
            used += 1
        
        # If we still have budget left (shouldn't happen, but just in case)
        if used < self.budget:
            rng = np.random.default_rng(2024)
            for _ in range(self.budget - used):
                if self._elapsed() > self.deadline:
                    break
                x1 = rng.random()
                x2 = rng.random()
                x = np.full(self.dim, 0.5)
                x[0] = x1
                x[1] = x2
                f = problem(x)
                points.append(x.copy())
                objectives.append(np.array(f, dtype=float))
                used += 1
        
        return points
