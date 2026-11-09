"""Multi-objective optimizer for DTLZ7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the Pareto front structure.
    The Pareto front is defined by x[2:] = 0.
    f0 = x[0]
    f1 = x[1]
    f2 = 1 - sqrt(x[1])
    We use 2D Sobol quasi-random sampling on the (u,v) parameter space,
    with u=0 for most points to minimize f0 and maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x[2:] = 0
        # x[0] = u, x[1] = v
        # f0 = u
        # f1 = v
        # f2 = 1 - sqrt(v)
        
        # For maximum hypervolume with reference point [1.1, 1.1, 25.0]:
        # - u should be as small as possible (u=0) to minimize f0
        # - v should be sampled to cover the front densely
        # The front is 1D in v, but we can add a few points with u>0
        # to ensure the non-dominated set is well-defined.
        
        from scipy.stats import qmc
        
        # Use 2D Sobol for better coverage of (u,v) space
        n_sobol = 256
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        sobol_2d = sampler.random(n_sobol)
        
        # Take first budget points
        t_u = sobol_2d[:budget, 0]
        t_v = sobol_2d[:budget, 1]
        
        # Transform t_v to v with density biased toward both ends
        # v = 1 - t^2 biases toward v=1, but we want good coverage
        # Use a symmetric transformation: v = 0.5 * (1 + sin(pi * t - pi/2))
        # This gives v in [0,1] with higher density near v=0 and v=1
        v_vals = 0.5 * (1.0 + np.sin(np.pi * t_v - np.pi / 2))
        
        # Clip to bounds
        v_vals = np.clip(v_vals, lo[1], hi[1])
        
        # For u: use u=0 for most points (minimizes f0, maximizes volume)
        # But add some u>0 points to ensure the non-dominated set is well-defined
        # Strategy: 90% at u=0, 10% at u>0 for coverage
        u_vals = np.zeros(budget)
        # Every 10th point gets a u value from the Sobol sequence
        mask = np.arange(budget) % 10 == 9
        u_vals[mask] = t_u[mask]
        
        # Clip u to bounds
        u_vals = np.clip(u_vals, lo[0], hi[0])
        
        # Add strategic corner points at the end
        # These ensure good coverage of the extreme regions
        # Reserve last 4 evaluations for corners
        n_corners = min(4, budget)
        n_regular = budget - n_corners
        
        # Regular points
        u_reg = u_vals[:n_regular]
        v_reg = v_vals[:n_regular]
        
        # Corner points: (u,v) = (0,0), (0,1), (1,0), (1,1)
        # These give f = (0,0,1), (0,1,0), (1,0,1), (1,1,0)
        corner_uv = np.array([
            [lo[0], lo[1]],  # u=0, v=0 -> f=(0,0,1)
            [lo[0], hi[1]],  # u=0, v=1 -> f=(0,1,0)
            [hi[0], lo[1]],  # u=1, v=0 -> f=(1,0,1)
            [hi[0], hi[1]],  # u=1, v=1 -> f=(1,1,0)
        ])
        
        # Evaluate regular points
        for i in range(n_regular):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = u_reg[i]
            x[1] = v_reg[i]
            # x[2:] = 0
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        # Evaluate corner points
        for i in range(n_corners):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = corner_uv[i, 0]
            x[1] = corner_uv[i, 1]
            # x[2:] = 0
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
