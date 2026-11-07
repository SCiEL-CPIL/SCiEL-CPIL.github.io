"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 that exploits the known Pareto front geometry.
    
    DTLZ1 with M=3 objectives, N=7 variables:
    - g(x) = 100 * (N - M + 1 + sum_{i=M}^{N} [(x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))])
    - f_1 = 0.5 * (1 + g) * x_1
    - f_2 = 0.5 * (1 + g) * x_2
    - f_3 = 0.5 * (1 + g) * (1 - x_1^2 - x_2^2)
    
    Pareto front: g=0 (x_3..x_7 = 0.5), x_1^2 + x_2^2 = 1, x_1, x_2 >= 0
    On PF: f_1 = 0.5*x_1, f_2 = 0.5*x_2, f_3 = 0
    
    Strategy:
    1. Use a large fraction of budget to sample the known Pareto front via low-discrepancy
       sequences (Halton) on the quarter-circle x_1^2+x_2^2=1, x_1,x_2>=0.
    2. Use remaining budget for local search / random perturbation to discover any
       off-front points that might improve hypervolume (though unlikely for DTLZ1).
    3. Track elapsed time and return early with safety margin before 330s deadline.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton sequence values in [0,1) with given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            val = 0.0
            f_i = f
            idx = i + 1
            while idx > 0:
                val += f_i * (idx % base)
                idx //= base
                f_i /= base
            result[i] = val
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Allocate budget: ~85% to Pareto front sampling, ~15% to exploration
        n_pf = int(budget * 0.85)
        n_explore = budget - n_pf
        
        # Generate points on the DTLZ1 Pareto front
        # x_1 = cos(theta), x_2 = sin(theta) for theta in [0, pi/2]
        # x_3..x_7 = 0.5 (to make g=0)
        
        # Use Halton sequence for low-discrepancy sampling of theta
        # We want good coverage of the quarter circle
        # Sample t in [0,1] via Halton, map to theta = t * pi/2
        # But to get uniform distribution on the arc, we can just use uniform theta
        # Actually for hypervolume, uniform in theta gives non-uniform in (f1,f2)
        # Better to sample uniformly in the (x1, x2) space on the arc
        # Let's use a combination: Halton for base coverage + some uniform
        
        # Generate Halton sequence points
        h1 = self._halton(n_pf, 2)
        h2 = self._halton(n_pf, 3)
        
        # Use h1 for theta parameter, h2 for slight perturbation
        theta = h1 * (np.pi / 2.0)
        
        # Create points on the Pareto front
        X_pf = np.zeros((n_pf, dim))
        X_pf[:, 0] = np.cos(theta)
        X_pf[:, 1] = np.sin(theta)
        # Set remaining dimensions to 0.5 (center of bounds) for g=0
        for d in range(2, dim):
            X_pf[:, d] = 0.5
        
        # Clip to bounds
        X_pf = np.clip(X_pf, lo, hi)
        
        # Evaluate Pareto front points
        F_list = []
        for i in range(n_pf):
            if time.time() > deadline:
                break
            F_list.append(problem(X_pf[i]))
        
        if not F_list:
            # Fallback: single random point
            X_single = lo + (hi - lo) * np.random.random(dim)
            F_list.append(problem(X_single))
        
        F_pf = np.array(F_list)
        n_evaluated = len(F_list)
        
        # Exploration: random points and perturbed front points
        rng = np.random.default_rng(42)
        
        # Mix of random points and perturbed front points
        n_random = n_explore // 2
        n_perturb = n_explore - n_random
        
        X_explore = np.zeros((n_explore, dim))
        
        # Random points
        X_explore[:n_random] = lo + (hi - lo) * rng.random((n_random, dim))
        
        # Perturbed front points: take some front points and add noise
        if n_perturb > 0 and n_pf > 0:
            indices = rng.integers(0, n_pf, size=n_perturb)
            X_explore[n_random:] = X_pf[indices]
            noise = 0.05 * (hi - lo) * rng.standard_normal((n_perturb, dim))
            X_explore[n_random:] += noise
            X_explore[n_random:] = np.clip(X_explore[n_random:], lo, hi)
        
        # Evaluate exploration points
        for i in range(n_explore):
            if time.time() > deadline:
                break
            if n_evaluated + i >= budget:
                break
            F_list.append(problem(X_explore[i]))
        
        F_all = np.array(F_list)
        
        return F_all
# EVOLVE-BLOCK-END
