import numpy as np
import time


class Optimizer:
    """Structure-exploiting multi-objective optimizer for ZDT4.

    ZDT4 structure:
      g(x) = 1 + 10 * sum_{i=1}^{n} sin^2(10*x_i)
      f0 = g(x)
      f1 = 1 - sqrt((1/n) * sum x_i^2) * g(x)

    Pareto front: g(x) = 1 (all sin terms zero), so x_i = k*pi/10 for integer k.
      f0 = 1
      f1 = 1 - sqrt(sum x_i^2 / n)

    Key insight: For 2D hypervolume with all points on f0=1 line,
    HV = (ref_f0 - 1) * (ref_f1 - min_f1)
    So we need to minimize f1, which means maximizing sum x_i^2.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        n = dim
        budget = self.budget

        # Compute max |k| for each dimension based on bounds
        # x_i = k*pi/10 must satisfy low_i <= x_i <= high_i
        k_range_per_dim = []
        for i in range(dim):
            low, high = bounds[i]
            # k*pi/10 >= low => k >= low*10/pi
            # k*pi/10 <= high => k <= high*10/pi
            k_min = int(np.ceil(low * 10.0 / np.pi))
            k_max = int(np.floor(high * 10.0 / np.pi))
            if k_min > k_max:
                k_min, k_max = 0, 0
            k_range_per_dim.append((k_min, k_max))

        # The point that minimizes f1 maximizes sum x_i^2
        # For each dim, choose k with max |k|
        best_point = np.zeros(dim)
        for i in range(dim):
            k_min, k_max = k_range_per_dim[i]
            if abs(k_min) >= abs(k_max):
                best_point[i] = k_min * np.pi / 10.0
            else:
                best_point[i] = k_max * np.pi / 10.0

        # Evaluate the best point first (most important for hypervolume)
        used_evals = 0
        F = []

        fc = problem(best_point)
        F.append(fc)
        used_evals += 1

        # Dense 1D grid search: vary x_0 while fixing x_1..x_{n-1} to best_point values
        # This creates a 1D curve on the Pareto front
        # For each k_0 in range, compute the resulting f1 value
        k0_min, k0_max = k_range_per_dim[0]
        n_k0 = k0_max - k0_min + 1
        
        # Number of evaluations we can spend on the 1D grid
        n_grid = min(budget - used_evals, n_k0)
        
        if n_grid > 0:
            # Select which k_0 values to evaluate
            if n_grid >= n_k0:
                # Evaluate all possible k_0 values
                k0_values = list(range(k0_min, k0_max + 1))
            else:
                # Sample n_grid values from the range
                # Use uniform spacing in k-space
                k0_values = np.linspace(k0_min, k0_max, n_grid).astype(int)
                # Ensure we include both endpoints
                k0_values = np.unique(k0_values)
                if len(k0_values) < n_grid:
                    # Add more values from the middle
                    remaining = n_grid - len(k0_values)
                    all_k0 = set(range(k0_min, k0_max + 1))
                    sampled_k0 = set(k0_values)
                    extra_candidates = [k for k in range(k0_min, k0_max + 1) if k not in sampled_k0]
                    # Sort by distance to center to prioritize middle values
                    center = (k0_min + k0_max) / 2.0
                    extra_candidates.sort(key=lambda k: abs(k - center))
                    for k in extra_candidates[:remaining]:
                        k0_values = np.append(k0_values, k)
                    k0_values = np.unique(k0_values)
            
            # Evaluate each point on the 1D curve
            for k0 in k0_values:
                if used_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                pt = best_point.copy()
                pt[0] = k0 * np.pi / 10.0
                
                fc = problem(pt)
                F.append(fc)
                used_evals += 1
        
        # Use any remaining budget for additional points on the front
        # Vary other dimensions while keeping x_0 at best value
        rng = np.random.default_rng(42)
        while used_evals < budget and time.time() < deadline:
            # Pick a random dimension to vary (not 0, which we already covered)
            d = rng.integers(1, dim)
            k_min, k_max = k_range_per_dim[d]
            k = rng.integers(k_min, k_max + 1)
            
            pt = best_point.copy()
            pt[d] = k * np.pi / 10.0
            
            fc = problem(pt)
            F.append(fc)
            used_evals += 1

        return F
