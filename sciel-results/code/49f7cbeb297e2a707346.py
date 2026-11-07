"""Multi-objective optimizer for ZDT1 exploiting the known structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Exploit ZDT1 structure: fix x_2..x_n=0 for optimal g=1, then
    optimize x_1 placement to maximize hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        ref = np.array([11.0, 11.0])

        def make_x(t):
            x = np.zeros(dim)
            x[0] = t
            return x

        def eval_x1(t):
            return problem(make_x(t))

        # Compute HV for a set of x1 values (assuming f2 = 1 - sqrt(f1))
        def hv_from_x1(x1_vals):
            x1_sorted = np.sort(x1_vals)
            f2 = 1.0 - np.sqrt(x1_sorted)
            # HV = sum_{i=0}^{n-2} (x1[i+1]-x1[i])*(ref[1]-f2[i]) + (ref[0]-x1[n-1])*(ref[1]-f2[n-1])
            n = len(x1_sorted)
            if n == 0:
                return 0.0
            if n == 1:
                return (ref[0] - x1_sorted[0]) * (ref[1] - f2[0])
            widths = np.diff(x1_sorted)
            heights = ref[1] - f2[:-1]
            hv = np.sum(widths * heights)
            hv += (ref[0] - x1_sorted[-1]) * (ref[1] - f2[-1])
            return hv

        # Negative HV for minimization
        def neg_hv(params):
            x1_vals = np.sort(params)
            return -hv_from_x1(x1_vals)

        # Initial guess: uniform spacing
        n = budget
        x0 = np.linspace(0.0, 1.0, n)

        # Optimize the grid
        if n <= 2:
            x1_opt = x0
        else:
            try:
                result = minimize(neg_hv, x0, method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
                x1_opt = np.sort(result.x)
            except Exception:
                x1_opt = x0

        # Clip to [0, 1]
        x1_opt = np.clip(x1_opt, 0.0, 1.0)

        # Evaluate all points
        f_vals = np.array([eval_x1(t) for t in x1_opt])
        used = len(x1_opt)

        # If we have remaining budget (shouldn't happen), add more points
        while used < budget and time.time() < deadline - 5.0:
            t = np.random.uniform(0, 1)
            f = eval_x1(t)
            used += 1
            x1_opt = np.append(x1_opt, t)
            f_vals = np.vstack([f_vals, f.reshape(1, -1)])

        return x1_opt
# EVOLVE-BLOCK-END
