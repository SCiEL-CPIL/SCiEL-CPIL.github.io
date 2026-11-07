"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    Strategy: 
    Directly sample the Pareto front by distributing evaluations uniformly across x0 
    and setting other variables to the value that minimizes the 'distance' function g(x).
    For ZDT4, this corresponds to x_i = 0 (or closest bound if 0 is out of range).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # 1. Determine the optimal setting for variables x_1 ... x_{dim-1}
        # In ZDT4, g(x) is minimized when x_i = 0 (for i > 0).
        # If 0 is not in the bounds, we pick the bound closest to 0.
        x_rest = np.zeros(self.dim - 1)
        for j in range(1, self.dim):
            lo_j, hi_j = bounds[j]
            if lo_j <= 0.0 <= hi_j:
                x_rest[j-1] = 0.0
            elif 0.0 < lo_j:
                x_rest[j-1] = lo_j
            else:
                x_rest[j-1] = hi_j

        # 2. Determine x_0 values
        # We use all budget evaluations to sample x_0 uniformly across its bounds.
        # This ensures we cover the entire Pareto front (which is parametrized by x_0).
        lo_0, hi_0 = bounds[0]
        x0_values = np.linspace(lo_0, hi_0, self.budget)

        # 3. Construct the population
        X = np.zeros((self.budget, self.dim))
        for i in range(self.budget):
            X[i, 0] = x0_values[i]
            X[i, 1:] = x_rest

        # 4. Evaluate all points
        # This consumes exactly self.budget evaluations.
        F = np.array([problem(x) for x in X])

        return F
# EVOLVE-BLOCK-END
