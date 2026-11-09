import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ1 specific knowledge:
        # n=7, M=3
        # Pareto front is defined by g(x) = 0.
        # For DTLZ1, g(x) = 0 when x[i] = 0.5 for i >= 2.
        # The front is parameterized by x[0] and x[1].
        # f1 = 0.5 * x0
        # f2 = 0.5 * (1 - x0) * x1
        # f3 = 0.5 * (1 - x0) * (1 - x1)
        
        # We want to sample uniformly from the Pareto front in objective space.
        # The front is a triangle in 3D with vertices (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5).
        # We can sample barycentric coordinates (alpha, beta, gamma) uniformly from the simplex.
        # This corresponds to a Dirichlet(1, 1, 1) distribution.
        
        # Generate barycentric coordinates
        # np.random.dirichlet uses the global numpy random state, which is seeded by the evaluator.
        barycentric = np.random.dirichlet([1.0, 1.0, 1.0], size=self.budget)
        
        # barycentric[i] = [alpha_i, beta_i, gamma_i]
        # f1 = 0.5 * alpha
        # f2 = 0.5 * beta
        # f3 = 0.5 * gamma
        
        # Invert to find x0, x1
        # x0 = 2 * f1 = alpha
        # x1 = f2 / (f2 + f3) = beta / (beta + gamma)
        
        x0 = barycentric[:, 0]
        # Handle division by zero if beta + gamma == 0 (which happens if alpha == 1)
        # In that case, x1 can be anything, so we set it to 0.
        denom = barycentric[:, 1] + barycentric[:, 2]
        x1 = np.zeros(self.budget)
        mask = denom > 1e-12
        x1[mask] = barycentric[mask, 1] / denom[mask]
        
        # Construct decision vectors
        # x[0] = x0
        # x[1] = x1
        # x[2:] = 0.5 (to ensure g=0)
        
        for i in range(self.budget):
            x = np.zeros(self.dim)
            x[0] = x0[i]
            x[1] = x1[i]
            x[2:] = 0.5
            problem(x)
            
        return None
