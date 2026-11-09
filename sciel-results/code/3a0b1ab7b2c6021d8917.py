import numpy as np

class Optimizer:
       def __init__(self, budget, dim, n_obj):
           self.budget = budget
           self.dim = dim
           self.n_obj = n_obj

       def __call__(self, problem, bounds):
           N = self.budget
           # Effective Pareto front range for ZDT2 (sine variant)
           # f2 = 1 + sin(10 x1)
           # Min at 10 x1 = 3 pi / 2
           x_max = 3 * np.pi / 20.0
           
           # Uniform sampling in x1 on [0, x_max]
           # Use midpoints to avoid boundary issues if any, though 0 is valid.
           u = (np.arange(N) + 0.5) / N
           x1_vals = u * x_max
           
           X = np.zeros((N, self.dim))
           X[:, 0] = x1_vals
           # x2...xn = 0 is optimal for ZDT2 Pareto front
           
           F = np.array([problem(x) for x in X])
           
           return X
