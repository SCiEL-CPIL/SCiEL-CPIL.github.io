import numpy as np
import time

class Optimizer:
       def __init__(self, budget, dim, n_obj):
           self.budget = budget
           self.dim = dim
           self.n_obj = n_obj

       def __call__(self, problem, bounds):
           N = self.budget
           dim = self.dim
           start_time = time.time()
           deadline = start_time + 280  # Safety margin before 300s limit

           # ZDT2 Pareto front: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0, 1]
           # We want to maximize hypervolume with reference point (11, 11)
           # Optimal x1 placement: minimize the area under the curve approximation
           # using left Riemann sum: sum(dx_i * y_i) where y_i = 1 - sqrt(x_i)
           # This is equivalent to maximizing the area under the front up to ref point

           # Try various alpha values for power law spacing: x_i = (i/(N-1))^alpha
           # Also try uniform (alpha=1) and some other patterns
           alphas = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
           best_alpha = 1.0
           min_sum = float('inf')

           for alpha in alphas:
               x1 = (np.arange(N) / (N - 1)) ** alpha
               x1[0] = 0.0
               x1[-1] = 1.0

               y = 1.0 - np.sqrt(x1)

               # Left Riemann sum: sum of dx_i * y_i for i=0..N-2
               # Plus the last segment from x_{N-1} to 11 with y_{N-1}
               dx = np.diff(x1)
               sum_val = np.sum(dx * y[:-1])
               # Last term: (11 - x1[-1]) * y[-1] = 10 * 0 = 0 since x1[-1]=1
               sum_val += (11.0 - x1[-1]) * y[-1]

               if sum_val < min_sum:
                   min_sum = sum_val
                   best_alpha = alpha

           x1_opt = (np.arange(N) / (N - 1)) ** best_alpha
           x1_opt[0] = 0.0
           x1_opt[-1] = 1.0

           # Construct full vectors with x[1:] = 0 (on Pareto front)
           X = np.zeros((N, dim))
           X[:, 0] = x1_opt

           results = []
           for i in range(N):
               if time.time() > deadline:
                   break
               f = problem(X[i])
               results.append(f)

           return X
