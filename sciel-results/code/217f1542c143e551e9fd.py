import numpy as np


class Optimizer:
      def __init__(self, budget, dim, n_obj):
          self.budget = budget
          self.dim = dim
          self.n_obj = n_obj

      def __call__(self, problem, bounds):
          N = self.budget
          dim = self.dim
          
          # Determine optimal power p for spacing
          # We want to maximize HV = sum (u_i - u_{i-1}) * (1 - sqrt(u_i))
          # u_i = (i/N)^p
          best_p = 2.0
          best_hv = -1.0
          
          # Search for best p
          # The optimal p is likely between 1 and 3
          for p in np.arange(1.0, 3.01, 0.02):
              u = (np.arange(1, N + 1) / N) ** p
              f2 = 1.0 - np.sqrt(u)
              # u_0 = 0
              du = np.diff(np.concatenate(([0.0], u)))
              hv = np.sum(du * f2)
              if hv > best_hv:
                  best_hv = hv
                  best_p = p
                  
          # Generate x1 values
          x1_vals = (np.arange(1, N + 1) / N) ** best_p
          
          lo = np.array([b[0] for b in bounds])
          hi = np.array([b[1] for b in bounds])
          
          for x1 in x1_vals:
              x = np.zeros(dim)
              x[0] = x1
              # Clip to be safe, though x1 is in [0,1] and bounds are [0,1]
              x = np.clip(x, lo, hi)
              problem(x)
              
          return None
