import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
              start_time = time.time()
              deadline = start_time + 290
              
              dim = self.dim
              budget = self.budget
              calls = [0]
              
              def wrapper(x):
                  if time.time() > deadline:
                      raise TimeoutError
                  if calls[0] >= budget:
                      raise StopIteration
                  calls[0] += 1
                  x_clipped = np.clip(x, -5, 5)
                  return func(x_clipped)
              
              # Initial evaluations
              # 1. Center
              x0 = np.zeros(dim)
              f0 = wrapper(x0)
              best_x = x0.copy()
              best_f = f0
              
              # 2. Probe axes
              # This helps if the function is shifted significantly
              for i in range(dim):
                  if calls[0] >= budget or time.time() > deadline: break
                  x_pos = np.zeros(dim)
                  x_pos[i] = 5
                  f_pos = wrapper(x_pos)
                  if f_pos < best_f:
                      best_f = f_pos
                      best_x = x_pos.copy()
                  
                  x_neg = np.zeros(dim)
                  x_neg[i] = -5
                  f_neg = wrapper(x_neg)
                  if f_neg < best_f:
                      best_f = f_neg
                      best_x = x_neg.copy()
              
              # Run Nelder-Mead from best_x
              # Options: maxiter, xatol, fatol
              # We don't have a direct "max calls" option, but we can use the wrapper to stop.
              # However, minimize might not handle StopIteration/Timeout gracefully.
              # It's safer to implement a simple ES or use a fixed number of iterations.
              
              # Let's use a simple (1+1)-ES but with better sigma adaptation?
              # Or just run Nelder-Mead with a small maxiter?
              
              # Actually, the current ES is probably fine, but maybe the initial point is the issue.
              # The current program uses `best_x` from the vertex probe.
              # If the min is at the center, `best_x` is the center.
              # If the min is at a vertex, `best_x` is the vertex.
              
              # Let's try CMA-ES? It's in scipy? No, cma package is separate.
              # SciPy has `dual_annealing`, `differential_evolution`.
              # `differential_evolution` is global but slow.
              
              # Let's stick to the ES but make it more robust.
              # The current ES uses `sigma = 0.5`.
              # Maybe start with a larger sigma if the function is flat?
              
              # Let's try Nelder-Mead.
              try:
                  res = minimize(wrapper, best_x, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
                  if res.fun < best_f:
                      best_f = res.fun
                      best_x = res.x
              except:
                  pass
              
              # If budget remains, run ES
              if calls[0] < budget and time.time() < deadline:
                  x = best_x.copy()
                  fx = best_f
                  sigma = 0.1
                  rng = np.random.default_rng(42)
                  while calls[0] < budget:
                      if time.time() > deadline: break
                      y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                      fy = wrapper(y)
                      if fy < fx:
                          x, fx = y, fy
                          sigma *= 1.1
                      else:
                          sigma *= 0.9
                      if sigma < 1e-8: break
                  if fx < best_f:
                      best_f = fx
                      best_x = x
              
              return best_x, best_f
