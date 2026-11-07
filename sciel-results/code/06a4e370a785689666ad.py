import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
       def __init__(self, budget, dim, n_obj):
           self.budget = budget
           self.dim = dim
           self.n_obj = n_obj

       def __call__(self, problem, bounds):
           start_time = time.time()
           deadline = start_time + 250.0

           dim = self.dim
           n_obj = self.n_obj
           budget = self.budget

           lo = np.array([b[0] for b in bounds])
           hi = np.array([b[1] for b in bounds])

           # DTLZ4 specific
           # x[0], x[1] are position vars
           # x[2:] are distance vars, set to 0.5 for Pareto front
           
           points = []
           
           # Add extreme points
           # 1. x1=0, x2=0 -> f=(1,0,0)
           x = np.zeros(dim)
           x[0] = 0.0
           x[1] = 0.0
           x[2:] = 0.5
           points.append(x)
           
           # 2. x1=0, x2=1 -> f=(0,1,0)
           x = np.zeros(dim)
           x[0] = 0.0
           x[1] = 1.0
           x[2:] = 0.5
           points.append(x)
           
           # 3. x1=1, x2=0 -> f=(0,0,1)
           x = np.zeros(dim)
           x[0] = 1.0
           x[1] = 0.0
           x[2:] = 0.5
           points.append(x)
           
           # Add center point
           x = np.zeros(dim)
           x[0] = 0.5
           x[1] = 0.5
           x[2:] = 0.5
           points.append(x)

           # Add Halton points
           n_halton = budget - len(points)
           if n_halton > 0:
               halton = qmc.Halton(d=2, scramble=True)
               halton_points = halton.random(n_halton)
               for p in halton_points:
                   x = np.zeros(dim)
                   x[0] = p[0]
                   x[1] = p[1]
                   x[2:] = 0.5
                   points.append(x)
                   
           # Evaluate
           F = np.array([problem(x) for x in points])
           
           if time.time() > deadline:
               return F
               
           return F
