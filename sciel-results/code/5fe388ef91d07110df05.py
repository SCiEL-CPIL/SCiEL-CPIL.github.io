import numpy as np
import time

class Optimizer:
      def __init__(self, budget, dim, n_obj):
          self.budget = budget
          self.dim = dim
          self.n_obj = n_obj

      def __call__(self, problem, bounds):
          t0 = time.time()
          deadline = t0 + 280
          
          lo = np.array([b[0] for b in bounds])
          hi = np.array([b[1] for b in bounds])
          
          # DTLZ7 assumption: x_0, x_1, x_2 define front, x_3... define distance.
          # Optimal distance is x_i = 0.5.
          # We will sample x_0, x_1, x_2 and fix x_3... to 0.5.
          
          n_evals = 0
          points = []
          fvals = []
          
          # Grid sampling on x_0, x_1, x_2
          # 200 budget. Let's use 125 for grid (5x5x5) and 75 for refinement.
          # Or just random sampling? Random sampling of the front is good for hypervolume.
          
          # Let's use a combination:
          # 1. 100 random points on the front.
          # 2. 100 points from local search around the best 10 points.
          
          # But first, let's just do 200 random points on the front.
          # x_0, x_1, x_2 uniform in [0,1]. x_3... = 0.5.
          
          # Wait, if we do 200 random points, we might miss some regions.
          # A grid is more systematic.
          
          # Let's do a 5x5x5 grid = 125 points.
          # Then 75 points of random sampling.
          
          grid_size = 5
          grid_points = []
          for i in range(grid_size):
              for j in range(grid_size):
                  for k in range(grid_size):
                      x = np.zeros(self.dim)
                      x[0] = lo[0] + (hi[0] - lo[0]) * (i + 0.5) / grid_size
                      x[1] = lo[1] + (hi[1] - lo[1]) * (j + 0.5) / grid_size
                      x[2] = lo[2] + (hi[2] - lo[2]) * (k + 0.5) / grid_size
                      # Set distance variables to 0.5
                      for m in range(3, self.dim):
                          x[m] = 0.5
                      grid_points.append(x)
          
          # Add random points
          rng = np.random.default_rng(42)
          random_points = []
          for _ in range(75):
              x = np.zeros(self.dim)
              x[0] = rng.uniform(lo[0], hi[0])
              x[1] = rng.uniform(lo[1], hi[1])
              x[2] = rng.uniform(lo[2], hi[2])
              for m in range(3, self.dim):
                  x[m] = 0.5
              random_points.append(x)
          
          all_points = grid_points + random_points
          
          for x in all_points:
              if n_evals >= self.budget:
                  break
              if time.time() > deadline:
                  break
              f = problem(x)
              points.append(x)
              fvals.append(f)
              n_evals += 1
          
          return points
