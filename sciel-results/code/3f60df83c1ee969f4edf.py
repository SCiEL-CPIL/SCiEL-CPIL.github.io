import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
      n = 16
      start_time = time.time()
      deadline = start_time + 590

      def get_score(points):
          dists = pdist(points, 'euclidean')
          d_min = np.min(dists)
          d_max = np.max(dists)
          if d_max < 1e-10: return 0.0
          return (d_min / d_max) ** 2

      def smooth_objective(params, beta=100.0):
          """
          Smooth approximation of -min(dist)/max(dist) using log-sum-exp.
          Minimizes this to maximize the ratio.
          """
          pts = params.reshape((n, 2))
          dists = pdist(pts, 'euclidean')
          
          if np.max(dists) < 1e-10:
              return 1.0
          
          # Soft-min: exp(-beta * min) ≈ sum(exp(-beta * d)) / m, but for minimization we want to push min up
          # Actually, to maximize min/max, we can minimize -log(min) + log(max)
          # Using log-sum-exp for smooth max: log(sum(exp(d))) ≈ max(d)
          # Using log-sum-exp for smooth min: -log(sum(exp(-d))) ≈ min(d)
          
          # Smooth max distance
          max_d = np.max(dists)
          log_sum_exp_max = max_d + np.log(np.sum(np.exp(dists - max_d)))
          
          # Smooth min distance
          min_d = np.min(dists)
          log_sum_exp_min = min_d + np.log(np.sum(np.exp(dists - min_d)))
          # For min, we want -log(sum(exp(-d)))
          neg_dists = -dists
          max_neg_d = np.max(neg_dists)
          log_sum_exp_neg = max_neg_d + np.log(np.sum(np.exp(neg_dists - max_neg_d)))
          smooth_min = -log_sum_exp_neg
          
          # Objective: minimize -smooth_min + smooth_max (equivalent to minimizing -min/max in log space)
          # Or simply minimize (smooth_max - smooth_min)
          return smooth_max - smooth_min

      def objective_fixed(params):
          # params: 28 values for points 2..15
          p0 = [0.0, 0.0]
          p1 = [1.0, 0.0]
          pts = np.array([p0, p1] + [list(params[i:i+2]) for i in range(0, 28, 2)])
          dists = pdist(pts, 'euclidean')
          d_min = np.min(dists)
          d_max = np.max(dists)
          if d_max < 1e-10: return 1.0 # Maximize ratio -> Minimize -ratio
          return -(d_min / d_max)

      best_points = None
      best_score = -1.0

      def update_best(pts):
          nonlocal best_points, best_score
          if pts is None or not np.all(np.isfinite(pts)): return
          s = get_score(pts)
          if s > best_score:
              best_score = s
              best_points = pts.copy()

      # 1. Global Search with Fixed Scale
      if time.time() < deadline - 120:
          bounds = [(-5, 5)] * 28
          try:
              res = differential_evolution(
                  objective_fixed, 
                  bounds=bounds, 
                  maxiter=100, 
                  popsize=15, 
                  tol=1e-5, 
                  seed=42,
                  workers=1
              )
              if res.success or hasattr(res, 'x'):
                  p0 = [0.0, 0.0]
                  p1 = [1.0, 0.0]
                  pts = np.array([p0, p1] + [list(res.x[i:i+2]) for i in range(0, 28, 2)])
                  update_best(pts)
          except Exception as e:
              print("DE failed:", e)

      # 2. Local Refinement with Smooth Objective (L-BFGS-B)
      if best_points is not None and time.time() < deadline - 30:
          try:
              res = minimize(smooth_objective, best_points.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass

      # 3. Additional local refinement with Powell on the non-smooth objective
      if best_points is not None and time.time() < deadline - 10:
          def obj_full(params):
              pts = params.reshape((n, 2))
              dists = pdist(pts, 'euclidean')
              d_min = np.min(dists)
              d_max = np.max(dists)
              if d_max < 1e-10: return 0.0
              return -(d_min / d_max)
          
          try:
              res = minimize(obj_full, best_points.flatten(), method='Powell', 
                           options={'maxiter': 300, 'ftol': 1e-12, 'xtol': 1e-12})
              if res.success and np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass

      # 4. Fallback: Grid/Hex if DE failed or is slow
      if time.time() < deadline - 20:
          # Grid
          x = np.arange(4)
          y = np.arange(4)
          xx, yy = np.meshgrid(x, y)
          grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
          
          try:
              res = minimize(smooth_objective, grid_pts.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10})
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass
          
          # Hex
          hex_pts = np.array([
              [0, 0], [1, 0], [2, 0], [3, 0],
              [0.5, 0.866], [1.5, 0.866], [2.5, 0.866], [3.5, 0.866],
              [0, 1.732], [1, 1.732], [2, 1.732], [3, 1.732],
              [0.5, 2.598], [1.5, 2.598], [2.5, 2.598], [3.5, 2.598]
          ])
          try:
              res = minimize(smooth_objective, hex_pts.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10})
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass

      if best_points is None:
          # Absolute fallback
          x = np.arange(4)
          y = np.arange(4)
          xx, yy = np.meshgrid(x, y)
          return np.column_stack((xx.ravel(), yy.ravel()))
          
      return best_points
