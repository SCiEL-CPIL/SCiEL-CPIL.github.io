import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time
import warnings
warnings.filterwarnings('ignore')

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

      def normalize_to_unit_max(pts):
          """Scale points so that max pairwise distance is exactly 1.0"""
          dists = pdist(pts, 'euclidean')
          d_max = np.max(dists)
          if d_max < 1e-10:
              return pts.copy()
          return pts / d_max

      def soft_min_distance(pts, beta=50.0):
          """
          Smooth approximation of min pairwise distance using log-sum-exp.
          Returns -log(sum(exp(-d/beta)))/beta which approximates min(d) as beta->inf
          """
          dists = pdist(pts, 'euclidean')
          if np.max(dists) < 1e-10:
              return 0.0
          # Soft min: -beta * log(mean(exp(-d/beta)))
          # For stability: let m = min(dists), then -beta*log(mean(exp(-(d-m)/beta))) + m
          m = np.min(dists)
          exp_terms = np.exp(-(dists - m) / beta)
          return m - beta * np.log(np.mean(exp_terms))

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
      
      def smooth_objective_scaled(params, beta=50.0):
          """
          Smooth objective: minimize -soft_min_distance after scaling to unit max.
          This removes scale invariance and provides smooth gradients.
          """
          pts = params.reshape((n, 2))
          # Normalize so max distance = 1
          pts_norm = normalize_to_unit_max(pts)
          # Return negative soft-min distance (we want to maximize min distance)
          return -soft_min_distance(pts_norm, beta=beta)

      def exact_objective_scaled(params):
          """
          Exact objective: minimize -min_distance after scaling to unit max.
          """
          pts = params.reshape((n, 2))
          pts_norm = normalize_to_unit_max(pts)
          dists = pdist(pts_norm, 'euclidean')
          d_min = np.min(dists)
          return -d_min

      best_points = None
      best_score = -1.0

      def update_best(pts):
          nonlocal best_points, best_score
          if pts is None or not np.all(np.isfinite(pts)): return
          s = get_score(pts)
          if s > best_score:
              best_score = s
              best_points = pts.copy()

      # Phase 1: Global search with fixed scale
      if time.time() < deadline - 120:
          bounds = [(-5, 5)] * 28
          try:
              res = differential_evolution(
                  objective_fixed, 
                  bounds=bounds, 
                  maxiter=200, 
                  popsize=15, 
                  tol=1e-6, 
                  seed=42,
                  workers=1,
                  polish=True,
                  init='sobol'
              )
              if res.success or hasattr(res, 'x'):
                  p0 = [0.0, 0.0]
                  p1 = [1.0, 0.0]
                  pts = np.array([p0, p1] + [list(res.x[i:i+2]) for i in range(0, 28, 2)])
                  update_best(pts)
          except Exception as e:
              print("DE failed:", e)

      # Phase 2: Gradient-based refinement using smooth scaled objective
      if best_points is not None and time.time() < deadline - 60:
          # Try multiple beta values for the soft-min approximation
          for beta in [30.0, 50.0, 100.0]:
              if time.time() > deadline - 60:
                  break
              try:
                  res_smooth = minimize(
                      lambda p: smooth_objective_scaled(p, beta=beta), 
                      best_points.flatten(), 
                      method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                  )
                  if np.all(np.isfinite(res_smooth.x)):
                      pts_smooth = res_smooth.x.reshape((n, 2))
                      s_smooth = get_score(pts_smooth)
                      if s_smooth > best_score:
                          best_score = s_smooth
                          best_points = pts_smooth.copy()
              except Exception:
                  pass

          # Phase 2b: Exact objective refinement with Powell
          if best_points is not None and time.time() < deadline - 30:
              try:
                  res = minimize(
                      exact_objective_scaled, 
                      best_points.flatten(), 
                      method='Powell', 
                      options={'maxiter': 2000, 'ftol': 1e-15, 'xtol': 1e-15}
                  )
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass

      # Phase 3: Multiple diverse initializations
      if time.time() < deadline - 90:
          def obj_full2(params):
              pts = params.reshape((n, 2))
              dists = pdist(pts, 'euclidean')
              d_min = np.min(dists)
              d_max = np.max(dists)
              if d_max < 1e-10: return 0.0
              return -(d_min / d_max)
          
          # Grid initialization
          x = np.arange(4)
          y = np.arange(4)
          xx, yy = np.meshgrid(x, y)
          grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
          try:
              res = minimize(obj_full2, grid_pts.flatten(), method='Powell', options={'maxiter': 500, 'ftol': 1e-14})
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass

          # Hexagonal lattice
          hex_pts = np.array([
              [0, 0], [1, 0], [2, 0], [3, 0],
              [0.5, 0.866], [1.5, 0.866], [2.5, 0.866], [3.5, 0.866],
              [0, 1.732], [1, 1.732], [2, 1.732], [3, 1.732],
              [0.5, 2.598], [1.5, 2.598], [2.5, 2.598], [3.5, 2.598]
          ])
          try:
              res = minimize(obj_full2, hex_pts.flatten(), method='Powell', options={'maxiter': 500, 'ftol': 1e-14})
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass

          # Circle
          if time.time() < deadline - 60:
              angles = np.linspace(0, 2*np.pi, n, endpoint=False)
              circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
              try:
                  res = minimize(obj_full2, circle_pts.flatten(), method='Powell', options={'maxiter': 500, 'ftol': 1e-14})
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass

          # Fibonacci sphere projected to 2D (spiral)
          if time.time() < deadline - 30:
              t = np.linspace(0, 6*np.pi, n)
              r = np.linspace(0.1, 1.0, n)
              spiral_pts = np.column_stack([r * np.cos(t), r * np.sin(t)])
              try:
                  res = minimize(obj_full2, spiral_pts.flatten(), method='Powell', options={'maxiter': 500, 'ftol': 1e-14})
                  if np.all(np.isfinite(res.x)):
                      update_best(res.x.reshape((n, 2)))
              except Exception:
                  pass

          # Random initializations
          if time.time() < deadline - 30:
              for seed in [1, 2, 3]:
                  if time.time() > deadline - 30:
                      break
                  rng = np.random.RandomState(seed)
                  rand_pts = rng.uniform(-2, 2, size=(n, 2))
                  try:
                      res = minimize(obj_full2, rand_pts.flatten(), method='Powell', options={'maxiter': 300, 'ftol': 1e-14})
                      if np.all(np.isfinite(res.x)):
                          update_best(res.x.reshape((n, 2)))
                  except Exception:
                      pass

      # Phase 4: High-precision final polish
      if best_points is not None and time.time() < deadline - 10:
          def obj_full_highprec(params):
              pts = params.reshape((n, 2))
              dists = pdist(pts, 'euclidean')
              d_min = np.min(dists)
              d_max = np.max(dists)
              if d_max < 1e-10: return 0.0
              return -(d_min / d_max)
          
          # Nelder-Mead with tight tolerance
          try:
              res = minimize(
                  obj_full_highprec, 
                  best_points.flatten(), 
                  method='Nelder-Mead', 
                  options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
              )
              if np.all(np.isfinite(res.x)):
                  update_best(res.x.reshape((n, 2)))
          except Exception:
              pass

          # Final Powell with very tight tolerance
          if time.time() < deadline - 3:
              try:
                  res = minimize(
                      obj_full_highprec, 
                      best_points.flatten(), 
                      method='Powell', 
                      options={'maxiter': 5000, 'ftol': 1e-16, 'xtol': 1e-16}
                  )
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
