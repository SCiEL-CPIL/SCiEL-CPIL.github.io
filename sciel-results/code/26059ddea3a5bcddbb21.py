import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
       start_time = time.time()
       deadline = start_time + 140  # 10s margin
   
       def score(points):
           # points: (8, 2) array
           d = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
           # Upper triangle indices
           iu = np.triu_indices(8, k=1)
           dists = d[iu]
           D = np.max(dists)
           if D == 0:
               return -np.inf
           # Objective: sum log(dists) - 28 log(D)
           # We want to maximize this, so minimize negative
           return -(np.sum(np.log(dists)) - 28 * np.log(D))
       
       # Initial population for DE
       # Use a regular octagon as one of the initial points?
       # DE creates its own population.
       # Bounds: |x|, |y| < 10^6. But scaling is invariant, so we can bound to [-2, 2] or similar.
       # If we bound too tight, we might miss the optimal shape if it's stretched?
       # No, scaling is invariant. We can fix the scale by bounding the diameter?
       # The objective function handles the scaling.
       # However, DE needs bounds.
       # Let's use bounds = [(-10, 10)] * 16.
       
       bounds = [(-10, 10)] * 16
       
       # Run DE
       # maxiter and popsize control the effort.
       # We need to check time.
       # Let's run DE for a fixed number of iterations, then check time.
       
       best_result = None
       best_score = -np.inf
       
       # Regular octagon score
       octagon = np.array([[np.cos(2*np.pi*k/8), np.sin(2*np.pi*k/8)] for k in range(8)])
       # Scale to diameter 2? The score function handles it.
       # But DE might get stuck.
       
       # Run DE
       # To speed up, use a smaller population and fewer iterations, then refine.
       # popsize=20, maxiter=100 might be fast enough.
       
       try:
           de_res = differential_evolution(
               score, 
               bounds, 
               maxiter=200, 
               popsize=20, 
               tol=1e-6, 
               seed=42,
               polish=False  # We will polish manually
           )
           if time.time() > deadline - 20:
               # Not enough time for refinement, return DE result
               pts = de_res.x.reshape(8, 2)
               return pts.tolist()
           
           # Refine DE result
           res = minimize(score, de_res.x, method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-12})
           if res.fun < best_score:
               best_score = res.fun
               best_result = res.x.reshape(8, 2)
               
           # Also try refining a perturbed octagon
           # The octagon is a local max, so we need to perturb it significantly.
           # DE should have found better points.
           
           # Run a second DE with different seed or parameters if time permits?
           # Let's check time.
           if time.time() < deadline - 40:
               de_res2 = differential_evolution(
                   score, 
                   bounds, 
                   maxiter=200, 
                   popsize=20, 
                   tol=1e-6, 
                   seed=123,
                   polish=False
               )
               if time.time() < deadline - 20:
                   res2 = minimize(score, de_res2.x, method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-12})
                   if res2.fun < best_score:
                       best_score = res2.fun
                       best_result = res2.x.reshape(8, 2)
                       
       except Exception:
           pass
       
       if best_result is None:
           # Fallback to regular octagon
           return octagon.tolist()
       
       # Ensure coordinates are within limits
       # The problem states |coordinate| < 10^6.
       # Our bounds were [-10, 10], so we are safe.
       
       return best_result.tolist()
