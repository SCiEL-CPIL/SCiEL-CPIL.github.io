import numpy as np
import itertools
from scipy.optimize import minimize
from scipy.special import logsumexp
import time

N = 16
D = 2

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = float('inf')
    best_pts = None
    
    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    # Center
    pts_hex = np.array([[0.0, 0.0]])
    # Ring 1
    for k in range(6):
        angle = k * np.pi / 3
        pts_hex = np.vstack([pts_hex, [np.cos(angle), np.sin(angle)]])
    # Ring 2 (12 points, pick 9)
    # Angles: k * pi / 6
    # Let's pick the first 9
    for k in range(9):
        angle = k * np.pi / 6
        # Distance from origin is 2?
        # In hex lattice, neighbors of neighbors.
        # (1,0) neighbor is (2,0) dist 1 from (1,0).
        # So ring 2 is at distance 2 from center.
        pts_hex = np.vstack([pts_hex, [2*np.cos(angle), 2*np.sin(angle)]])
    inits.append(pts_hex)
    
    # 2. Perturbed Hex
    for _ in range(3):
        pts_pert = pts_hex + np.random.normal(0, 0.1, (16, 2))
        inits.append(pts_pert)
        
    # 3. Circle
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts_circle)
    
    # 4. Random
    for _ in range(3):
        pts_rand = np.random.uniform(-1, 1, (N, 2))
        inits.append(pts_rand)

    beta = 20.0 # Sharpness for softmax/min
    
    def objective(params):
        pts = params.reshape(N, 2)
        # Center of mass
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Upper triangle indices
        iu = np.triu_indices(N, 1)
        d_flat = dist[iu]
        
        # Avoid zero distance
        d_flat = np.maximum(d_flat, 1e-10)
        
        max_d = logsumexp(beta * d_flat) / beta
        min_d = -logsumexp(-beta * d_flat) / beta
        
        if min_d < 1e-10:
            return 1e10
        
        ratio = max_d / min_d
        return ratio**2

    def gradient(params):
        pts = params.reshape(N, 2)
        pts = pts - np.mean(pts, axis=0)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, 1)
        d_flat = dist[iu]
        d_flat = np.maximum(d_flat, 1e-10)
        
        # Gradient of logsumexp(beta * d) / beta
        # d/dbeta? No, wrt params.
        # f = 1/beta * log(sum(exp(beta * d)))
        # df/dp = 1/beta * (sum(exp(beta*d) * beta * dd/dp) / sum(exp(beta*d)))
        # = sum(exp(beta*d) * dd/dp) / sum(exp(beta*d))
        
        exp_max = np.exp(beta * d_flat - logsumexp(beta * d_flat)) # Normalized weights
        # Weights for max
        w_max = exp_max / np.sum(exp_max)
        
        exp_min = np.exp(-beta * d_flat - logsumexp(-beta * d_flat))
        # Weights for min
        # g = -1/beta * log(sum(exp(-beta * d)))
        # dg/dp = -1/beta * (sum(exp(-beta*d) * (-beta) * dd/dp) / sum(exp(-beta*d)))
        # = sum(exp(-beta*d) * dd/dp) / sum(exp(-beta*d))
        w_min = exp_min / np.sum(exp_min)
        
        # d_ratio = d_max - d_min (in log space? No, we are optimizing ratio^2)
        # Let's just use numerical gradient to be safe and simple.
        # Analytical gradient of ratio^2 is complex.
        return None

    # Use numerical gradient
    for i, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        res = minimize(objective, init.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_pts = res.x.reshape(N, 2)
            
            # Center and scale
            best_pts = best_pts - np.mean(best_pts, axis=0)
            
        # Check time
        if time.time() > deadline - 10.0:
            break

    if best_pts is None:
        # Fallback to hex
        best_pts = inits[0]
        
    # Final polish?
    # Maybe run one more optimization with higher beta?
    if time.time() < deadline - 5.0:
         beta_final = 50.0
         # Redefine objective with beta_final
         def objective_final(params):
             pts = params.reshape(N, 2)
             pts = pts - np.mean(pts, axis=0)
             diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
             dist = np.linalg.norm(diff, axis=2)
             iu = np.triu_indices(N, 1)
             d_flat = dist[iu]
             d_flat = np.maximum(d_flat, 1e-10)
             max_d = logsumexp(beta_final * d_flat) / beta_final
             min_d = -logsumexp(-beta_final * d_flat) / beta_final
             if min_d < 1e-10: return 1e10
             return (max_d / min_d)**2
             
         res_final = minimize(objective_final, best_pts.flatten(), method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
         if res_final.fun < best_score:
             best_pts = res_final.x.reshape(N, 2)
             best_pts = best_pts - np.mean(best_pts, axis=0)

    return best_pts
