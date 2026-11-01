import numpy as np
import itertools
import time
from scipy.optimize import minimize

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit for safety
    
    best_points = None
    best_score = float('inf')
    
    # Generate initial candidates
    candidates = []
    
    # Candidate 1: 8 corners + 6 face centers of a cube
    corners = np.array(list(itertools.product([-0.5, 0.5], repeat=3)))
    face_centers = np.array([
        [0.5, 0, 0], [-0.5, 0, 0],
        [0, 0.5, 0], [0, -0.5, 0],
        [0, 0, 0.5], [0, 0, -0.5]
    ])
    cand1 = np.vstack([corners, face_centers])
    candidates.append(cand1)
    
    # Candidate 2: Icosahedron (12) + 2 random
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize ico
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    norms[norms == 0] = 1
    ico_verts = ico_verts / norms
    # Add 2 points on sphere
    rng = np.random.default_rng(42)
    extra = rng.normal(size=(2, 3))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    cand2 = np.vstack([ico_verts, extra])
    candidates.append(cand2)
    
    # Candidate 3: Random points in sphere
    rng2 = np.random.default_rng(123)
    cand3 = rng2.normal(size=(N, D))
    cand3 /= np.linalg.norm(cand3, axis=1, keepdims=True)
    candidates.append(cand3)
    
    # Optimization function
    def objective_and_grad(x, lam=50.0):
        points = x.reshape(N, D)
        
        # Compute pairwise squared distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=-1)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        d2_flat = d2[mask]
        
        if len(d2_flat) == 0:
            return 1.0, np.zeros_like(x)
            
        min_d2 = np.min(d2_flat)
        max_d2 = np.max(d2_flat)
        
        # If min_d2 is too small, points are coincident. Penalize.
        if min_d2 < 1e-10:
            return 1e10, np.zeros_like(x)
            
        # Soft max and min using log-sum-exp for stability
        # S_max = max_d2 + (1/lam) * log(sum(exp(lam * (d2 - max_d2))))
        # S_min = min_d2 - (1/lam) * log(sum(exp(-lam * (d2 - min_d2))))
        
        # Weights for gradient
        # w_max[i,j] = exp(lam * (d2[i,j] - max_d2)) / sum(exp(lam * (d2 - max_d2)))
        # Note: We only care about off-diagonal terms for the ratio, but the sum includes all.
        # To be consistent with S_max/S_min, we should use the full sum or just off-diagonal?
        # The objective is max/min. The soft version is logsumexp(lam*d2) / logsumexp(-lam*d2).
        # Usually, we sum over all pairs.
        
        # For numerical stability, we compute the log-sum-exp directly.
        # But for gradient, we need the softmax weights.
        
        # Let's use the full matrix d2, but set diagonal to -inf for max and +inf for min?
        # No, just mask them in the sum.
        
        # Create masked d2
        d2_masked = d2.copy()
        d2_masked[~mask] = -np.inf # For max, these should be 0 contribution
        # Wait, exp(-inf) = 0. So this works for sum.
        
        # For min, we want exp(-lam * d2). If d2 is large, exp is small.
        # If we set diagonal to +inf, exp(-lam * inf) = 0.
        
        # Let's compute S_max and S_min properly.
        # S_max = (1/lam) * log( sum_{i!=j} exp(lam * d2_ij) )
        # S_min = -(1/lam) * log( sum_{i!=j} exp(-lam * d2_ij) )
        
        # To avoid overflow, subtract max for max-sum, and min for min-sum (with sign flip).
        
        # Max part
        d2_for_max = d2.copy()
        d2_for_max[~mask] = -1e100 # Very small, so exp is 0
        m_max = np.max(d2_for_max)
        exp_max = np.exp(lam * (d2_for_max - m_max))
        sum_exp_max = np.sum(exp_max)
        S_max = m_max + (1.0 / lam) * np.log(sum_exp_max)
        
        # Min part
        d2_for_min = d2.copy()
        d2_for_min[~mask] = 1e100 # Very large, so exp(-lam * d2) is 0
        m_min = np.min(d2_for_min)
        exp_min = np.exp(-lam * (d2_for_min - m_min))
        sum_exp_min = np.sum(exp_min)
        S_min = m_min - (1.0 / lam) * np.log(sum_exp_min)
        
        if S_min < 1e-10:
            return 1e10, np.zeros_like(x)
            
        val = S_max / S_min
        
        # Gradient
        # d S_max / dx_i = sum_j w_max[i,j] * 2 * (x_i - x_j)
        # where w_max[i,j] = exp(lam * (d2[i,j] - m_max)) / sum_exp_max
        # Note: The diagonal terms have d2=0, but they are masked out in the sum?
        # In the sum for S_max, we set diagonal to -1e100, so exp is 0. So w_max[i,i] = 0.
        
        w_max = exp_max / sum_exp_max
        # w_max is (N, N)
        
        # d S_min / dx_i = sum_j w_min[i,j] * 2 * (x_i - x_j)
        # where w_min[i,j] = exp(-lam * (d2[i,j] - m_min)) / sum_exp_min
        # Note: S_min = m_min - (1/lam) log(sum exp(-lam(d2-m_min)))
        # d S_min = - (1/lam) * (1/sum) * sum (exp * (-lam) d(d2)) = sum (exp/sum) d(d2)
        # So the sign is positive for the gradient of S_min with respect to d2?
        # Let's check: S_min decreases as d2 increases? No, min distance is the smallest d2.
        # If all d2 increase, min d2 increases.
        # Soft min: -logsumexp(-lam * d2). Derivative w.r.t d2 is sum(exp(-lam d2) * lam) / sum(exp(-lam d2)) = sum(w). Positive.
        # So d S_min / d x = sum w_min * 2(x_i - x_j).
        
        w_min = exp_min / sum_exp_min
        
        # Compute gradients
        # grad_S_max[i] = 2 * sum_j w_max[i,j] (x_i - x_j)
        # = 2 * (x_i * sum_j w_max[i,j] - sum_j w_max[i,j] x_j)
        
        sum_w_max_row = np.sum(w_max, axis=1)
        grad_S_max = 2.0 * (points * sum_w_max_row[:, np.newaxis] - (w_max @ points))
        
        sum_w_min_row = np.sum(w_min, axis=1)
        grad_S_min = 2.0 * (points * sum_w_min_row[:, np.newaxis] - (w_min @ points))
        
        # J = S_max / S_min
        # dJ = (S_min dS_max - S_max dS_min) / S_min^2
        
        grad_J = (S_min * grad_S_max - S_max * grad_S_min) / (S_min ** 2)
        
        return val, grad_J.flatten()
    
    # Optimization loop
    for cand in candidates:
        if time.time() > deadline: break
        
        x0 = cand.flatten()
        
        try:
            res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-6})
            
            if res.success or res.fun < best_score:
                x_opt = res.x.reshape(N, D)
                
                # Calculate actual raw score
                d2_opt = np.sum((x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :])**2, axis=-1)
                d2_opt_flat = d2_opt[~np.eye(N, dtype=bool)]
                min_d2 = np.min(d2_opt_flat)
                max_d2 = np.max(d2_opt_flat)
                
                if min_d2 > 1e-9:
                    raw_score = max_d2 / min_d2
                    if raw_score < best_score:
                        best_score = raw_score
                        best_points = x_opt
        except Exception:
            continue

    if best_points is None:
        # Fallback to seed
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
        
    return best_points
