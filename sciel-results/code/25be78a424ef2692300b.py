import numpy as np
import random
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute indices for all triangles
    tri_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    num_tris = len(tri_indices)
    
    # Beta for soft-min approximation
    # Higher beta -> closer to hard min, but steeper gradients
    beta = 100.0
    
    def soft_min_obj(params):
        # params shape (N, 3)
        # Convert to barycentric coordinates
        # Use softmax to ensure u+v+w=1 and u,v,w > 0
        # Subtract max for numerical stability
        p = params
        p_max = np.max(p, axis=1, keepdims=True)
        exp_p = np.exp(p - p_max)
        sum_exp = np.sum(exp_p, axis=1, keepdims=True)
        bary = exp_p / sum_exp # (N, 3)
        
        # Extract u, v, w
        # bary[:, 0] = u, bary[:, 1] = v, bary[:, 2] = w
        
        # Calculate areas for all triangles
        # Area is determinant of the 3x3 matrix of barycentric coords
        # B[i,j,k] = bary[tri_indices[:,0][i], tri_indices[:,1][i], tri_indices[:,2][i]]
        
        # Get barycentric coords for all points in triangles
        # Shape (num_tris, 3, 3)
        B = bary[tri_indices]
        
        # Calculate determinants
        # det([[u1, v1, w1], [u2, v2, w2], [u3, v3, w3]])
        # We can compute this efficiently
        # det = u1(v2*w3 - w2*v3) - v1(u2*w3 - w2*u3) + w1(u2*v3 - v2*u3)
        
        u1, v1, w1 = B[:, 0, 0], B[:, 0, 1], B[:, 0, 2]
        u2, v2, w2 = B[:, 1, 0], B[:, 1, 1], B[:, 1, 2]
        u3, v3, w3 = B[:, 2, 0], B[:, 2, 1], B[:, 2, 2]
        
        det = u1 * (v2 * w3 - w2 * v3) - v1 * (u2 * w3 - w2 * u3) + w1 * (u2 * v3 - v2 * u3)
        
        # Areas are absolute values of determinants
        areas = np.abs(det)
        
        # Soft min: -1/beta * log(sum(exp(-beta * areas)))
        # To maximize min area, we minimize -soft_min(areas)
        # -soft_min(areas) = 1/beta * log(sum(exp(-beta * areas)))
        
        # Numerical stability for log-sum-exp
        # Let z = -beta * areas
        # log(sum(exp(z))) = max_z + log(sum(exp(z - max_z)))
        z = -beta * areas
        z_max = np.max(z)
        if not np.isfinite(z_max):
            return 1e10
        log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)))
        
        return log_sum_exp / beta

    best_params = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Number of restarts
    # We don't know how fast minimize is, so we loop until deadline
    max_restarts = 200
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize parameters randomly
        # Uniform random in barycentric space is hard to sample directly in log-space
        # Just sample random positive numbers for the exponent inputs
        params0 = rng.uniform(0, 1, size=(N, 3))
        
        # Run optimizer
        try:
            res = minimize(soft_min_obj, params0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Evaluate the actual hard min area for this configuration
                # to get the true score
                p = res.x
                p_max = np.max(p, axis=1, keepdims=True)
                exp_p = np.exp(p - p_max)
                sum_exp = np.sum(exp_p, axis=1, keepdims=True)
                bary = exp_p / sum_exp
                
                B = bary[tri_indices]
                u1, v1, w1 = B[:, 0, 0], B[:, 0, 1], B[:, 0, 2]
                u2, v2, w2 = B[:, 1, 0], B[:, 1, 1], B[:, 1, 2]
                u3, v3, w3 = B[:, 2, 0], B[:, 2, 1], B[:, 2, 2]
                det = u1 * (v2 * w3 - w2 * v3) - v1 * (u2 * w3 - w2 * u3) + w1 * (u2 * v3 - v2 * u3)
                areas = np.abs(det)
                min_area = np.min(areas)
                
                if min_area > best_score:
                    best_score = min_area
                    best_params = res.x.copy()
                    
        except Exception as e:
            pass
            
    # Convert best_params to Cartesian coordinates
    if best_params is None:
        # Fallback to random points if optimization failed completely
        points = []
        for _ in range(N):
            u, v = rng.random(), rng.random()
            if u + v < 1:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
            else:
                points.append([0.5, (3 ** 0.5) / 6])
        return np.array(points)
        
    p = best_params
    p_max = np.max(p, axis=1, keepdims=True)
    exp_p = np.exp(p - p_max)
    sum_exp = np.sum(exp_p, axis=1, keepdims=True)
    bary = exp_p / sum_exp
    
    # Barycentric to Cartesian
    # P = u * A + v * B + w * C
    # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # x = u*0 + v*1 + w*0.5 = v + 0.5*w
    # y = u*0 + v*0 + w*sqrt(3)/2 = w*sqrt(3)/2
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = bary[:, 2]
    
    x = v + 0.5 * w
    y = w * (3 ** 0.5) / 2
    
    return np.column_stack((x, y))

# EVOLVE-BLOCK-END
