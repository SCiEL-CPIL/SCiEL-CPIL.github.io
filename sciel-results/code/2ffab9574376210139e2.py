import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _min_area_bary(points_bary):
    # points_bary is (N, 2) array of (u, v). w = 1-u-v
    n = len(points_bary)
    min_a = float('inf')
    # Precompute w
    w = 1 - points_bary[:, 0] - points_bary[:, 1]
    # If any w < 0, it's invalid, but we handle that in objective
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Determinant of 3x3 matrix
                # | u_i v_i w_i |
                # | u_j v_j w_j |
                # | u_k v_k w_k |
                # u = points_bary[:,0], v = points_bary[:,1], w = 1-u-v
                u_i, v_i, w_i = points_bary[i,0], points_bary[i,1], w[i]
                u_j, v_j, w_j = points_bary[j,0], points_bary[j,1], w[j]
                u_k, v_k, w_k = points_bary[k,0], points_bary[k,1], w[k]
                
                det = u_i * (v_j * w_k - w_j * v_k) - v_i * (u_j * w_k - w_j * u_k) + w_i * (u_j * v_k - v_j * u_k)
                a = abs(det)
                if a < min_a:
                    min_a = a
    return min_a

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -float('inf')
    best_points = None
    
    # Initial configurations
    inits = []
    
    # 1. 10-point grid + center
    grid_10 = [
        (1.0, 0.0), (2/3, 1/3), (1/3, 2/3), (0.0, 1.0),
        (2/3, 0.0), (1/3, 1/3), (0.0, 2/3),
        (1/3, 0.0), (0.0, 1/3),
        (0.0, 0.0)
    ]
    # Wait, the list above has 10 points?
    # (1,0), (2/3,1/3), (1/3,2/3), (0,1) -> 4
    # (2/3,0), (1/3,1/3), (0,2/3) -> 3
    # (1/3,0), (0,1/3) -> 2
    # (0,0) -> 1
    # Total 10.
    # Add center (1/3, 1/3) is already there.
    # So add a point near center, e.g., (0.4, 0.4) -> w=0.2
    inits.append(np.array(grid_10 + [(0.4, 0.4)]))
    
    # 2. 15-point grid, remove 4
    grid_15 = []
    for i in range(4):
        for j in range(4-i):
            u = i/4.0
            v = j/4.0
            w = 1 - u - v
            if w >= 0:
                grid_15.append((u, v))
    # grid_15 has 15 points
    # Remove 4 points. Let's remove corners or edges to keep symmetry?
    # Or just take a random subset.
    # Better: start with 15 points and optimize, then drop the 4 with smallest contribution?
    # No, we must fix N=11.
    # Let's just use random perturbations of the 10-point grid.
    
    rng = np.random.RandomState(42)
    
    for i in range(20):
        # Perturb the 10-point grid + center
        base = np.array(grid_10 + [(0.4, 0.4)])
        noise = rng.uniform(-0.05, 0.05, size=base.shape)
        perturbed = base + noise
        # Clamp to valid region u>=0, v>=0, u+v<=1
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
        # Ensure u+v <= 1
        sum_uv = perturbed[:, 0] + perturbed[:, 1]
        mask = sum_uv > 1
        if np.any(mask):
            # Scale down u, v
            factor = 1.0 / sum_uv[mask]
            perturbed[mask, 0] *= factor
            perturbed[mask, 1] *= factor
        inits.append(perturbed)
        
        # Also try random Dirichlet
        bary = rng.dirichlet([1, 1, 1], size=N)
        inits.append(bary[:, :2])
        
        if time.time() > deadline - 10:
            break

    # Optimization loop
    for init in inits:
        if time.time() > deadline - 5:
            break
            
        def objective(params):
            uv = params.reshape(N, 2)
            # Penalty for going outside
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1 - u - v
            # If w < 0, penalty
            penalty = 0
            if np.any(w < 0):
                penalty = 100 * np.sum(w[w < 0]**2)
            if np.any(u < 0) or np.any(v < 0):
                penalty += 100 * (np.sum(u[u < 0]**2) + np.sum(v[v < 0]**2))
            
            min_a = _min_area_bary(uv)
            return -min_a + penalty
        
        try:
            res = minimize(
                objective,
                init.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            # Check if result is valid
            uv_opt = res.x.reshape(N, 2)
            w_opt = 1 - uv_opt[:, 0] - uv_opt[:, 1]
            if np.any(w_opt < -1e-6) or np.any(uv_opt < -1e-6):
                continue
                
            score = -res.fun
            # Subtract penalty if any (should be 0 if valid)
            # Actually, objective returns -min_a + penalty.
            # If penalty is 0, score is min_a.
            # We should check penalty explicitly.
            u_opt = uv_opt[:, 0]
            v_opt = uv_opt[:, 1]
            w_opt = 1 - u_opt - v_opt
            if np.any(w_opt < 0) or np.any(u_opt < 0) or np.any(v_opt < 0):
                continue
                
            min_a_opt = _min_area_bary(uv_opt)
            if min_a_opt > best_score:
                best_score = min_a_opt
                best_points = uv_opt.copy()
                
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = np.array(grid_10 + [(0.4, 0.4)])
        
    # Convert barycentric to Cartesian
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    pts = []
    for i in range(N):
        u, v = best_points[i]
        w = 1 - u - v
        p = u * A + v * B + w * C
        pts.append(p)
        
    return np.array(pts)
