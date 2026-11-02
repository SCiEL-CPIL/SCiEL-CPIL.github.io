import math
import numpy as np
import time
import random

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute normals for overlap check (relative orientation)
    # For two unit hexagons with same orientation, the Minkowski sum is a hexagon of side 2.
    # Non-overlap condition: |d . n_k| >= sqrt(3) for all 6 normals n_k.
    # Normals for a hexagon aligned with axes (pointy top? flat top?).
    # Standard axial to cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r.
    # Vertices of unit hexagon (centered at 0,0):
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # Wait, side length 1. Distance from center to vertex is 1.
    # Vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # This is a "pointy-top" hexagon if y is up?
    # Let's stick to the axial coordinate system which implies a specific orientation.
    # Axial (0,0) -> (0,0).
    # Axial (1,0) -> (sqrt(3), 0).
    # Axial (0,1) -> (sqrt(3)/2, 1.5).
    # The vector between (0,0) and (1,0) is (sqrt(3), 0). Length sqrt(3).
    # For non-overlap, distance must be >= 2 * apothem = sqrt(3).
    # So the normals for the "forbidden zone" (Minkowski sum) are perpendicular to the edges of the hexagon.
    # The Minkowski sum of two unit hexagons (same orientation) is a hexagon with side 2.
    # Its apothem is 2 * (sqrt(3)/2) = sqrt(3).
    # The normals to the edges of the Minkowski sum hexagon are the same as the normals to the edges of the unit hexagon.
    # Unit hexagon edges are perpendicular to directions: 30, 90, 150, 210, 270, 330 degrees?
    # Let's check vertices: (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), (-0.5, -0.866), (0.5, -0.866).
    # Edge 1: (1,0) to (0.5, 0.866). Vector (-0.5, 0.866). Normal (0.866, 0.5) = (cos 30, sin 30).
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    
    angles_norm = [math.pi/6 + k * math.pi/3 for k in range(6)]
    normals = np.array([[math.cos(a), math.sin(a)] for a in angles_norm])
    
    def check_overlap(pts):
        """
        Returns True if any two hexagons overlap.
        pts: (N, 2) array of centers.
        """
        n_pts = len(pts)
        # Vectorized check is faster, but N=11 is small.
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs = np.abs(d @ normals.T)
                if np.min(projs) < math.sqrt(3) - 1e-9:
                    return True
        return False

    def calc_S(pts, beta, cx, cy):
        """
        Calculate the side length S of the enclosing hexagon.
        beta: rotation of the outer hexagon.
        cx, cy: center of the outer hexagon.
        """
        # Normals of the outer hexagon
        outer_normals = np.array([[math.cos(beta + math.pi/6 + k * math.pi/3), 
                                   math.sin(beta + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ outer_normals.T
        
        # D is the max absolute projection
        D = np.max(np.abs(projs))
        
        # S = (D + APOTHEM) / APOTHEM
        return (D + APOTHEM) / APOTHEM

    def energy(pts, beta, cx, cy, temp):
        """
        Potential energy function for annealing.
        """
        # 1. Overlap penalty (hard)
        # We use a smooth approximation to allow gradients, or just a large penalty.
        # For annealing, a smooth potential is better.
        overlap_energy = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = pts[i] - pts[j]
                projs = np.abs(d @ normals.T)
                # min_proj is the distance to the boundary of the Minkowski sum
                min_proj = np.min(projs)
                if min_proj < math.sqrt(3):
                    # Violation
                    viol = math.sqrt(3) - min_proj
                    # Quadratic penalty
                    overlap_energy += viol * viol * 100.0
                    
        # 2. Enclosing size (soft)
        S = calc_S(pts, beta, cx, cy)
        
        # 3. Centering penalty (keep points near origin to avoid drift)
        center_penalty = np.sum(pts**2) * 0.01
        
        # 4. Regularization for beta? Not needed, it's a parameter.
        
        return S + overlap_energy + center_penalty

    def random_init():
        # Start with a loose grid or random cluster
        # A loose hexagonal grid is a good start
        pts = []
        # Generate a larger hexagonal grid and pick 11 points
        # Or just random points in a circle
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            pts.append([x, y])
        pts = np.array(pts)
        # Random beta
        beta = random.uniform(0, math.pi/6)
        # Center
        cx = 0.0
        cy = 0.0
        return pts, beta, cx, cy

    def anneal_step(pts, beta, cx, cy, temp, dt):
        """
        Perform one step of Langevin dynamics.
        """
        # Calculate forces
        forces = np.zeros_like(pts)
        
        # Overlap repulsion
        for i in range(N):
            for j in range(i+1, N):
                d = pts[i] - pts[j]
                projs = np.abs(d @ normals.T)
                min_proj = np.min(projs)
                if min_proj < math.sqrt(3):
                    viol = math.sqrt(3) - min_proj
                    # Force direction: towards increasing min_proj
                    # The gradient of min_proj w.r.t d is the normal corresponding to the min projection
                    # We need to push them apart along that normal.
                    # Identify which normal gave the min
                    min_idx = np.argmin(projs)
                    n = normals[min_idx]
                    # d is pts[i] - pts[j].
                    # We want to increase d . n if d . n > 0, or decrease if < 0.
                    # Actually, we want to increase |d . n|.
                    # So force on i is +n * sign(d . n), force on j is -n * sign(d . n)
                    # Wait, if d . n is positive, we want d to be larger in n direction.
                    # So force on i is along n, force on j is along -n.
                    sign_d = 1.0 if d @ n > 0 else -1.0
                    f = n * sign_d * viol * 10.0
                    forces[i] += f
                    forces[j] -= f

        # Centering force
        for i in range(N):
            forces[i] -= pts[i] * 0.01
            
        # Gradient of S w.r.t pts
        # S = (D + A) / A. D = max_k | (pts - c) . n_k^outer |
        # This is non-differentiable at the max. We use a soft-max or just subgradient.
        # For simplicity, use a smooth approximation: sum of exp
        # Or just use the subgradient of the max.
        outer_normals = np.array([[math.cos(beta + math.pi/6 + k * math.pi/3), 
                                   math.sin(beta + math.pi/6 + k * math.pi/3)] for k in range(6)])
        
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ outer_normals.T
        abs_projs = np.abs(projs)
        
        # Find the max projection
        max_val = np.max(abs_projs)
        
        # Gradient of max |x| is sign(x) * n_k
        # We sum the gradients of all active constraints (those close to max)
        # Soft max weight
        temp_s = 0.1
        weights = np.exp((abs_projs - max_val) / temp_s)
        weights /= np.sum(weights)
        
        # Force on pts to reduce D
        # dS/dpts = (1/A) * sum_k w_k * sign(projs_ik) * n_k
        # We want to minimize S, so we move pts in the direction of -grad S.
        # Grad S w.r.t pts_i = (1/A) * sum_k w_ik * sign(projs_ik) * n_k
        # Note: projs_ik = (pts_i - c) . n_k
        # sign(projs_ik) * n_k is the direction that increases the projection.
        # So we subtract this to decrease it.
        
        grad_S_pts = np.zeros_like(pts)
        for k in range(6):
            n_k = outer_normals[k]
            signs = np.sign(projs[:, k])
            w = weights[:, k]
            # Vectorized: sum over k of w_ik * signs_ik * n_k
            # w is (N, 6), signs is (N, 6)
            # grad_S_pts[i] = sum_k (w[i,k] * signs[i,k]) * n_k
            # This is a weighted sum of normals
            # We can compute this as: for each i, sum_k (w[i,k]*signs[i,k]) * n_k
            # Using einsum or loop
            for i in range(N):
                grad_S_pts[i] += np.sum([w[i,k] * signs[i,k] * n_k for k in range(6)])
        
        grad_S_pts /= APOTHEM
        forces -= grad_S_pts
        
        # Gradient of S w.r.t beta
        # dS/dbeta = (1/A) * sum_k w_k * d|proj_k|/dbeta
        # proj_k = (pts - c) . n_k(beta)
        # d proj_k / dbeta = (pts - c) . n_k'(beta)
        # n_k'(beta) = (-sin(angle), cos(angle))
        # d|proj_k|/dbeta = sign(proj_k) * (pts - c) . n_k'(beta)
        
        grad_S_beta = 0.0
        for k in range(6):
            n_k = outer_normals[k]
            angle_k = beta + math.pi/6 + k * math.pi/3
            n_k_prime = np.array([-math.sin(angle_k), math.cos(angle_k)])
            proj_k = projs[:, k]
            signs_k = np.sign(proj_k)
            # (pts - c) . n_k_prime for each point
            d_proj_dbeta = (pts_centered @ n_k_prime)
            # Sum over points weighted by w
            term = np.sum(weights[:, k] * signs_k * d_proj_dbeta)
            grad_S_beta += term
        grad_S_beta /= APOTHEM
        force_beta = -grad_S_beta
        
        # Gradient of S w.r.t c
        # dS/dc = - dS/dpts (since pts - c)
        # So force on c is + grad_S_pts (to minimize S, we want to move c in direction of grad_S_pts? 
        # Wait. S depends on pts - c. 
        # dS/dc = - dS/d(pts-c) = - grad_S_pts.
        # We want to minimize S, so we move c in direction of -dS/dc = grad_S_pts.
        force_c = np.sum(grad_S_pts, axis=0) / N # Average? No, c is a single point.
        # Actually, c is a parameter. The force on c is the negative gradient of the energy w.r.t c.
        # E = S(pts, beta, c).
        # dS/dc = - sum_i dS/dpts_i ? No.
        # S = f(max_k | (pts_i - c) . n_k |)
        # Let u_i = pts_i - c.
        # dS/dc = sum_i dS/du_i * du_i/dc = sum_i dS/du_i * (-I) = - sum_i dS/du_i.
        # So force on c (to minimize) is - dS/dc = sum_i dS/du_i.
        # dS/du_i is the gradient we computed as grad_S_pts[i].
        # So force_c = sum_i grad_S_pts[i].
        
        # Update positions
        # Langevin: v = v * friction + force * dt + noise * sqrt(temp)
        # We don't track velocity, so we do:
        # pts_new = pts + forces * dt + random_noise
        noise_pts = np.random.normal(0, math.sqrt(temp * dt), size=pts.shape)
        pts_new = pts + forces * dt + noise_pts
        
        noise_beta = np.random.normal(0, math.sqrt(temp * dt), size=1)[0]
        beta_new = beta + force_beta * dt + noise_beta
        beta_new = ((beta_new % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        noise_c = np.random.normal(0, math.sqrt(temp * dt), size=2)
        c_new = np.array([cx, cy]) + force_c * dt + noise_c
        
        return pts_new, beta_new, float(c_new[0]), float(c_new[1])

    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_c = (0.0, 0.0)

    # Multiple restarts
    n_restarts = 10
    steps_per_restart = 2000
    
    for r in range(n_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts, beta, cx, cy = random_init()
        
        # Cooling schedule
        temp_start = 1.0
        temp_end = 0.01
        temp = temp_start
        dt = 0.05
        
        for step in range(steps_per_restart):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            # Update temp
            if step < steps_per_restart * 0.5:
                temp = temp_start
            else:
                # Linear cooling
                progress = (step - steps_per_restart * 0.5) / (steps_per_restart * 0.5)
                temp = temp_start * (1 - progress) + temp_end * progress
                
            pts, beta, cx, cy = anneal_step(pts, beta, cx, cy, temp, dt)
            
            # Evaluate exact S and check overlap
            if not check_overlap(pts):
                S = calc_S(pts, beta, cx, cy)
                if S < best_S:
                    best_S = S
                    best_pts = pts.copy()
                    best_beta = beta
                    best_c = (cx, cy)

    # Final polish: local optimization of the best found
    if best_pts is not None:
        # Use scipy minimize on the exact objective
        from scipy.optimize import minimize
        
        def objective(params):
            n_pts = N
            pts = params[:2*n_pts].reshape(n_pts, 2)
            beta = params[2*n_pts]
            cx = params[2*n_pts+1]
            cy = params[2*n_pts+2]
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            if check_overlap(pts):
                return 1e10
            S = calc_S(pts, beta, cx, cy)
            return S

        init_params = np.concatenate([best_pts.ravel(), [best_beta, best_c[0], best_c[1]]])
        
        try:
            res = minimize(objective, init_params, method='L-BFGS-B', options={'maxiter': 200})
            if res.fun < best_S:
                n_pts = N
                best_pts = res.x[:2*n_pts].reshape(n_pts, 2)
                best_beta = ((res.x[2*n_pts] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                best_c = (res.x[2*n_pts+1], res.x[2*n_pts+2])
                best_S = res.fun
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        # Simple grid
        pts = []
        for i in range(3):
            for j in range(3):
                pts.append([i, j])
        pts = np.array(pts[:11])
        best_pts = pts
        best_S = 10.0
        best_beta = 0.0
        best_c = (0.0, 0.0)

    # Prepare output
    cx, cy = best_c
    angle_deg = math.degrees(best_beta)
    
    # Scale slightly to ensure no overlap due to floating point
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in best_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)
    
    # Recalculate S for scaled points
    outer_normals = np.array([[math.cos(best_beta + math.pi/6 + k * math.pi/3), 
                               math.sin(best_beta + math.pi/6 + k * math.pi/3)] for k in range(6)])
    pts_centered = pts_scaled - np.array([cx, cy])
    projs = pts_centered @ outer_normals.T
    D = np.max(np.abs(projs))
    S_final = (D + APOTHEM) / APOTHEM

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
