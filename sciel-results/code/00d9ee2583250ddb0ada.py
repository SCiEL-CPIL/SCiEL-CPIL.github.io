import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback to a known valid configuration
        # A 3x4 grid of hexagons is a safe bet, though not optimal
        # Let's return a simple line or grid
        pts = []
        for i in range(N):
            pts.append([float(i * 2 * APOTHEM), 0.0, 0.0])
        # Container needs to be large
        max_proj = max(abs(p[0]) for p in pts) + APOTHEM
        S = max_proj / APOTHEM
        return pts, [0.0, 0.0], float(S), 0.0

    def hexagon_normals(theta):
        """Returns 6 unit normals for a regular hexagon rotated by theta."""
        normals = []
        for k in range(6):
            angle = theta + k * math.pi / 3.0
            normals.append([math.cos(angle), math.sin(angle)])
        return np.array(normals)

    def smooth_max(values, temperature=0.1):
        """Soft-max approximation of the maximum."""
        shifted = values / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted)))
        return temperature * log_sum + np.max(values) # This is actually soft-max + max? 
        # Standard soft-max: T * log(sum(exp(v/T)))
        # As T -> 0, it approaches max(v).
        # To be safe and smooth, use T * log(sum(exp((v - max_v)/T))) + max_v
        # This is always >= max_v.
        # For minimization, we want to approximate max_v.
        # If we use T * log(sum(exp(v/T))), it is convex and smooth.
        pass

    def objective(params):
        """
        params: [cx, cy, theta, x1, y1, ..., x11, y11]
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf')
        
        cx = params[0]
        cy = params[1]
        theta = params[2]
        
        # Extract points
        p_coords = params[3:].reshape(N, 2)
        p_centered = p_coords - np.array([cx, cy])
        
        normals = hexagon_normals(theta)
        
        # Projections: (6, N)
        projs = normals @ p_centered.T
        abs_projs = np.abs(projs)
        
        # Smooth max of absolute projections
        # We want to approximate max(abs_projs)
        # Using log-sum-exp: T * log(sum(exp(abs_projs/T)))
        T = 0.1
        shifted = abs_projs / T
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted)))
        max_proj_smooth = T * log_sum
        
        # Container side length S
        # r_out = max_proj + APOTHEM
        # S = r_out / APOTHEM
        S = (max_proj_smooth + APOTHEM) / APOTHEM
        
        # Overlap penalty
        # d_ij >= 2 * APOTHEM
        penalty = 0.0
        min_dist = 2 * APOTHEM
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(p_coords[i] - p_coords[j])
                if d < min_dist:
                    # Quadratic penalty
                    penalty += (min_dist - d) ** 2
        
        # Weight for penalty
        # We want to minimize S, but penalty should be significant
        # If penalty is 0, S is the only term.
        # If penalty > 0, we add it.
        # To balance scales, we can multiply penalty by a factor.
        # S is around 2-3. Penalty can be large.
        # Let's scale penalty to be comparable to S.
        penalty_weight = 10.0
        return S + penalty_weight * penalty

    def initial_config():
        """Generate a compact initial hexagonal cluster."""
        pts = []
        # Center
        pts.append([0.0, 0.0])
        # Ring 1
        for k in range(6):
            angle = k * math.pi / 3.0
            r = 2 * APOTHEM
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            pts.append([x, y])
        # Ring 2 (partial, to get 11 total)
        # We have 7 so far. Need 4 more.
        # Place them in the gaps of Ring 1?
        # Distance between Ring 1 neighbors is 2 * APOTHEM.
        # They touch. So we can't put more in the gaps of Ring 1.
        # We must extend to Ring 2.
        # Ring 2 positions are at distance 4 * APOTHEM from center? No.
        # Hex grid spacing is 2 * APOTHEM.
        # Axial coords:
        # (0,0) -> (0,0)
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Next ring: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
        # Let's pick 4 from Ring 2.
        axial = [(2,0), (1,1), (0,2), (-1,2)]
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        
        return np.array(pts)

    best_S = float('inf')
    best_params = None
    best_pts = None
    best_theta = 0.0
    best_c = [0.0, 0.0]

    # Multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts_init = initial_config()
        
        # Add noise
        noise_scale = 0.1 if r == 0 else 0.5
        pts_perturbed = pts_init + np.random.normal(0, noise_scale, pts_init.shape)
        
        # Initial center is centroid
        c_init = np.mean(pts_perturbed, axis=0)
        
        # Initial theta
        theta_init = np.random.uniform(0, math.pi/6)
        
        params_init = np.concatenate(([c_init[0], c_init[1], theta_init], pts_perturbed.ravel()))
        
        # Optimize
        try:
            res = minimize(objective, params_init, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_S:
                # Check if valid (no overlaps)
                p_opt = res.x[3:].reshape(N, 2)
                c_opt = [res.x[0], res.x[1]]
                theta_opt = res.x[2]
                
                # Verify no overlaps
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if np.linalg.norm(p_opt[i] - p_opt[j]) < 2 * APOTHEM - 1e-5:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    # Calculate exact S
                    normals = hexagon_normals(theta_opt)
                    p_centered = p_opt - np.array(c_opt)
                    projs = normals @ p_centered.T
                    max_proj = np.max(np.abs(projs))
                    S_exact = (max_proj + APOTHEM) / APOTHEM
                    
                    if S_exact < best_S:
                        best_S = S_exact
                        best_params = res.x
                        best_pts = p_opt
                        best_theta = theta_opt
                        best_c = c_opt
                        
        except Exception as e:
            pass

    if best_pts is None:
        # Fallback
        pts = initial_config()
        c = [0.0, 0.0]
        theta = 0.0
        normals = hexagon_normals(theta)
        p_centered = pts - np.array(c)
        projs = normals @ p_centered.T
        max_proj = np.max(np.abs(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        return [[float(x), float(y), 0.0] for x, y in pts], [float(c[0]), float(c[1])], float(S), 0.0

    # Final polish: Use the best result to do a local search with tighter tolerance
    # We can try to refine the exact S by adjusting theta slightly
    # But Nelder-Mead should have done a decent job.
    
    # Construct output
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(math.degrees(best_theta))])
    
    center = [float(best_c[0]), float(best_c[1])]
    side = float(best_S)
    rotation = float(math.degrees(best_theta))
    
    return inner, center, side, rotation

