import numpy as np
import time
from scipy.optimize import minimize

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.linspace(0, 2*np.pi, 7)[:-1]
    
    def cost_func(params):
        # params is a 1D array of 25 elements: [x1, y1, ..., x11, y11, rot, cx, cy]
        points = params[:22].reshape(-1, 2)
        rot = params[22]
        center = params[23:25]
        
        # Calculate normals for the rotated hexagon
        angles = base_angles + rot
        normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
        
        # Vector from outer center to each point
        diffs = points - center
        
        # Projections onto normals
        projections = np.abs(diffs @ normals.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        # Calculate overlap penalty
        # We want distance >= MIN_DIST
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        # We want to minimize L_enc, but heavily penalize overlaps
        return L_enc + 100.0 * penalty

    # Random restarts to find a good local minimum
    best_L = float('inf')
    best_params = None
    
    # Initial guess: place points in a hexagonal grid pattern roughly
    for restart in range(100):
        # Random initialization
        if restart == 0:
            # Hexagonal lattice initialization
            pts = []
            for q in range(-3, 4):
                for r in range(-3, 4):
                    # Convert axial to cartesian
                    x = 1.5 * q
                    y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                    pts.append([x, y])
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            indices = np.argsort(dists)[:N]
            init_pts = pts[indices]
            # Add some noise
            init_pts += np.random.normal(0, 0.1, init_pts.shape)
        elif restart == 1:
            # Star pattern: 1 center, 6 around, 4 more
            init_pts = np.zeros((N, 2))
            # Center is (0,0)
            for i in range(6):
                angle = i * np.pi / 3
                init_pts[i+1] = [MIN_DIST * np.cos(angle), MIN_DIST * np.sin(angle)]
            # Add 4 more in second ring
            for i in range(4):
                angle = i * np.pi / 2 + np.pi/4
                init_pts[i+7] = [2.0 * MIN_DIST * np.cos(angle), 2.0 * MIN_DIST * np.sin(angle)]
            init_pts += np.random.normal(0, 0.05, init_pts.shape)
        else:
            init_pts = np.random.uniform(-3, 3, (N, 2))
            
        # Initial rotation and center
        init_rot = np.random.uniform(0, np.pi/3)
        init_center = np.mean(init_pts, axis=0)
        
        initial_params = np.concatenate([init_pts.flatten(), [init_rot, init_center[0], init_center[1]]])
        
        # Use Nelder-Mead for optimization
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.success:
            final_params = res.x
            final_pts = final_params[:22].reshape(-1, 2)
            final_rot = final_params[22]
            final_center = final_params[23:25]
            
            # Recalculate L_enc and check validity
            angles = base_angles + final_rot
            normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
            diffs = final_pts - final_center
            projections = np.abs(diffs @ normals.T)
            max_proj = np.max(projections)
            L_enc = max_proj / (np.sqrt(3) / 2)
            
            # Check for overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(final_pts[i] - final_pts[j])
                    if d < MIN_DIST - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and L_enc < best_L:
                best_L = L_enc
                best_params = final_params

    # If no valid solution found, fallback to a safe large solution
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the output
    inner_pts = best_params[:22].reshape(-1, 2)
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    outer_side = best_L
    outer_center = (float(best_params[23]), float(best_params[24]))
    outer_angle = float(best_params[22] * 180.0 / np.pi)
    
    return inner, outer_center, outer_side, outer_angle
