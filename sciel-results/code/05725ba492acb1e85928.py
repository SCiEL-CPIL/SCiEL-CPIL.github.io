import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Centers of the 11 hexagons
    # 1 center
    centers = np.array([
        [0.0, 0.0]
    ])
    
    # 6 neighbors at dist sqrt(3)
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
        
    # 4 more at dist 3
    # Angles 30, 90, 150, 210
    r2 = 3.0
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    n_hex = len(centers)
    
    # Vertices of a unit hexagon (side 1)
    # Base vertices at angle 0 (pointy right)
    # v_k = (cos(60k), sin(60k))
    base_vertices = np.array([
        [np.cos(np.deg2rad(60*k)), np.sin(np.deg2rad(60*k))] for k in range(6)
    ])
    
    def get_all_vertices(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        Returns: Array of shape (n_hex * 6, 2) containing all vertices of all inner hexagons
        """
        theta_out = params[0]
        phis = params[1:]
        
        all_vertices = []
        for i in range(n_hex):
            phi = phis[i]
            # Rotate base vertices by phi
            c = np.cos(phi)
            s = np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            all_vertices.append(verts)
            
        return np.vstack(all_vertices)
    
    def objective(params):
        """
        Minimizes the side length S of the outer hexagon.
        Outer hexagon is centered at (0,0), rotated by theta_out.
        """
        theta_out = params[0]
        
        # Get all vertices
        verts = get_all_vertices(params)
        
        # Normals of the outer hexagon
        # A regular hexagon with side S and rotation theta has sides with normals at angles theta + 30 + k*60
        # The apothem (distance from center to side) is h = S * sqrt(3) / 2
        # Condition for point P to be inside: P . n_k <= h for all k
        # So S >= (P . n_k) / (sqrt(3)/2)
        
        normals = []
        for k in range(6):
            angle = np.deg2rad(30 + k * 60) + theta_out
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals) # shape (6, 2)
        
        # Project all vertices onto all normals
        # dots: (n_points, 6)
        dots = verts @ normals.T
        
        # Max projection
        max_proj = np.max(dots)
        
        # Required S
        S = max_proj / (np.sqrt(3) / 2)
        
        return S

    # Initial guess
    # theta_out = 0
    # phis = [0] * 11
    x0 = np.zeros(n_hex + 1)
    
    # Bounds
    # theta_out in [0, pi/3)
    # phis in [0, pi/3)
    bounds = [(0, np.pi/3)] * (n_hex + 1)
    
    best_score = None
    best_params = None
    
    # Multi-start
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline:
            break
            
        if i == 0:
            x_start = x0
        else:
            x_start = np.random.uniform(0, np.pi/3, n_hex + 1)
            
        try:
            res = minimize(objective, x_start, method='Powell', bounds=bounds, options={'maxiter': 200, 'disp': False})
            
            if res.fun < (best_score if best_score is not None else float('inf')):
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        # Fallback to current hardcoded solution
        return _fallback_solution()
        
    # Extract best solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
        
    # The side length is the objective value
    side = best_score
    
    # Add tiny margin for safety
    side += 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

def _fallback_solution():
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
        
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    return inner, (0.0, 0.0), side, 30.0

if __name__ == "__main__":
    construct()
