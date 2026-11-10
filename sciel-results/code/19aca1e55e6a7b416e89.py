import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    deadline = time.time() + 580  # Safety margin
    
    def get_hex_vertices(cx, cy, angle_deg, s=1.0):
        """Get vertices of a regular hexagon with side length s."""
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle_rad + i * np.pi / 3
            x = cx + s * np.cos(theta)
            y = cy + s * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)
    
    def max_separation(verts1, verts2):
        """
        Compute the maximum separation between two convex polygons using SAT.
        Returns the maximum gap. If > 0, they are separated.
        """
        max_sep = -np.inf
        
        # Check 6 edge normals from hex1
        for i in range(6):
            p1 = verts1[i]
            p2 = verts1[(i+1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            max1 = np.max(proj1)
            min1 = np.min(proj1)
            max2 = np.max(proj2)
            min2 = np.min(proj2)
            
            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                sep = -min(max1 - min2, max2 - min1)
            
            max_sep = max(max_sep, sep)
        
        # Check 6 edge normals from hex2
        for i in range(6):
            p1 = verts2[i]
            p2 = verts2[(i+1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            
            max1 = np.max(proj1)
            min1 = np.min(proj1)
            max2 = np.max(proj2)
            min2 = np.min(proj2)
            
            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                sep = -min(max1 - min2, max2 - min1)
            
            max_sep = max(max_sep, sep)
        
        return max_sep

    def compute_outer_side(centers, alpha):
        """
        Compute the minimum side length of a regular hexagon centered at (0,0) with orientation alpha
        that encloses all hexagons defined by centers.
        """
        all_verts = []
        for c in centers:
            v = get_hex_vertices(c[0], c[1], 0.0)
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        max_proj = 0
        for i in range(3):
            theta = alpha + i * np.pi / 3
            n = np.array([np.cos(theta), np.sin(theta)])
            projs = np.abs(all_verts @ n)
            max_proj = max(max_proj, np.max(projs))
        
        return 2.0 * max_proj / np.sqrt(3)

    def objective(params):
        """
        Minimize outer side length + penalty for intersections.
        """
        if time.time() > deadline:
            return 1e10
            
        centers = params.reshape(11, 2)
        
        verts = [get_hex_vertices(c[0], c[1], 0.0) for c in centers]
        
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                sep = max_separation(verts[i], verts[j])
                if sep < 1e-6:
                    penalty += (1e-6 - sep)
        
        S = compute_outer_side(centers, 0.0)
        
        return S + 1000.0 * penalty

    # Initial configuration
    centers = [[0.0, 0.0]]
    for i in range(6):
        theta = i * np.pi / 3
        centers.append([np.sqrt(3) * np.cos(theta), np.sqrt(3) * np.sin(theta)])
        
    # Add 4 more at the "bays" of the outer ring
    # Bays are at angles 30, 90, 150, 210, 270, 330
    # Let's place them at radius 2 * sqrt(3) * cos(30) = 3?
    # Actually, let's place them at angles 30, 90, 150, 210
    # Radius 2.5
    bay_angles = [np.pi/6, np.pi/2, 3*np.pi/6, 5*np.pi/6]
    for theta in bay_angles:
        centers.append([2.5 * np.cos(theta), 2.5 * np.sin(theta)])
        
    params0 = np.array(centers).flatten()
    
    best_params = params0.copy()
    best_score = objective(params0)
    
    try:
        result = minimize(objective, params0, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun
            
    except Exception:
        pass
        
    centers_final = best_params.reshape(11, 2)
    
    # Optimize alpha
    def S_alpha(alpha):
        return compute_outer_side(centers_final, alpha)
    
    alphas = np.linspace(0, np.pi/3, 100)
    S_vals = [S_alpha(a) for a in alphas]
    best_alpha = alphas[np.argmin(S_vals)]
    best_S = min(S_vals)
    
    inner_hex_data = np.zeros((11, 3))
    for i, c in enumerate(centers_final):
        inner_hex_data[i, 0] = c[0]
        inner_hex_data[i, 1] = c[1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, np.rad2deg(best_alpha)])
    outer_hex_side_length = float(best_S)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
