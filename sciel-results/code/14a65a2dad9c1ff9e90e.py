"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Define the rigid cluster of 12 hexagons in a 3x4 honeycomb lattice block
    # Local coordinates centered at (0,0)
    # Row 0 (y=-sqrt(3)): x = -3.5, -1.5, 0.5, 2.5
    # Row 1 (y=0):        x = -2.5, -0.5, 1.5, 3.5
    # Row 2 (y=sqrt(3)):  x = -3.5, -1.5, 0.5, 2.5
    
    local_pos = [
        (-3.5, -sqrt3), (-1.5, -sqrt3), (0.5, -sqrt3), (2.5, -sqrt3),
        (-2.5, 0.0),    (-0.5, 0.0),    (1.5, 0.0),    (3.5, 0.0),
        (-3.5, sqrt3),  (-1.5, sqrt3),  (0.5, sqrt3),  (2.5, sqrt3)
    ]
    
    # Vertices of a unit hexagon (side 1)
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def get_support_function(theta_rad):
        """
        Calculate the support function H_k(theta) for the 6 normal directions of the outer hexagon.
        The outer hexagon normals are at 30, 90, 150, 210, 270, 330 degrees.
        The inner hexagons are rotated by theta_rad.
        So the effective direction for the inner hexagon support is n_k - theta_rad.
        """
        H = np.zeros(6)
        for k in range(6):
            # Normal angle of outer hexagon
            normal_angle = math.radians(30 + k * 60)
            # Direction relative to inner hexagon frame
            # Inner hexagon is rotated by theta_rad, so its local frame is rotated by -theta_rad relative to global.
            # The vector n_k in the local frame of the inner hexagon is R(-theta_rad) n_k.
            # Angle of n_k is normal_angle.
            # Angle in local frame is normal_angle - theta_rad.
            local_angle = normal_angle - theta_rad
            
            # Calculate max dot product of unit_verts with unit vector at local_angle
            u = np.array([math.cos(local_angle), math.sin(local_angle)])
            projs = unit_verts @ u
            H[k] = np.max(projs)
        return H

    def calc_side(params):
        """
        params: [theta_deg, tx, ty]
        Returns the side length S of the outer hexagon.
        """
        if time.time() > deadline:
            return 1e10
            
        theta_deg, tx, ty = params
        theta_rad = math.radians(theta_deg)
        
        # Rotation matrix for the cluster
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        
        # Rotated local positions
        # P_i = R(theta) * p_i + t
        # We need max and min projections of P_i on the 6 outer normals.
        
        H = get_support_function(theta_rad)
        
        # Outer normals
        normals = []
        for k in range(6):
            ang = math.radians(30 + k * 60)
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0.0
        min_proj = 0.0
        
        for k in range(6):
            nx, ny = normals[k]
            
            max_proj_k = -1e9
            min_proj_k = 1e9
            
            for px, py in local_pos:
                # Rotate point
                rx = cos_t * px - sin_t * py
                ry = sin_t * px + cos_t * py
                
                # Translate
                gx = rx + tx
                gy = ry + ty
                
                proj = gx * nx + gy * ny
                
                if proj > max_proj_k:
                    max_proj_k = proj
                if proj < min_proj_k:
                    min_proj_k = proj
            
            # Add support function
            max_proj_k += H[k]
            min_proj_k += H[k] # H is symmetric for hexagon
            
            if max_proj_k > max_proj:
                max_proj = max_proj_k
            if min_proj_k < min_proj:
                min_proj = min_proj_k
        
        # The outer hexagon must contain the cluster.
        # Inradius R_in = S * sqrt(3) / 2
        # We need R_in >= max_proj AND R_in >= -min_proj
        # So S * sqrt(3) / 2 >= max(max_proj, -min_proj)
        
        required_inradius = max(max_proj, -min_proj)
        S = required_inradius * 2.0 / sqrt3
        
        return S

    # Initial guess
    # theta = 0, tx = 0, ty = 0
    init_params = [0.0, 0.0, 0.0]
    
    best_score = calc_side(init_params)
    best_params = init_params[:]
    
    # Optimize
    try:
        res = minimize(calc_side, init_params, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass

    # Try a few other starting points if time permits
    if time.time() < deadline - 10:
        for theta_init in [5, 10, 15, 20, 25]:
            if time.time() > deadline - 5:
                break
            p = [theta_init, 0.0, 0.0]
            try:
                res = minimize(calc_side, p, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
            except Exception:
                pass

    theta_deg, tx, ty = best_params
    
    # Construct inner hexagon positions
    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    
    inner = []
    for px, py in local_pos:
        rx = cos_t * px - sin_t * py
        ry = sin_t * px + cos_t * py
        gx = rx + tx
        gy = ry + ty
        # The inner hexagons are rotated by theta_deg
        inner.append([gx, gy, theta_deg])
        
    # Outer hexagon center is (0,0), side is best_score, angle is 0
    return inner, (0.0, 0.0), best_score, 0.0
# EVOLVE-BLOCK-END
