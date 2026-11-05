import numpy as np
import time

N = 11

def construct():
    # Start time for budget
    t_start = time.time()
    
    # Define centers for 11 hexagons
    # Symmetric set: Center, Ring 1, and 4 from Ring 2 (corners)
    # Q2, Q5, Q7, Q10 from previous analysis
    sqrt3 = np.sqrt(3)
    centers = [
        (0.0, 0.0),
        (sqrt3, 0.0), (-sqrt3, 0.0),
        (sqrt3/2, 1.5), (-sqrt3/2, 1.5),
        (sqrt3/2, -1.5), (-sqrt3/2, -1.5),
        (3*sqrt3/2, 3.0), (-3*sqrt3/2, 3.0),
        (3*sqrt3/2, -3.0), (-3*sqrt3/2, -3.0)
    ]
    
    # Inner hexagons are all aligned with angle 0
    # Vertices of a unit hexagon with angle 0 (flat top? no, vertex at 0)
    # Let's assume standard orientation: vertices at 0, 60, 120...
    # Support function of unit hexagon (circumradius 1) in direction alpha:
    # h(alpha) = cos(alpha mod 60 deg) -- wait, if vertex is at 0, max proj is 1.
    # If normal is at 30 deg, max proj is cos(30) = sqrt(3)/2.
    # So h(alpha) = cos(alpha_mod_60) where alpha_mod_60 is in [-30, 30].
    
    def support_in(alpha):
        # alpha in radians
        # Map to [-pi/6, pi/6] relative to nearest vertex (0, pi/3, ...)
        # Actually, just take modulo pi/3 and center it.
        a = alpha % (np.pi / 3)
        if a > np.pi / 6:
            a = np.pi / 3 - a
        return np.cos(a)
        
    def calc_r(phi0):
        # phi0 is rotation of outer hexagon
        # Normals of outer hexagon are at phi0 + k*pi/3
        max_r = 0.0
        for cx, cy in centers:
            for k in range(6):
                nk_angle = phi0 + k * np.pi / 3
                nx = np.cos(nk_angle)
                ny = np.sin(nk_angle)
                # Projection of center
                proj_c = nx * cx + ny * cy
                # Support of inner hexagon in this direction
                # Inner hex angle is 0.
                # Direction of normal relative to inner hex is nk_angle - 0
                h_in = support_in(nk_angle)
                req_r = proj_c + h_in
                if req_r > max_r:
                    max_r = req_r
        return max_r

    # Optimize phi0 in [0, pi/3]
    # Use a simple grid search or minimize_scalar
    from scipy.optimize import minimize_scalar
    
    def objective(phi0):
        return calc_r(phi0)
        
    res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-6})
    
    best_phi0 = res.x
    best_r = res.fun
    
    # Add small safety margin
    best_r *= 1.001
    
    # Calculate side length S
    # r is inradius. S = 2 * r / sqrt(3)
    S = 2 * best_r / np.sqrt(3)
    
    # Check time
    elapsed = time.time() - t_start
    if elapsed > 140:
        # Return a safe fallback if optimization took too long (unlikely)
        S = 10.0
        best_phi0 = 0.0
        
    # Construct inner list
    # inner rows (x, y, angle_degrees)
    inner = [[cx, cy, 0.0] for cx, cy in centers]
    
    # Outer hexagon center (0,0), side S, rotation best_phi0 (in degrees? task says angle_degrees for inner, but for outer? "outer hexagon center (x, y), side length and rotation")
    # The interface says: (inner, center, side, angle).
    # angle is likely in degrees or radians? Inner uses angle_degrees.
    # Usually consistent. Let's assume degrees for outer too, or check seed.
    # Seed returns 0.0.
    # Let's return degrees.
    angle_deg = np.degrees(best_phi0)
    
    return inner, (0.0, 0.0), S, angle_deg
