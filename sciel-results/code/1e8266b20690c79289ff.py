import numpy as np
import time
from scipy.optimize import minimize

# Grid definition
X, Y = np.meshgrid(np.arange(-3, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
GRID_POINTS = np.column_stack((X.ravel(), Y.ravel()))
N_POINTS = len(GRID_POINTS)

def get_corridor_mask(x, y, a_deg):
    a_rad = np.deg2rad(a_deg)
    cos_a = np.cos(-a_rad)
    sin_a = np.sin(-a_rad)
    
    dx = GRID_POINTS[:, 0] - x
    dy = GRID_POINTS[:, 1] - y
    
    u = cos_a * dx - sin_a * dy
    v = sin_a * dx + cos_a * dy
    
    mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    return mask1 | mask2

def compute_area(path):
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    seq = np.vstack([start, path, end])
    mask = get_corridor_mask(0.0, 0.0, 0.0)
    
    for i in range(len(seq) - 1):
        p1 = seq[i]
        p2 = seq[i + 1]
        a1 = p1[2]
        a2 = p2[2]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for k in range(1, m + 1):
            t = k / m
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            a = a1 + t * (a2 - a1)
            step_mask = get_corridor_mask(x, y, a)
            mask &= step_mask
            
    return np.sum(mask) * 0.0001

def objective(h_param):
    h = h_param[0]
    # Circle center (-1, h)
    cx, cy = -1.0, h
    R = np.sqrt(1.0 + h*h)
    
    # Start point (0,0) relative to center: (1, -h)
    # End point (-2,0) relative to center: (-1, -h)
    
    angle_start = np.arctan2(-h, 1.0)
    angle_end = np.arctan2(-h, -1.0)
    
    # We want to go from angle_start to angle_end.
    # If h > 0, angle_start is negative, angle_end is more negative (or positive depending on branch).
    # Let's ensure we take the shorter arc or the one consistent with the path.
    # The path should stay in the upper half plane if h>0?
    # Start (0,0), End (-2,0). Center (-1, h).
    # If h>0, the arc is above the x-axis.
    # angle_start = atan2(-h, 1) (4th quad)
    # angle_end = atan2(-h, -1) (3rd quad)
    # We want to go from 4th to 3rd quad via 2nd/1st? No, that's the long way.
    # We want the short way, which goes through the top (if h>0, the center is above, so the arc bulges up).
    # Wait, if center is (-1, h) with h>0, the points (0,0) and (-2,0) are below the center.
    # The arc connecting them passing "above" the chord (which is on x-axis) is the minor arc if R is small?
    # Actually, if h is large, R is large, the arc is flatter.
    # Let's just linearly interpolate the angle.
    
    # We need to handle the angle wrap-around.
    # Let's define the path angle from start to end.
    # If h > 0, the arc is convex up.
    # angle_start is approx -h. angle_end is approx pi + h.
    # Difference is approx pi + 2h.
    # This is > pi. So it's the major arc?
    # No, the minor arc is the one that doesn't cross the negative x-axis?
    # Let's just sample the arc parametrically.
    
    t = np.linspace(0, 1, 10)
    # Interpolate angle
    # We want the angle to change monotonically.
    # Let's check the angles.
    a0 = np.arctan2(-h, 1.0)
    a1 = np.arctan2(-h, -1.0)
    
    # If h > 0, a0 is in (-pi/2, 0), a1 is in (-pi, -pi/2).
    # a1 < a0.
    # So we decrease angle.
    angles = a0 + t * (a1 - a0)
    
    x = cx + R * np.cos(angles)
    y = cy + R * np.sin(angles)
    
    # Rotation of the sofa
    a_sofa = 90.0 * t
    
    # Return intermediate points
    # Exclude t=0 and t=1
    t_int = np.linspace(0.1, 0.9, 8)
    angles_int = a0 + t_int * (a1 - a0)
    x_int = cx + R * np.cos(angles_int)
    y_int = cy + R * np.sin(angles_int)
    a_int = 90.0 * t_int
    
    path = np.column_stack((x_int, y_int, a_int))
    
    area = compute_area(path)
    return -area

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 500
    
    # Initial guess: h=0 (straight line, degenerate circle)
    # If h=0, R=1, center (-1,0).
    # a0 = 0, a1 = pi.
    # Angles 0 to pi.
    # x = -1 + cos(angle), y = sin(angle).
    # This is a semi-circle above the x-axis.
    
    params_init = np.array([0.5]) # Small bulge
    
    best_params = params_init.copy()
    best_score = -objective(params_init)
    
    # Optimize
    try:
        res = minimize(objective, params_init, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
        if res.fun < -best_score:
            best_score = -res.fun
            best_params = res.x
    except Exception:
        pass
        
    # Try another start
    if time.time() < deadline - 100:
        params_init2 = np.array([1.5])
        try:
            res2 = minimize(objective, params_init2, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
            if res2.fun < -best_score:
                best_score = -res2.fun
                best_params = res2.x
        except Exception:
            pass

    h = best_params[0]
    cx, cy = -1.0, h
    R = np.sqrt(1.0 + h*h)
    a0 = np.arctan2(-h, 1.0)
    a1 = np.arctan2(-h, -1.0)
    
    t_int = np.linspace(0.1, 0.9, 8)
    angles_int = a0 + t_int * (a1 - a0)
    x_int = cx + R * np.cos(angles_int)
    y_int = cy + R * np.sin(angles_int)
    a_int = 90.0 * t_int
    
    path = np.column_stack((x_int, y_int, a_int))
    
    return path
