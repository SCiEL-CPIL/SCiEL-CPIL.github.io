import numpy as np
import time

N = 11
DEADLINE = 140.0

def _hex_support(angle, phi):
    """
    Calculate the distance from the center to the boundary of a unit regular hexagon
    rotated by 'angle' in the direction 'phi'.
    angle and phi are in radians.
    """
    # The support function of a regular polygon with n sides, apothem a, in direction phi
    # relative to a normal is a / cos(delta) where delta is angle from nearest normal.
    # For hexagon, normals are at k * pi/3.
    # If rotated by 'angle', normals are at angle + k * pi/3.
    # We want distance in direction phi.
    # Relative angle delta = phi - angle.
    # Find nearest multiple of pi/3 to delta.
    delta = np.mod(phi - angle, np.pi/3)
    # delta is in [0, pi/3). The distance to the nearest normal is min(delta, pi/3 - delta).
    d_norm = np.minimum(delta, np.pi/3 - delta)
    # Apothem of unit hexagon is sqrt(3)/2.
    a = np.sqrt(3) / 2
    return a / np.cos(d_norm)

def _compute_min_side(centers, inner_angle, outer_angle):
    """
    Compute the minimum side length S of the outer hexagon (rotated by outer_angle)
    required to contain the cluster of hexagons (rotated by inner_angle).
    """
    # Sample phi from 0 to 2*pi
    phis = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    max_s = 0.0
    
    # Precompute cluster support for all phis
    # h_cluster(phi) = max_i ( x_i cos phi + y_i sin phi + h_unit(phi - inner_angle) )
    # We can vectorize this partially.
    
    # h_unit for all phis given inner_angle
    h_unit = _hex_support(inner_angle, phis)
    
    # h_outer_unit for all phis given outer_angle
    h_outer_unit = _hex_support(outer_angle, phis)
    
    # Calculate h_cluster
    # centers is (N, 2)
    # x_i cos phi + y_i sin phi
    # shape (N, 360)
    proj = centers[:, 0] * np.cos(phis) + centers[:, 1] * np.sin(phis)
    # Add h_unit (broadcasted)
    # proj is (N, 360), h_unit is (360,)
    h_cluster = np.max(proj + h_unit, axis=0)
    
    # S(phi) = h_cluster(phi) / h_outer_unit(phi)
    # Avoid division by zero (should not happen for convex hex)
    s_vals = h_cluster / h_outer_unit
    
    return np.max(s_vals)

def _get_centers_pointy():
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        centers.append((x, y))
    return np.array(centers)

def construct():
    start_time = time.time()
    
    # Seed solution
    centers = _get_centers_pointy()
    inner_angle = np.deg2rad(30) # Pointy top
    outer_angle = np.deg2rad(30)
    
    best_s = _compute_min_side(centers, inner_angle, outer_angle)
    best_config = (centers, inner_angle, outer_angle)
    
    # Try Flat Top
    inner_angle_flat = 0.0
    s_flat = _compute_min_side(centers, inner_angle_flat, outer_angle)
    if s_flat < best_s:
        best_s = s_flat
        best_config = (centers, inner_angle_flat, outer_angle)
    
    # Optimize Outer Angle for the best configuration found so far
    # Use simple grid search or scipy
    from scipy.optimize import minimize_scalar
    
    def obj(theta_out):
        return _compute_min_side(centers, best_config[1], theta_out)
    
    # Search in range -30 to 30 degrees relative to current best?
    # The function is periodic with period 60 deg.
    # Let's search 0 to 60 deg.
    res = minimize_scalar(obj, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-4})
    
    if res.success:
        s_opt = res.fun
        if s_opt < best_s:
            best_s = s_opt
            best_config = (centers, best_config[1], res.x)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        pass # Return best so far
    
    # Prepare output
    inner = []
    for x, y in best_config[0]:
        inner.append([x, y, np.rad2deg(best_config[1])])
        
    outer_center = (0.0, 0.0)
    side = best_s
    outer_angle = np.rad2deg(best_config[2])
    
    return inner, outer_center, side, outer_angle

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
