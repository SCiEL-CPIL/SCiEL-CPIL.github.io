"""Heilbronn triangle problem: maximize minimum triangle area for 11 points.

Strategy: Parameterized Symmetric Search.
Instead of optimizing 22 independent coordinates, we fix the topology to a 
symmetric configuration (center + 2 hexagonal rings) and optimize only 5 
parameters: radial scales for the rings, angular rotation, and a global 
scale factor. This reduces the search space dimension from 22 to 5, allowing 
global optimizers like Differential Evolution to navigate the landscape 
effectively without getting stuck in local minima of the high-dimensional 
stochastic landscape.
"""
import numpy as np
import time
import random
from scipy.optimize import differential_evolution

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 118.0  # safety margin before 120s limit

# Base symmetric configuration:
# 1 center point
# 6 points on a hexagonal ring (radius r1)
# 4 points on an outer hexagonal ring (radius r2), specifically alternating vertices
# Note: The equilateral triangle is not rotationally symmetric, so we must be careful.
# However, we can define the points relative to the centroid and then scale/rotate.
# A better approach for the triangle: 
# Use a "triangular lattice" subset or concentric rings centered at the centroid.
# Centroid of (0,0), (1,0), (0.5, S3H) is (0.5, S3H/3).
# Let's define points in polar coordinates relative to the centroid, then map to cartesian.
# Due to the triangle's 3-fold symmetry, we can exploit that.

CENTROID = np.array([0.5, S3H / 3.0])

def _generate_symmetric_config(params):
    """
    Generate 11 points based on 5 parameters:
    p0: scale factor for inner ring (6 points)
    p1: scale factor for outer ring (4 points)
    p2: angular rotation offset
    p3: radial offset for the 4 outer points (to break symmetry slightly if needed, or just scale)
    p4: global scale/shift adjustment? 
    Actually, let's stick to:
    p0: radius of inner hexagon (6 pts)
    p1: radius of outer hexagon (4 pts)
    p2: angle rotation
    p3: radial "wobble" or phase shift for outer ring? 
    Let's use:
    p0: r_inner
    p1: r_outer
    p2: theta_offset
    p3: r_center_offset_x (small shift)
    p4: r_center_offset_y (small shift)
    
    Wait, the center point should probably stay near the centroid for symmetry.
    Let's try:
    p0: r_inner (6 points)
    p1: r_outer (4 points)
    p2: theta_offset
    p3: center_x_offset
    p4: center_y_offset
    """
    r_inner = params[0]
    r_outer = params[1]
    theta_offset = params[2]
    cx_off = params[3]
    cy_off = params[4]
    
    center = CENTROID + np.array([cx_off, cy_off])
    pts = [center]
    
    # Inner ring: 6 points
    # Angles: 0, 60, 120, 180, 240, 300 degrees + offset
    for i in range(6):
        angle = np.deg2rad(60 * i + theta_offset)
        x = center[0] + r_inner * np.cos(angle)
        y = center[1] + r_inner * np.sin(angle)
        pts.append([x, y])
        
    # Outer ring: 4 points
    # To fit inside the triangle, we can't just use a full hexagon ring.
    # Let's place 4 points at specific angles that align with the triangle's symmetry or fill gaps.
    # A common pattern for 11 points might involve 3 corners, 3 mid-sides, 1 center, etc.
    # But the "ring" approach assumes radial symmetry around the centroid.
    # The triangle is only 3-fold symmetric. 
    # Let's adjust the outer ring to have 4 points. 
    # Maybe 2 points on one side and 2 on another? 
    # Or just 4 points at 0, 90, 180, 270? That breaks 3-fold symmetry.
    # Let's try to maintain 3-fold symmetry as much as possible.
    # 1 center + 6 inner (hexagon) + 4 outer? 4 is not divisible by 3.
    # So perfect 3-fold symmetry is impossible for 11 points with this ring structure.
    # Let's allow the outer ring to be asymmetric or use a different base.
    
    # Alternative Base:
    # 3 corners (scaled in)
    # 3 mid-sides
    # 3 "inner" points
    # 1 center
    # This is a 3-fold symmetric configuration.
    
    # Let's switch to a 3-fold symmetric parameterization.
    # Parameters:
    # p0: scale of corner points (distance from centroid)
    # p1: scale of mid-side points
    # p2: scale of inner ring points (3 points)
    # p3: rotation angle
    # p4: center point offset? (Usually 0 for symmetry, but allow small shift)
    
    # Re-implementing with 3-fold symmetry
    pts = []
    
    # Center point
    center = CENTROID + np.array([cx_off, cy_off])
    pts.append(center)
    
    # 3 Corner points
    # Vertices of equilateral triangle relative to centroid:
    # V1: (0.5, S3H) -> angle 90 deg, dist = 2/3 * height = 2/3 * S3H
    # V2: (0, 0) -> angle 210 deg
    # V3: (1, 0) -> angle 330 deg
    # Dist from centroid to vertex: R_tri = 2/3 * S3H
    R_tri = 2.0 / 3.0 * S3H
    
    corner_scale = params[0]
    mid_scale = params[1]
    inner_scale = params[2]
    rot = np.deg2rad(params[3])
    
    # Base angles for corners: 90, 210, 330
    corner_angles = np.array([90, 210, 330]) + np.rad2deg(rot)
    for a in corner_angles:
        rad = np.deg2rad(a)
        x = center[0] + corner_scale * R_tri * np.cos(rad)
        y = center[1] + corner_scale * R_tri * np.sin(rad)
        pts.append([x, y])
        
    # 3 Mid-side points
    # Midpoints of sides relative to centroid:
    # M1 (mid of top): (0.5, S3H/2) -> angle 90 deg? No.
    # Centroid (0.5, S3H/3).
    # Midpoint of (0,0)-(1,0) is (0.5, 0). Vector from centroid: (0, -S3H/3). Angle 270.
    # Midpoint of (1,0)-(0.5, S3H) is (0.75, S3H/2). Vector: (0.25, S3H/6).
    # Let's just use polar coordinates.
    # Dist from centroid to mid-side: R_mid = 1/3 * S3H
    R_mid = 1.0 / 3.0 * S3H
    
    mid_angles = np.array([270, 30, 150]) + np.rad2deg(rot)
    for a in mid_angles:
        rad = np.deg2rad(a)
        x = center[0] + mid_scale * R_mid * np.cos(rad)
        y = center[1] + mid_scale * R_mid * np.sin(rad)
        pts.append([x, y])
        
    # 3 Inner ring points
    # These can be placed between the center and the corners or mids.
    # Let's place them at angles 90, 210, 330 (aligned with corners) but at a different radius.
    # Or 270, 30, 150 (aligned with mids).
    # Let's use a parameter for their radius and angle offset relative to corners.
    # For simplicity, let's align them with the corners for now.
    inner_angles = np.array([90, 210, 330]) + np.rad2deg(rot)
    # Radius for inner points: let's use a fraction of R_tri
    # We have 1 center + 3 corners + 3 mids + 3 inner = 10 points.
    # We need 11.
    # So we have 1 center, 3 corners, 3 mids, 3 inner. That's 10.
    # Where is the 11th?
    # Maybe split the center into a small triangle? No, keep it simple.
    # Let's add a 4th inner point? Or make the "inner" ring have 4 points?
    # If we break 3-fold symmetry, we lose the benefit.
    # Let's stick to 3-fold symmetry. 
    # 1 center + 3 corners + 3 mids + 3 inner = 10.
    # We need 1 more.
    # Perhaps the "center" is actually a small triangle of 3 points? 
    # Or one of the rings has 4 points?
    # Let's try: 1 center + 3 corners + 3 mids + 4 inner.
    # The 4th inner point breaks symmetry.
    # Let's try: 3 corners + 3 mids + 3 inner + 1 center + 1 extra?
    # Let's just use 3-fold symmetry for 10 points and add 1 point that is also symmetric? 
    # Impossible for 11 points with 3-fold symmetry unless one point is at the center (fixed) and others are in orbits of 3.
    # 1 (center) + 3 + 3 + 3 = 10.
    # 1 + 3 + 3 + 4 = 11. (4 is not a multiple of 3)
    # So we cannot have full 3-fold symmetry for 11 points.
    # We must break symmetry slightly.
    
    # Let's define the 11th point as a small perturbation from the center or one of the orbits.
    # Or, let's use 1 center + 3 corners + 3 mids + 4 inner.
    # The 4 inner points can be at angles 0, 90, 180, 270 + rot?
    
    # Let's refine the parameterization:
    # p0: corner_scale
    # p1: mid_scale
    # p2: inner_scale (radius for 3 inner points aligned with corners)
    # p3: rot
    # p4: extra_point_angle (for the 11th point, which is an "inner" point at a different angle)
    # And we fix the radius of the 11th point to be same as inner_scale?
    
    # Let's generate 3 inner points at corner angles.
    for a in inner_angles:
        rad = np.deg2rad(a)
        x = center[0] + inner_scale * R_tri * np.cos(rad)
        y = center[1] + inner_scale * R_tri * np.sin(rad)
        pts.append([x, y])
        
    # Now we have 10 points.
    # Add 1 more point. Let's place it at an angle between the corners.
    # Angle: 150 + rot (midway between 90 and 210)
    extra_angle = 150 + np.rad2deg(rot)
    rad = np.deg2rad(extra_angle)
    x = center[0] + inner_scale * R_tri * np.cos(rad)
    y = center[1] + inner_scale * R_tri * np.sin(rad)
    pts.append([x, y])
    
    return np.array(pts)

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-7):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _objective(params):
    """
    Objective function for minimization (scipy minimizes).
    We want to maximize min_area, so we return -min_area.
    If points are outside, return a large penalty.
    """
    pts = _generate_symmetric_config(params)
    
    if not _check_inside(pts):
        return 100.0  # Penalty for invalid configuration
    
    min_area = _min_triangle_area(pts)
    return -min_area

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    
    # Bounds for parameters:
    # p0: corner_scale [0.5, 1.0] (1.0 is the actual vertex)
    # p1: mid_scale [0.5, 1.0]
    # p2: inner_scale [0.1, 0.8]
    # p3: rot [0, 360]
    # p4: extra_point_angle [0, 360] (Actually, I hardcoded the extra point angle to 150+rot in the generator. 
    # Let's make it a parameter to allow optimization of the 11th point's position relative to the symmetry.)
    
    # Wait, I hardcoded the extra point in _generate_symmetric_config. 
    # Let's modify _generate_symmetric_config to take 5 params:
    # p0: corner_scale
    # p1: mid_scale
    # p2: inner_scale
    # p3: rot
    # p4: extra_angle_offset (angle of the 11th point relative to the 150 deg mark)
    
    # I will rewrite _generate_symmetric_config to be cleaner.
    
    def _gen_config(params):
        corner_scale = params[0]
        mid_scale = params[1]
        inner_scale = params[2]
        rot_deg = params[3]
        extra_angle_off = params[4]
        
        center = CENTROID
        pts = [center]
        
        R_tri = 2.0 / 3.0 * S3H
        R_mid = 1.0 / 3.0 * S3H
        
        corner_angles = np.array([90, 210, 330]) + rot_deg
        for a in corner_angles:
            rad = np.deg2rad(a)
            x = center[0] + corner_scale * R_tri * np.cos(rad)
            y = center[1] + corner_scale * R_tri * np.sin(rad)
            pts.append([x, y])
            
        mid_angles = np.array([270, 30, 150]) + rot_deg
        for a in mid_angles:
            rad = np.deg2rad(a)
            x = center[0] + mid_scale * R_mid * np.cos(rad)
            y = center[1] + mid_scale * R_mid * np.sin(rad)
            pts.append([x, y])
            
        inner_angles = np.array([90, 210, 330]) + rot_deg
        for a in inner_angles:
            rad = np.deg2rad(a)
            x = center[0] + inner_scale * R_tri * np.cos(rad)
            y = center[1] + inner_scale * R_tri * np.sin(rad)
            pts.append([x, y])
            
        # 11th point
        # Base angle 150 (midway between 90 and 210)
        extra_angle = 150 + rot_deg + extra_angle_off
        rad = np.deg2rad(extra_angle)
        # Use same radius as inner points? Or separate? 
        # Let's use inner_scale for radius to keep params low.
        x = center[0] + inner_scale * R_tri * np.cos(rad)
        y = center[1] + inner_scale * R_tri * np.sin(rad)
        pts.append([x, y])
        
        return np.array(pts)

    def _obj(params):
        pts = _gen_config(params)
        if not _check_inside(pts):
            return 100.0
        return -_min_triangle_area(pts)

    bounds = [
        (0.5, 1.0),    # corner_scale
        (0.5, 1.0),    # mid_scale
        (0.1, 0.8),    # inner_scale
        (0, 360),      # rot
        (0, 360)       # extra_angle_off
    ]
    
    # Use Differential Evolution
    # seed=42 for reproducibility
    # maxiter: we have ~118s. DE is fast in 5D.
    # Let's run it with a reasonable number of iterations.
    
    try:
        result = differential_evolution(
            _obj, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-6,
            polish=True  # Use L-BFGS-B for final polish
        )
    except Exception as e:
        # Fallback to a heuristic if DE fails
        return _heuristic_fallback()
    
    best_params = result.x
    best_pts = _gen_config(best_params)
    
    # If the best point is outside (due to numerical issues in DE), try to project or fallback
    if not _check_inside(best_pts):
        # Try a few random valid points from the parameter space
        rng = random.Random(42)
        for _ in range(100):
            p = [rng.uniform(b[0], b[1]) for b in bounds]
            pts = _gen_config(p)
            if _check_inside(pts):
                best_pts = pts
                break
    
    # Final check
    if not _check_inside(best_pts):
        # Absolute fallback: grid points
        return _heuristic_fallback()
        
    return best_pts

def _heuristic_fallback():
    """Simple grid-based fallback."""
    pts = []
    # 3x4 grid approximation
    for i in range(3):
        for j in range(4):
            if len(pts) >= 11:
                break
            x = (j + 0.5) / 4.0
            y_max = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
            y = (i + 0.5) / 3.0 * y_max
            pts.append([x, y])
    while len(pts) < 11:
        pts.append([0.5, S3H / 3.0]) # Center
    return np.array(pts)

if __name__ == "__main__":
    pts = construct()
    print(pts)
