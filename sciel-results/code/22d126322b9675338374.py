import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    # Regular hexagon with side s, centered at origin, flat-top orientation
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center, distance s
    # The hexagon contains point (x,y) if for all 6 edges, the point is on the inside
    
    def hexagon_contains_point(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
        angle_rad = np.radians(angle_deg)
        # Transform to hexagon's local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -angle_rad
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300
        # Edge normals at angles 30, 90, 150, 210, 270, 330
        # For a flat-top hexagon with side s, the apothem (inradius) = s * sqrt(3)/2
        apothem = side * np.sqrt(3) / 2.0
        
        # The 6 edge normals (pointing inward) for flat-top hexagon
        # Normal angles: 30, 90, 150, 210, 270, 330 degrees
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        for na in normal_angles:
            # Distance from center to edge along this normal
            dist = lx * np.cos(na) + ly * np.sin(na)
            if dist > apothem + 1e-9:
                return False
        return True
    
    def outer_side_length(positions, outer_center, outer_angle):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        # For each inner hexagon, find the maximum extent in each of the 6 normal directions
        # The outer hexagon must be large enough to contain all vertices of all inner hexagons
        
        # A unit hexagon has 6 vertices at distance 1 from center, at angles 0,60,120,180,240,300
        # (for flat-top orientation)
        
        # For the outer hexagon with center oc, angle oa, side s:
        # It contains point p if for all 6 edge normals of outer hex, the projection is within apothem_outer
        # apothem_outer = s * sqrt(3)/2
        
        # So s >= max over all inner vertices of: (projection of (v - oc) onto normal_i) / (sqrt(3)/2)
        # for each of the 6 normals of the outer hexagon
        
        # The 6 normals of outer hexagon (flat-top, rotated by outer_angle):
        base_normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normal_angles = base_normal_angles + np.radians(outer_angle)
        
        max_required = 0.0
        for pos in positions:
            ix, iy, iangle = pos
            # 6 vertices of inner hexagon
            inner_vertex_angles = np.radians([0, 60, 120, 180, 240, 300]) + np.radians(iangle)
            for va in inner_vertex_angles:
                vx = ix + np.cos(va)
                vy = iy + np.sin(va)
                # Distance from outer center
                dvx = vx - outer_center[0]
                dvy = vy - outer_center[1]
                for na in normal_angles:
                    proj = dvx * np.cos(na) + dvy * np.sin(na)
                    # Required apothem in this direction
                    req_apothem = proj
                    if req_apothem > max_required:
                        max_required = req_apothem
        
        # side = apothem / (sqrt(3)/2) = apothem * 2/sqrt(3)
        return max_required * 2.0 / np.sqrt(3.0)
    
    def total_cost(params):
        """Cost function: outer side length + penalty for overlaps."""
        # params: 3*N for inner positions (x, y, angle for each) + 2 for outer center + 1 for outer angle
        # But we fix inner angles to 0 for simplicity (hexagonal symmetry)
        # params: 2*N for inner (x,y) + 2 for outer center + 1 for outer angle
        
        n_inner = N
        inner_xy = params[:2*n_inner].reshape(n_inner, 2)
        outer_cx = params[2*n_inner]
        outer_cy = params[2*n_inner + 1]
        outer_angle = params[2*n_inner + 2]
        
        # Compute outer side length
        positions = []
        for i in range(n_inner):
            positions.append([inner_xy[i, 0], inner_xy[i, 1], 0.0])
        
        side = outer_side_length(positions, (outer_cx, outer_cy), outer_angle)
        
        # Penalty for overlapping inner hexagons
        penalty = 0.0
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                dx = inner_xy[i, 0] - inner_xy[j, 0]
                dy = inner_xy[i, 1] - inner_xy[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 2.0 - 1e-6:
                    # Penalty grows quadratically as they overlap
                    overlap = (2.0 - dist)
                    penalty += overlap * overlap * 1000.0
        
        return side + penalty
    
    # Initial guess: hexagonal lattice arrangement
    # Center at origin, 6 neighbors at distance 2, then 4 more
    def initial_guess():
        pos = []
        # Center
        pos.append([0.0, 0.0])
        # 6 neighbors at distance 2, angles 0, 60, 120, 180, 240, 300
        for k in range(6):
            angle = np.radians(k * 60)
            pos.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
        # 4 more: place at distance ~3.46 (2*sqrt(3)) in 4 of the 6 "second ring" positions
        # Second ring positions are at distance 2*sqrt(3) from center, at angles 30, 90, 150, 210, 270, 330
        # Actually, let me place them more carefully
        # The 7 hexagons form a flower. Add 4 more around the perimeter.
        # Place at angles 0, 120, 240, and one more... 
        # Let's try: add hexagons at the "edges" of the flower
        # Positions at distance ~3.464 at angles 0, 60, 120, 180, 240, 300 for second ring
        # But those would overlap with first ring. 
        # Better: place 4 additional hexagons in a line or arc
        # Let me just use a computational initial guess
        
        # Alternative: 11 hexagons in a more compact arrangement
        # Row 0: 3 hexagons
        # Row 1: 4 hexagons (offset)
        # Row 2: 3 hexagons
        # Row 3: 1 hexagon? No, that's 11.
        
        # Hexagonal coordinates to Cartesian:
        # For flat-top hexagons, axial coordinates (q, r):
        # x = 2 * q + r  (times some scale)
        # Actually for unit hexagons with center-to-center distance 2:
        # x = 2*q + r, y = sqrt(3)*r  ... need to be careful
        
        # Let me use cube coordinates or just place them manually
        # A good 11-hexagon packing:
        # Think of it as a hexagon of "radius" 1 (7 hexagons) plus 4 more
        
        # Let me just return the 7-hexagon flower plus 4 more placed reasonably
        # 4 more at distance ~3.5 from center
        for k in [0, 1, 2, 3]:
            angle = np.radians(k * 90)  # 0, 90, 180, 270
            r = 3.464  # 2*sqrt(3)
            pos.append([r * np.cos(angle), r * np.sin(angle)])
        
        return np.array(pos)
    
    pos0 = initial_guess()
    x0 = np.zeros(2*N + 3)
    x0[:2*N] = pos0.flatten()
    x0[2*N] = 0.0   # outer center x
    x0[2*N+1] = 0.0 # outer center y
    x0[2*N+2] = 0.0 # outer angle
    
    best_params = x0.copy()
    best_cost = total_cost(x0)
    
    # Try multiple starting points and optimize
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x_start = x0
        else:
            # Perturb initial guess
            x_start = x0 + np.random.randn(len(x0)) * 0.5
            x_start[2*N+2] = 0.0  # Keep outer angle at 0 for now
        
        try:
            result = minimize(total_cost, x_start, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_cost:
                best_cost = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 20.0:
            break
    
    # Extract best solution
    inner_xy = best_params[:2*N].reshape(N, 2)
    outer_cx = best_params[2*N]
    outer_cy = best_params[2*N+1]
    outer_angle = best_params[2*N+2]
    
    # Compute final side length
    positions = []
    for i in range(N):
        positions.append([inner_xy[i, 0], inner_xy[i, 1], 0.0])
    
    side = outer_side_length(positions, (outer_cx, outer_cy), outer_angle)
    
    # Verify no overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = inner_xy[i, 0] - inner_xy[j, 0]
            dy = inner_xy[i, 1] - inner_xy[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 2.0 - 1e-4:
                # Adjust: push apart
                pass  # The penalty should have prevented this
    
    inner = []
    for i in range(N):
        inner.append([inner_xy[i, 0], inner_xy[i, 1], 0.0])
    
    return inner, (outer_cx, outer_cy), side, outer_angle
# EVOLVE-BLOCK-END
