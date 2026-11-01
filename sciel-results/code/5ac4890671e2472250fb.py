"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Helper to check if a point is inside a hexagon of side L centered at origin (flat-top)
    def point_in_hexagon(x, y, L):
        # Flat top hexagon inequalities:
        # |x| <= L
        # |y| <= sqrt(3)/2 * L
        # |sqrt(3)/2 * x + 1/2 * y| <= sqrt(3)/2 * L
        if abs(x) > L: return False
        if abs(y) > np.sqrt(3)/2 * L: return False
        if abs(np.sqrt(3)/2 * x + 0.5 * y) > np.sqrt(3)/2 * L: return False
        return True

    def hexagon_containment_penalty(x, y, L):
        """
        Returns penalty for a unit hexagon at (x,y) being outside a hexagon of side L at (0,0).
        We check all 6 vertices of the inner hexagon.
        """
        # Vertices of unit hexagon at (0,0)
        verts = [
            (1.0, 0.0),
            (0.5, np.sqrt(3)/2),
            (-0.5, np.sqrt(3)/2),
            (-1.0, 0.0),
            (-0.5, -np.sqrt(3)/2),
            (0.5, -np.sqrt(3)/2)
        ]
        
        penalty = 0.0
        for vx, vy in verts:
            px = x + vx
            py = y + vy
            # Calculate violation depth
            # We want max(0, violation)
            v1 = abs(px) - L
            v2 = abs(py) - np.sqrt(3)/2 * L
            v3 = abs(np.sqrt(3)/2 * px + 0.5 * py) - np.sqrt(3)/2 * L
            
            # If any violation is positive, the point is outside.
            # We take the max of the violations to represent the depth.
            depth = max(0, v1, v2, v3)
            penalty += depth**2
        return penalty

    def hexagon_overlap_penalty(x1, y1, x2, y2):
        """
        Returns penalty for two unit hexagons at (x1,y1) and (x2,y2) overlapping.
        Uses Minkowski sum logic.
        The difference vector d = (x1-x2, y1-y2) must be outside the hexagon of side 2.
        """
        dx = x1 - x2
        dy = y1 - y2
        
        # Hexagon of side 2 (Minkowski sum of two unit hexagons)
        # Inequalities for side 2:
        # |x| <= 2
        # |y| <= sqrt(3)
        # |sqrt(3)/2 * x + 1/2 * y| <= sqrt(3)
        
        s1 = 2.0 - abs(dx)
        s2 = np.sqrt(3) - abs(dy)
        s3 = np.sqrt(3) - abs(np.sqrt(3)/2 * dx + 0.5 * dy)
        
        # If all s_i > 0, they overlap.
        # The depth of overlap is the minimum slack.
        depth = min(s1, s2, s3)
        
        if depth > 0:
            return depth**2
        return 0.0

    def objective(params, L):
        """
        params: array of 2*N floats (x1, y1, ..., xN, yN)
        L: outer hexagon side length
        """
        penalty = 0.0
        # Containment
        for i in range(N):
            x = params[2*i]
            y = params[2*i+1]
            penalty += hexagon_containment_penalty(x, y, L)
            
        # Overlap
        for i in range(N):
            for j in range(i+1, N):
                x1 = params[2*i]
                y1 = params[2*i+1]
                x2 = params[2*j]
                y2 = params[2*j+1]
                penalty += hexagon_overlap_penalty(x1, y1, x2, y2)
                
        return penalty

    def is_valid_config(inner, L):
        """
        Hard check for validity.
        inner: list of (x, y, angle)
        L: side length
        """
        # Check containment
        for i in range(N):
            x, y, _ = inner[i]
            verts = [
                (1.0, 0.0), (0.5, np.sqrt(3)/2), (-0.5, np.sqrt(3)/2),
                (-1.0, 0.0), (-0.5, -np.sqrt(3)/2), (0.5, -np.sqrt(3)/2)
            ]
            for vx, vy in verts:
                if not point_in_hexagon(x+vx, y+vy, L):
                    return False
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, _ = inner[i]
                x2, y2, _ = inner[j]
                dx = x1 - x2
                dy = y1 - y2
                # Check if d is inside Minkowski hexagon (side 2)
                if (abs(dx) <= 2.0 and 
                    abs(dy) <= np.sqrt(3) and 
                    abs(np.sqrt(3)/2 * dx + 0.5 * dy) <= np.sqrt(3)):
                    return False
        return True

    # Generate initial lattice configuration
    def get_lattice_positions():
        positions = []
        # Hexagonal lattice vectors
        # v1 = (sqrt(3), 0)
        # v2 = (sqrt(3)/2, 3/2)
        
        # Generate a cluster of 12
        # Center (0,0)
        positions.append((0, 0))
        # Ring 1: 6
        for i in range(6):
            # Directions: 0, 60, 120, 180, 240, 300 degrees in lattice coords?
            # Lattice coords (q, r)
            # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
            coords = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
            q, r = coords[i]
            x = np.sqrt(3) * q + np.sqrt(3)/2 * r
            y = 1.5 * r
            positions.append((x, y))
            
        # Ring 2: Need 5 more
        # Candidates from ring 2: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,2)
        # Let's pick 5 that are roughly symmetric
        # (2,0), (1,-2), (0,-2), (-1,-2), (-2,-1) ?
        # Let's just pick the first 5 in a spiral
        ring2_coords = [(2,0), (2,-1), (1,-2), (0,-2), (-1,-2)]
        for q, r in ring2_coords:
            x = np.sqrt(3) * q + np.sqrt(3)/2 * r
            y = 1.5 * r
            positions.append((x, y))
            
        return positions

    best_L = 10.0
    best_params = None
    
    # Binary search on L
    L_low = 3.0
    L_high = 6.0
    
    # Find minimal L that allows a valid configuration
    # We will iterate a few times, tightening L_high
    
    for iter in range(10):
        if time.time() > deadline - 10:
            break
            
        L_mid = (L_low + L_high) / 2.0
        
        # Try to find a valid configuration for L_mid
        initial_pos = get_lattice_positions()
        params0 = np.array([p[0] for p in initial_pos] + [p[1] for p in initial_pos])
        
        # Optimize
        try:
            result = minimize(lambda p: objective(p, L_mid), params0, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            
            # Check if result is valid
            inner = [[float(result.x[2*i]), float(result.x[2*i+1]), 0.0] for i in range(N)]
            
            if is_valid_config(inner, L_mid):
                # Found valid config for L_mid
                L_high = L_mid
                best_L = L_mid
                best_params = result.x
            else:
                # Not valid, increase L
                L_low = L_mid
        except Exception:
            L_low = L_mid

    # If we didn't find a valid one, use a large L
    if best_params is None:
        L_high = 10.0
        initial_pos = get_lattice_positions()
        params0 = np.array([p[0] for p in initial_pos] + [p[1] for p in initial_pos])
        try:
            result = minimize(lambda p: objective(p, L_high), params0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            best_L = L_high
            best_params = result.x
        except:
            best_L = 10.0
            best_params = np.array([p[0] for p in get_lattice_positions()] + [p[1] for p in get_lattice_positions()])

    inner = [[float(best_params[2*i]), float(best_params[2*i+1]), 0.0] for i in range(N)]
    center = (0.0, 0.0)
    side = float(best_L)
    angle = 0.0
    
    # Final validation
    if not is_valid_config(inner, side):
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        center = (0.0, 0.0)
        side = 40.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
