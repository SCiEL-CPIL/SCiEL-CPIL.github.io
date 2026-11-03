"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Regular hexagon with side length s, centered at origin, rotated by angle phi
    # Vertices: v_k = s * (cos(phi + k*pi/3), sin(phi + k*pi/3)) for k=0..5
    # Support function in direction theta: h(theta) = s * max_k cos(theta - phi - k*pi/3)
    # For a regular hexagon, the support function is periodic with period pi/3
    # h(theta) = s * max(1, cos(theta - phi), cos(theta - phi - pi/3), ...)
    # Actually, for a regular hexagon with circumradius R = s (since side = circumradius for regular hexagon)
    # Wait, for a regular hexagon, side length = circumradius.
    # The support function in direction theta is R * max_k cos(theta - angle_k)
    # where angle_k are the directions of the vertices.
    
    # Let's define the geometry more carefully.
    # A regular hexagon with side length s has circumradius s.
    # If centered at origin with one vertex at angle 0, vertices are at angles k*pi/3.
    # The support function h(theta) = s * max_k cos(theta - k*pi/3)
    # But we can rotate the hexagon by angle alpha.
    # h(theta) = s * max_k cos(theta - alpha - k*pi/3)
    
    # For a convex polygon, a point p is inside if and only if for all support directions,
    # the projection of p onto that direction is <= the support function value.
    # For a hexagon with 6 sides, the relevant directions are the normals to the sides.
    # The side normals are at angles alpha + pi/6 + k*pi/3 for k=0..5.
    # The support function in direction normal_k is s * cos(pi/6) = s * sqrt(3)/2.
    # Wait, let me re-derive.
    
    # Regular hexagon centered at origin, side length s.
    # Vertices at distance s from center.
    # If vertex 0 is at angle 0: v_k = s * (cos(k*pi/3), sin(k*pi/3))
    # Side k connects v_k and v_{k+1}.
    # Normal to side k points outward.
    # Midpoint of side k: s * (cos(k*pi/3 + pi/6), sin(k*pi/3 + pi/6))
    # The normal direction is k*pi/3 + pi/6.
    # Distance from center to side = s * cos(pi/6) = s * sqrt(3)/2.
    
    # So for the outer hexagon with center c_out, side length S, rotation phi_out:
    # Normal directions: n_k = phi_out + pi/6 + k*pi/3, k=0..5
    # Support value: S * sqrt(3)/2
    # Constraint: for each inner hexagon i, and for each normal direction n_k:
    #   (p_i - c_out) . n_k + h_inner_i(n_k) <= S * sqrt(3)/2
    # where h_inner_i(n_k) is the support function of inner hexagon i in direction n_k.
    # Inner hexagon i has center c_i, side length 1, rotation phi_i.
    # h_inner_i(n_k) = 1 * max_j cos(n_k - phi_i - j*pi/3)
    # But since the inner hexagon is regular with side 1, its support function in any direction
    # is 1 * max_j cos(theta - phi_i - j*pi/3) where theta is the direction.
    # Actually, for a regular hexagon with circumradius 1 (side=1), the support function is:
    # h(theta) = max_j cos(theta - phi_i - j*pi/3)
    # This equals cos(theta - phi_i - round((theta - phi_i)/(pi/3))*pi/3) ... not quite.
    # Let's just compute it numerically for the 6 normal directions.
    
    def support_hex(theta, phi, s):
        """Support function of regular hexagon with center at origin, side s, rotation phi, in direction theta."""
        angles = phi + np.arange(6) * np.pi / 3
        vals = np.cos(theta - angles)
        return s * np.max(vals)
    
    def compute_required_side(inner_positions, inner_angles, outer_center, outer_rotation):
        """Compute the minimum side length of outer hexagon to contain all inner hexagons."""
        # inner_positions: array of shape (N, 2)
        # inner_angles: array of shape (N,)
        # outer_center: array of shape (2,)
        # outer_rotation: scalar
        
        # Normal directions of outer hexagon
        normal_dirs = outer_rotation + np.pi/6 + np.arange(6) * np.pi/3
        
        # For each normal direction, compute the maximum extent
        required = np.zeros(6)
        for k in range(6):
            n_dir = normal_dirs[k]
            n_vec = np.array([np.cos(n_dir), np.sin(n_dir)])
            max_extent = -np.inf
            for i in range(N):
                # Vector from outer center to inner center
                d = inner_positions[i] - outer_center
                proj = np.dot(d, n_vec)
                # Support of inner hexagon in direction n_dir
                h_inner = support_hex(n_dir, inner_angles[i], 1.0)
                extent = proj + h_inner
                if extent > max_extent:
                    max_extent = extent
            # Also need to check the opposite side
            # The constraint is: max_extent <= S * sqrt(3)/2
            # But we also need: -min_extent <= S * sqrt(3)/2, i.e., min_extent >= -S * sqrt(3)/2
            # So we need both max_extent and -min_extent to be <= S * sqrt(3)/2
            # Let's compute min_extent too
            min_extent = np.inf
            for i in range(N):
                d = inner_positions[i] - outer_center
                proj = np.dot(d, n_vec)
                # Support in opposite direction
                h_inner_opp = support_hex(n_dir + np.pi, inner_angles[i], 1.0)
                # The extent in the negative direction
                extent_opp = -proj + h_inner_opp
                if extent_opp > min_extent:
                    min_extent = extent_opp
            
            # The required half-width in this direction is max(max_extent, min_extent)
            # S * sqrt(3)/2 >= max(max_extent, min_extent)
            # S >= 2/sqrt(3) * max(max_extent, min_extent)
            required[k] = 2.0 / np.sqrt(3.0) * max(max_extent, min_extent, 0.0)
        
        return np.max(required)
    
    def objective(params):
        """Objective function: minimize the outer side length."""
        # params: 36 + 3 = 39 parameters
        # 0..23: inner hexagon positions (12 * 2)
        # 24..35: inner hexagon angles (12)
        # 36..37: outer center (2)
        # 38: outer rotation
        
        inner_pos = params[:24].reshape(12, 2)
        inner_angles = params[24:36]
        outer_center = params[36:38]
        outer_rotation = params[38]
        
        return compute_required_side(inner_pos, inner_angles, outer_center, outer_rotation)
    
    # Initial guess: place hexagons in a compact arrangement
    # A hexagonal lattice arrangement
    def initial_guess():
        # Place 12 hexagons in a 3x4 or similar grid
        # Hexagonal lattice: spacing between centers = 2 * sqrt(3) for side 1 hexagons
        # Actually, for two adjacent hexagons with side 1, center-to-center distance = 2 * sqrt(3)/2 * 2 = 2*sqrt(3)? 
        # No. The distance from center to side is sqrt(3)/2. Two adjacent hexagons share a side,
        # so center-to-center distance = 2 * sqrt(3)/2 = sqrt(3).
        
        # Let's use a hexagonal packing arrangement
        # 12 hexagons: maybe a 4x3 grid or a hexagonal cluster
        # For now, use a simple grid
        positions = []
        angles = []
        # Hexagonal lattice spacing
        dx = np.sqrt(3)
        dy = 1.5
        
        # 4 columns, 3 rows
        for row in range(3):
            for col in range(4):
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                positions.append([x, y])
                angles.append(0.0)
        
        positions = np.array(positions)
        # Center the arrangement
        positions -= np.mean(positions, axis=0)
        
        return np.concatenate([positions.flatten(), np.array(angles), [0.0, 0.0], [0.0]])
    
    x0 = initial_guess()
    
    # Use Nelder-Mead or L-BFGS-B
    # Since the objective is non-smooth (max of smooth functions), use Nelder-Mead
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Try multiple restarts with different initial configurations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the initial guess
            x_start = x0 + np.random.randn(39) * 0.5
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best result
    inner_pos = best_params[:24].reshape(12, 2)
    inner_angles = best_params[24:36]
    outer_center = best_params[36:38]
    outer_rotation = best_params[38]
    
    # Convert angles to degrees
    inner = []
    for i in range(12):
        inner.append([float(inner_pos[i, 0]), float(inner_pos[i, 1]), float(np.degrees(inner_angles[i]))])
    
    side_length = float(best_val)
    
    return inner, (float(outer_center[0]), float(outer_center[1])), side_length, float(np.degrees(outer_rotation))
# EVOLVE-BLOCK-END
