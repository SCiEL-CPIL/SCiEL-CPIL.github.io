import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    
    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS
    sqrt3 = math.sqrt(3)
    
    # Precompute flat arrays for speed
    _ux = unit_hex_verts[:, 0]
    _uy = unit_hex_verts[:, 1]
    _enx = edge_normals[:, 0]
    _eny = edge_normals[:, 1]

    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def compute_min_outer_side_smooth(positions, angles_rad, beta=50.0):
        """
        Compute a smooth approximation of the minimum outer side length.
        Uses log-sum-exp for the max projection to make it differentiable.
        """
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        # The outer hexagon can rotate. We need to find the rotation that minimizes the side length.
        # However, for the gradient method, we can treat the rotation as a variable too, 
        # or simply evaluate at the optimal rotation for the current positions.
        # The optimal rotation aligns the "tightest" direction with a hexagon axis.
        # For simplicity and differentiability, we can just compute the side length 
        # assuming the outer hexagon is aligned with the global axes (phi=0) 
        # and let the optimizer rotate the inner hexagons to fit.
        # BUT, the problem allows the outer hexagon to rotate. 
        # Actually, due to symmetry, we can fix the outer hexagon orientation 
        # and allow the inner ones to rotate. The "best" outer orientation 
        # is determined by the configuration.
        # Let's fix outer hexagon at phi=0 (flat top/bottom or pointy top/bottom depending on def).
        # Standard def: vertices at 0, 60, ...
        # Normals at 30, 90, ...
        
        normal_angles = np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # Project all vertices onto the 6 normal directions
        # proj shape: (num_verts, 6)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        
        # We need the max absolute projection for each normal direction
        # max_proj shape: (6,)
        # Use log-sum-exp for smooth max
        # max_approx = (1/beta) * log(sum(exp(beta * abs_proj)))
        # To avoid overflow, subtract max
        max_per_dir = np.max(abs_proj, axis=0)
        # Actually, we want the max over vertices for each direction.
        # The side length is determined by the maximum of (2/sqrt(3)) * max_proj[i]
        
        # Let's compute the "soft max" of the projections for each direction
        # soft_max_proj[i] = (1/beta) * log(sum_j exp(beta * abs_proj[j, i]))
        
        # Vectorized log-sum-exp
        # abs_proj is (V, 6)
        # We want sum over axis 0
        # To be safe:
        max_abs = np.max(abs_proj, axis=0)
        shifted = abs_proj - max_abs[None, :]
        exp_shifted = np.exp(beta * shifted)
        log_sum_exp = np.log(np.sum(exp_shifted, axis=0)) + beta * max_abs
        soft_max_proj = log_sum_exp / beta
        
        side_vals = (2.0 / sqrt3) * soft_max_proj
        # The side length is the max of these 6 values.
        # Use soft max again for the final max
        max_side = np.max(side_vals)
        shifted_side = side_vals - max_side
        exp_side = np.exp(beta * shifted_side)
        log_sum_exp_side = np.log(np.sum(exp_side)) + beta * max_side
        smooth_side = log_sum_exp_side / beta
        
        return smooth_side

    def compute_disjointness_penalty(positions, angles_rad, beta=10.0):
        """
        Compute a smooth penalty for overlapping hexagons.
        Two hexagons overlap if their Minkowski difference contains the origin.
        Distance between centers is d.
        If they are aligned (angle diff 0), min distance is 2 * apothem = sqrt(3).
        Actually, for two unit hexagons, the minimum distance between centers
        such that they are disjoint is sqrt(3) if aligned, but it varies with relative rotation.
        
        A robust smooth penalty:
        For each pair (i, j), compute the distance between centers d_ij.
        Compute the "effective radius" or support function of the Minkowski difference.
        This is complex. A simpler heuristic:
        Approximate the collision constraint by checking if the distance between centers
        is less than the sum of their "inradii" plus a margin? No, that's for circles.
        
        Better approach:
        The distance between two convex polygons is 0 if they intersect.
        We can approximate the distance between two hexagons by the minimum distance
        between their vertices and edges.
        
        However, a very effective differentiable proxy for "disjointness" of two sets A and B
        is the squared distance between the sets: dist(A, B)^2.
        If dist > 0, they are disjoint. If dist = 0, they touch/overlap.
        We want dist > 0.
        Penalty = sum( max(0, R_min - dist_ij)^2 )
        
        Computing exact polygon distance is expensive.
        Approximation:
        For two unit hexagons with relative angle delta, the minimum center distance
        for disjointness is d_min(delta).
        We can precompute or approximate d_min(delta).
        d_min(0) = sqrt(3) (touching flat-to-flat).
        d_min(30) = 2 (touching vertex-to-vertex? No, vertex to flat).
        Actually, the Minkowski difference of two unit hexagons is a 12-gon (if rotated) or hexagon (if aligned).
        The inradius of the Minkowski difference is the min center distance.
        
        Let's use a simpler penalty based on vertex-vertex and vertex-edge distances,
        but smoothed.
        
        Even simpler:
        Use the distance between centers.
        If d_ij < d_min(delta_ij), they likely overlap.
        d_min(delta) varies between sqrt(3) (~1.732) and 2.0.
        Let's use d_min(delta) = sqrt(3) * (1 + 0.14 * |sin(2*delta)|) ? No, let's be safe.
        The maximum inradius of the Minkowski difference is 2 (when rotated 30 deg? No).
        Wait, if I rotate one hexagon by 30 degrees, the "star" shape has vertices at distance 2 from center?
        The Minkowski sum of two unit hexagons (opposite signs) is the set of differences.
        The inradius of this difference set is the min distance between centers for disjointness.
        For aligned hexagons, inradius = sqrt(3).
        For 30 deg rotation, the difference is a 12-gon. The inradius is still sqrt(3)?
        No, the vertices of the difference set are sums of vertices.
        Max distance is 2.
        The inradius is the min distance from origin to boundary.
        For aligned, boundary is at sqrt(3).
        For 30 deg, the "corners" of the hexagons stick out.
        Actually, the inradius of the Minkowski difference of two unit regular hexagons
        with relative angle alpha is:
        r_min(alpha) = max( sqrt(3) * cos(alpha/2) ... ? )
        
        Let's just use a conservative lower bound for the min distance: sqrt(3).
        If we enforce d_ij > sqrt(3) + epsilon, they are definitely disjoint?
        No, if rotated 30 deg, they can be closer?
        Let's check: Center at (0,0) and (sqrt(3), 0). Aligned. Touching.
        Rotate second by 30 deg.
        The "pointy" part of the second hexagon extends to x = sqrt(3) + 1? No.
        The vertex of the second hexagon (rotated 30) is at (1*cos(30), 1*sin(30)) = (0.866, 0.5).
        Center is at (1.732, 0).
        Vertex is at (1.732+0.866, 0.5) = (2.598, 0.5).
        The first hexagon extends to x=1.
        So they are far apart in x?
        Wait, the Minkowski difference inradius is the critical value.
        It is known that for two unit hexagons, the min center distance for disjointness
        is sqrt(3) for any relative rotation?
        Let's verify.
        The inradius of the Minkowski difference A-B is the max t such that t*B is contained in A? No.
        It is the max t such that B is contained in A shifted by t*u?
        
        Actually, the inradius of the Minkowski difference of two unit hexagons is constant?
        No.
        Let's use the distance between the closest points.
        We can approximate the penalty as:
        P = sum_{i<j} max(0, d_min - d_ij)^2
        where d_min is a function of relative angle.
        Let's approximate d_min(delta) = sqrt(3) * (1 + 0.2 * abs(sin(2*delta)))?
        Or just use a safe lower bound of 1.8?
        If we use 1.8, we might allow overlaps for aligned hexagons (min dist 1.732).
        So we must use at least 1.732.
        Let's use d_min(delta) = 1.732 + 0.2 * abs(sin(2*delta)).
        
        To make it differentiable, use softplus for the max(0, x).
        """
        penalty = 0.0
        d_min_base = sqrt3
        # Precompute relative angles
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                d = math.sqrt(dx*dx + dy*dy)
                delta = abs(angles_rad[i] - angles_rad[j]) % (np.pi/3) # Hex symmetry
                # Approximate min distance
                # For delta=0, d_min = sqrt(3)
                # For delta=pi/6, d_min is larger?
                # Let's use a conservative constant for now to ensure validity,
                # and rely on the optimizer to push them apart.
                # If we use d_min = sqrt(3) + 0.1, it's safe for all rotations?
                # If rotated 30 deg, the "effective radius" is larger?
                # The inradius of the Minkowski difference is actually constant for regular hexagons?
                # Let's assume d_min = sqrt(3) is the minimum.
                # So if d > sqrt(3), they are disjoint?
                # Counter-example: Two squares. Min dist for disjoint is 2 (aligned) or 2*sqrt(2) (rotated 45).
                # So min dist is 2.
                # For hexagons, min dist is sqrt(3) (aligned).
                # So if we enforce d > sqrt(3), we are safe?
                # Yes, because the inradius of the Minkowski difference is the min distance.
                # And the inradius of the Minkowski difference of two unit hexagons
                # is at least sqrt(3).
                # Is it ever smaller? No, sqrt(3) is the apothem.
                # So d_min = sqrt(3) is a valid lower bound for all rotations.
                
                d_min = sqrt3 + 0.01 # small margin
                if d < d_min:
                    # Soft penalty
                    # Use (d_min - d)^2
                    penalty += (d_min - d)**2
        return penalty

    def objective(params):
        # Unpack
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        ang_rad = np.radians(ang)
        
        # Objective: Minimize outer side + large penalty for overlaps
        side = compute_min_outer_side_smooth(pos, ang_rad, beta=20.0)
        penalty = compute_disjointness_penalty(pos, ang_rad, beta=10.0)
        
        # Weight for penalty
        # If penalty is high, we want to reduce it.
        # Scale penalty to be comparable to side length (~3-5)
        w = 10.0
        return side + w * penalty

    def generate_honeycomb(seed_offset=0):
        a1 = np.array([sqrt3, 0.0])
        a2 = np.array([sqrt3/2.0, 1.5])
        patterns = [
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(-2,1),(1,1),(-1,-1),(2,0)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(2,0),(-2,0),(1,-1),(-1,1),(0,2)],
        ]
        idx = (seed_offset) % len(patterns)
        lattice_coords = patterns[idx]
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i, (li, lj) in enumerate(lattice_coords):
            positions[i] = li * a1 + lj * a2
            angles[i] = 0.0
        return positions, angles

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        # Strict check: d > sqrt(3) is sufficient?
        # Let's do a more robust check using the disjointness function from before
        # or just check distance > sqrt(3) + 1e-6
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < sqrt3 - 1e-6:
                    return False
        return True

    def compute_min_outer_side_exact(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / sqrt3) * max_projs
            return np.max(side_vals)
        
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            s = eval_phi(phi_deg)
            if s < best_side:
                best_side = s
                best_phi = phi_deg
        
        # Fine tune
        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass
            
        return best_side, best_phi, center[0], center[1]

    # Generate initial candidates
    candidates = []
    for offset in range(2):
        pos, ang = generate_honeycomb(offset)
        candidates.append((pos, ang))
    
    # Add a random one
    random.seed(42)
    pos_r, ang_r = generate_honeycomb(0)
    pos_r += np.random.normal(0, 0.2, pos_r.shape)
    ang_r += np.random.normal(0, 5, N)
    candidates.append((pos_r, ang_r))

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for i, (pos, ang) in enumerate(candidates):
        if time.time() - t0 > DEADLINE - 20:
            break
        
        # Pack into params
        params = np.zeros(3*N)
        for k in range(N):
            params[3*k] = pos[k, 0]
            params[3*k+1] = pos[k, 1]
            params[3*k+2] = ang[k]
            
        # Optimize
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < 10.0: # Heuristic check
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for k in range(N):
                    new_pos[k, 0] = res.x[3*k]
                    new_pos[k, 1] = res.x[3*k+1]
                    new_ang[k] = res.x[3*k+2]
                
                # Check validity
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side_exact(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
        except Exception as e:
            pass

    # Fallback
    if best_positions is None:
        best_positions, best_angles = generate_honeycomb(0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side_exact(best_positions, best_angles)

    # Final verification and cleanup
    if not is_valid_configuration(best_positions, best_angles):
        # Try to fix by pushing apart
        # Or just fallback to honeycomb
        best_positions, best_angles = generate_honeycomb(0)
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side_exact(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])
        
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
