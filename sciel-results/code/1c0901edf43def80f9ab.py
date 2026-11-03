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

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Returns 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts

    def polygons_overlap_sat(v1, v2):
        """
        Check if two convex polygons (lists of vertices) overlap using SAT.
        Returns True if they overlap.
        """
        def project(poly, axis):
            dots = [np.dot(p, axis) for p in poly]
            return min(dots), max(dots)
            
        def get_axes(poly):
            axes = []
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                # Edge vector
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                # Normal vector (perpendicular)
                # Normalize
                length = math.hypot(ex, ey)
                if length < 1e-9:
                    continue
                nx = -ey / length
                ny = ex / length
                axes.append((nx, ny))
            return axes

        # Check axes from v1
        for axis in get_axes(v1):
            min1, max1 = project(v1, axis)
            min2, max2 = project(v2, axis)
            if max1 < min2 or max2 < min1:
                return False # Separated

        # Check axes from v2
        for axis in get_axes(v2):
            min1, max1 = project(v1, axis)
            min2, max2 = project(v2, axis)
            if max1 < min2 or max2 < min1:
                return False # Separated

        return True # Overlap

    def compute_min_outer_side(positions, angles, outer_angle_deg):
        """
        Calculate minimum side length of outer hexagon centered at origin
        with given angle to contain all inner hexagons.
        """
        theta_rad = math.radians(outer_angle_deg)
        # Normals of the outer hexagon
        # Vertices are at outer_angle + k*60. Normals are at outer_angle + 30 + k*60.
        normals = []
        for k in range(6):
            normal_angle = theta_rad + math.pi/6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            normals.append((nx, ny))
            
        max_proj = 0.0
        for i in range(len(positions)):
            cx, cy = positions[i]
            angle = angles[i]
            verts = get_hex_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                for nx, ny in normals:
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
                        
        # inradius = S * sqrt(3) / 2
        # S = 2 * inradius / sqrt(3)
        return 2.0 * max_proj / sqrt3

    def objective(params):
        if time.time() > deadline:
            return 1e10

        # params: [x0, y0, ..., x11, y11, outer_angle]
        # We fix inner angles to 0 for this version to reduce search space and ensure stability.
        # If we need rotation, we can add it, but lattice packing is usually optimal with aligned hexes.
        
        positions = []
        for i in range(N):
            positions.append((params[2*i], params[2*i+1]))
        
        outer_angle = params[24]
        angles = [0.0] * N
        
        # Calculate min side
        side = compute_min_outer_side(positions, angles, outer_angle)
        
        # Check overlaps
        penalty = 0.0
        # Pre-compute vertices for reuse? 
        # For 12 hexes, 66 pairs. 6*6 vertices checks is fine.
        verts_cache = [get_hex_vertices(pos[0], pos[1], 1.0, 0.0) for pos in positions]
        
        for i in range(N):
            for j in range(i+1, N):
                # Quick distance check
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 4.0: # (2*inradius)^2 approx, actually min dist is sqrt(3) ~ 1.73, sq ~ 3.
                    # Detailed SAT check
                    if polygons_overlap_sat(verts_cache[i], verts_cache[j]):
                        penalty += 10.0
                        if penalty > 100.0:
                            return side + penalty

        return side + penalty

    # Initial Guess: Lattice 3x4
    # u = (1.5, sqrt(3)/2), v = (0, sqrt(3))
    u = (1.5, sqrt3/2)
    v = (0, sqrt3)
    
    pos_lattice = []
    for i in range(3):
        for j in range(4):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            pos_lattice.append((x, y))
            
    # Center it
    xs = [p[0] for p in pos_lattice]
    ys = [p[1] for p in pos_lattice]
    cx_off = (min(xs) + max(xs)) / 2.0
    cy_off = (min(ys) + max(ys)) / 2.0
    pos_lattice = [(x - cx_off, y - cy_off) for x, y in pos_lattice]
    
    init_lattice = []
    for x, y in pos_lattice:
        init_lattice.extend([x, y])
    init_lattice.append(0.0) # outer angle
    
    # Try a few different initial angles for the outer hexagon
    best_side = float('inf')
    best_params = None
    
    initial_guesses = [init_lattice]
    
    # Maybe try rotating the lattice slightly?
    # The lattice has a specific symmetry. The outer hexagon can rotate to fit better.
    # The objective function handles the outer angle.
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        # Evaluate initial guess
        val = objective(guess)
        if val < best_side:
            best_side = val
            best_params = guess[:]
            
        # Local optimization
        try:
            # Use Nelder-Mead for robustness with non-smooth penalties
            res = minimize(
                objective,
                guess,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True, 'initial_simplex': None}
            )
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x[:]
        except Exception as e:
            pass

    if best_params is None:
        best_params = init_lattice

    # Extract final answer
    inner = []
    for i in range(N):
        ix = best_params[2*i]
        iy = best_params[2*i+1]
        inner.append([ix, iy, 0.0]) # Fixed angle 0
        
    outer_cx = 0.0
    outer_cy = 0.0
    outer_angle = best_params[24]
    
    positions = [(best_params[2*i], best_params[2*i+1]) for i in range(N)]
    angles = [0.0] * N
    outer_side = compute_min_outer_side(positions, angles, outer_angle)
    
    # Final validation
    # Check if overlaps exist. If so, the penalty should have been high, 
    # but if Nelder-Mead converged to a bad local min with overlaps, we should check.
    verts_cache = [get_hex_vertices(pos[0], pos[1], 1.0, 0.0) for pos in positions]
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if polygons_overlap_sat(verts_cache[i], verts_cache[j]):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to initial lattice if it was valid
        # (It should be valid as it's a perfect lattice)
        positions = [(init_lattice[2*i], init_lattice[2*i+1]) for i in range(N)]
        angles = [0.0] * N
        outer_angle = init_lattice[24]
        outer_side = compute_min_outer_side(positions, angles, outer_angle)
        inner = [[init_lattice[2*i], init_lattice[2*i+1], 0.0] for i in range(N)]
        
    return inner, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END
