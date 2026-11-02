"""
Strategy: Full-DOF Gradient Descent with Smooth Convex Packing Objective.
Instead of restricting to a symmetric 6-DOF cluster and using discrete SAT checks,
we treat all 12 inner hexagons as free variables (36 DOF) and use a smooth
objective function that penalizes collisions via softplus (differentiable) and
minimizes the size of the minimal enclosing regular hexagon. L-BFGS-B exploits
the gradients to navigate the high-dimensional landscape efficiently, allowing
the hexagons to break symmetry and pack tighter than any symmetric constraint.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)
APOHEM_UNIT = SQRT3 / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def min_outer_side(verts_all, outer_center, outer_angle_deg):
        """
        Compute minimal side length of a regular hexagon (center, angle) containing all points.
        """
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        max_proj = 0.0
        for na in normals:
            nx = math.cos(na)
            ny = math.sin(na)
            projs = (verts_all[:, 0] - outer_center[0]) * nx + (verts_all[:, 1] - outer_center[1]) * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        return max_proj / (SQRT3 / 2)

    def objective(params):
        # params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, ox, oy, oa]
        # Last 3 are outer center x, y, angle
        inner_params = params[:N*3]
        outer_cx = params[N*3]
        outer_cy = params[N*3+1]
        outer_angle = params[N*3+2]
        
        verts_list = []
        for i in range(N):
            cx = inner_params[i*3]
            cy = inner_params[i*3+1]
            ang = inner_params[i*3+2]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            verts_list.append(verts)
        
        all_verts = np.vstack(verts_list)
        
        # 1. Outer Hexagon Constraint (Size)
        # We want to minimize the side length. 
        # To make it differentiable and avoid hard constraints, we can just return the side length.
        # But we also need to ensure they are inside. 
        # Actually, if we minimize the side length of the hexagon that CONTAINS them, 
        # they are automatically inside by definition of the calculation.
        # So the objective is simply the side length.
        side = min_outer_side(all_verts, (outer_cx, outer_cy), outer_angle)
        
        # 2. Collision Penalty (Softplus)
        # For each pair, check SAT. If overlapping, add penalty.
        # To be differentiable, we can't use simple SAT boolean.
        # Instead, we can use a smooth approximation of the distance between polygons.
        # A simpler differentiable proxy: 
        # For each pair, compute the maximum depth of penetration? 
        # Or simpler: Use the fact that for convex polygons, if they overlap, 
        # the minimum distance is 0. 
        # A common smooth penalty is sum of softplus of (distance - margin) for closest points?
        # That's hard to compute gradients for directly without custom logic.
        
        # Alternative: Since we are minimizing the container, collisions are the main issue.
        # We can use a "soft" SAT penalty.
        # For each axis, if projections overlap, penalty += softplus(overlap_depth).
        
        penalty = 0.0
        for i in range(N):
            v1 = verts_list[i]
            for j in range(i+1, N):
                v2 = verts_list[j]
                
                # Get axes from both polygons
                axes = []
                for k in range(6):
                    edge1 = v1[(k+1)%6] - v1[k]
                    n1 = np.array([-edge1[1], edge1[0]])
                    norm1 = np.linalg.norm(n1)
                    if norm1 > 1e-9: axes.append(n1/norm1)
                    edge2 = v2[(k+1)%6] - v2[k]
                    n2 = np.array([-edge2[1], edge2[0]])
                    norm2 = np.linalg.norm(n2)
                    if norm2 > 1e-9: axes.append(n2/norm2)
                
                for axis in axes:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # Overlap depth
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap > 0:
                        # Softplus penalty for overlap
                        # scale=10 makes it steeper
                        penalty += 10.0 * (np.log(1 + np.exp(10.0 * overlap)) - np.log(2))
        
        return side + penalty

    # Initial Guess: Hexagonal packing
    np.random.seed(42)
    best_params = None
    best_score = float('inf')
    
    # Generate multiple random initial configurations to escape local minima
    # Config 1: Perfect hexagonal lattice
    # Center at 0,0. Ring 1 at r=2. Ring 2 at r=4.
    # But we have 12. 
    # Let's start with a loose lattice and let the optimizer compress it.
    
    def make_initial_config(seed_offset=0):
        np.random.seed(42 + seed_offset)
        params = np.zeros(N*3 + 3)
        # Place them in a rough hexagonal pattern
        # 1 center
        params[0:3] = [0, 0, 0]
        # 6 ring 1
        for k in range(6):
            ang = np.radians(60*k)
            r = 2.1 # Slightly larger than 2.0 to avoid initial collision
            idx = 1 + k
            params[idx*3:idx*3+3] = [r*np.cos(ang), r*np.sin(ang), 0]
        # 5 ring 2 (skip one to make 12 total? No, 1+6+5=12. Yes.)
        for k in range(5):
            ang = np.radians(60*k + 30)
            r = 4.1
            idx = 7 + k
            params[idx*3:idx*3+3] = [r*np.cos(ang), r*np.sin(ang), 0]
        
        # Outer center and angle
        params[N*3] = 0
        params[N*3+1] = 0
        params[N*3+2] = 0
        
        # Add small noise to break symmetry
        params[:N*3] += np.random.uniform(-0.1, 0.1, N*3)
        return params

    # Multi-start L-BFGS-B
    num_starts = 5
    for i in range(num_starts):
        if time.time() > deadline - 30:
            break
        x0 = make_initial_config(i)
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 2000})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = make_initial_config(0)
        best_score = objective(best_params)

    # Polish with a few more iterations or different method if time permits
    if time.time() < deadline - 10:
        try:
            res2 = minimize(objective, best_params, method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 1000})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[i*3]
        cy = best_params[i*3+1]
        ang = best_params[i*3+2]
        inner.append((float(cx), float(cy), float(ang)))
    
    center = (float(best_params[N*3]), float(best_params[N*3+1]))
    angle = float(best_params[N*3+2])
    
    # Calculate final side length precisely
    verts_list = []
    for cx, cy, ang in inner:
        verts_list.append(get_hex_verts(cx, cy, 1.0, ang))
    all_verts = np.vstack(verts_list)
    side = min_outer_side(all_verts, center, angle)
    
    # Validation: Ensure all vertices are inside the outer hexagon
    # If not, expand side slightly
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-8:
                return False
        return True

    valid = True
    for cx, cy, ang in inner:
        verts = get_hex_verts(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        side *= 1.01

    return inner, center, float(side), float(angle)
