"""
Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: Gradient-based optimization (L-BFGS-B) with smooth penalty functions.
Instead of discrete SA, we treat the positions and angles as continuous variables
and minimize a smooth objective that penalizes overlaps and containment violations.
"""

import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _support_function(point, normal):
    return np.dot(point, normal)

def _min_enclosing_side_smooth(inner_positions, outer_center, outer_angle, beta=100.0):
    """
    Calculate the minimum side length of the outer hexagon required to contain all inner hexagons.
    Uses a soft-max approximation for the maximum over all vertices and faces.
    
    The distance of a point p from the center c in the direction of a face normal n 
    determines the required side length s such that p is inside.
    For a hexagon with inradius r_in = s * sqrt(3)/2, the condition is p . n <= r_in.
    So s >= 2 * (p . n) / sqrt(3).
    We need the max over all vertices of all inner hexagons and all 6 face normals of the outer hex.
    """
    cx, cy = outer_center
    outer_angle_rad = np.radians(outer_angle)
    
    # Normals for the 6 faces of the outer hexagon (pointing inward? No, outward from center to face)
    # Face k normal angle: 30 + k*60 degrees
    face_normals = []
    for k in range(6):
        a = np.radians(30 + k * 60)
        face_normals.append([np.cos(a), np.sin(a)])
    face_normals = np.array(face_normals)
    
    # Rotate face normals by outer_angle
    ca = np.cos(outer_angle_rad)
    sa = np.sin(outer_angle_rad)
    rot = np.array([[ca, -sa], [sa, ca]])
    face_normals = face_normals @ rot.T
    
    max_projection = 0.0
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        # Relative to outer center
        iverts_rel = iverts - np.array(outer_center)
        
        # Project all vertices onto all face normals
        # shape: (6, 6) -> max over vertices for each normal
        projections = iverts_rel @ face_normals.T
        
        # We want the maximum projection value across all vertices and normals
        # The required side length s is determined by max_proj = s * sqrt(3)/2
        # So s = 2 * max_proj / sqrt(3)
        
        # Use soft max to make it differentiable? 
        # Actually, max is non-differentiable. 
        # We can approximate max(x) with soft_max(x, beta) = (1/beta) * log(sum(exp(beta*x)))
        # However, for a single optimizer step, we can just take the max and rely on subgradients 
        # or use a smooth max. L-BFGS-B handles non-smoothness poorly. 
        # Let's use a smooth max approximation for the global max.
        
        # Combine all projections into one vector
        all_projs = projections.flatten()
        
        # Smooth max
        # Shift to avoid overflow
        m = np.max(all_projs)
        if m > 0:
            soft_max = m + np.log(np.sum(np.exp(beta * (all_projs - m)))) / beta
        else:
            soft_max = 0.0 # If all negative, side length is 0 (impossible physically, but mathematically)
            
        if soft_max > max_projection:
            max_projection = soft_max
            
    return 2.0 * max_projection / np.sqrt(3)

def _min_distance_hexagons_smooth(v1, v2, beta=50.0):
    """
    Estimate minimum distance between two convex hexagons.
    Uses a smooth approximation of the minimum distance between any pair of edges or vertices.
    For simplicity and speed, we approximate the distance by checking distance between centers 
    and subtracting the sum of inradii, but this is inaccurate for rotated hexagons.
    
    Better approach: 
    The distance between two convex polygons can be found by checking the minimum distance between edges.
    However, this is non-differentiable.
    
    Alternative: Use a potential energy function that is smooth.
    U = max(0, R1 + R2 - d_centers)^2 is a common approximation, but weak.
    
    Let's use a more rigorous smooth penalty:
    For each pair of edges (one from hex 1, one from hex 2), compute the minimum distance.
    Then take the soft-min of all these distances.
    If the soft-min is < 0, they overlap.
    
    This is computationally expensive (6x6=36 pairs per hex pair, 55 hex pairs).
    36 * 55 = 1980 edge pair checks per objective evaluation.
    With 150s budget, we can afford ~1000 evaluations. 1.9M ops is fine.
    """
    # v1, v2 are arrays of 6 vertices each
    min_dist = float('inf')
    
    # To speed up, we can first check center distance
    c1 = np.mean(v1, axis=0)
    c2 = np.mean(v2, axis=0)
    d_c = np.linalg.norm(c1 - c2)
    
    # Approximate inradius is sqrt(3)/2 ~ 0.866
    # If d_c > 2 * 1.0 (circumradius), they are definitely disjoint
    # If d_c < 2 * 0.866, they likely overlap
    # We need a smooth distance metric.
    
    # Let's compute the exact minimum distance between edges for the closest few pairs?
    # No, let's use a robust smooth approximation:
    # Distance between two line segments.
    
    # For efficiency in the optimizer, we will use a simplified smooth penalty:
    # Penalty = sum of exp(beta * (overlap_depth)) for all vertex-edge pairs?
    # Or simply: max(0, (R1+R2 - d_centers))^2 is too loose.
    
    # Let's use the "Smooth Minimum Distance" between convex sets via Support Functions?
    # Too complex.
    
    # Practical approach: 
    # Calculate the minimum distance between every vertex of v1 and every edge of v2, and vice versa.
    # Take the minimum of all these distances.
    # Approximate this minimum with a soft-min.
    
    dists = []
    for i in range(6):
        p1 = v1[i]
        for j in range(6):
            p2_start = v2[j]
            p2_end = v2[(j+1)%6]
            # Distance from point p1 to segment p2_start-p2_end
            # d = min(t in [0,1] | |p1 - (p2_start + t*(p2_end-p2_start))| )
            
            v_vec = p2_end - p2_start
            w = p1 - p2_start
            vv = np.dot(v_vec, v_vec)
            if vv < 1e-12:
                dist = np.linalg.norm(w)
            else:
                t = np.dot(w, v_vec) / vv
                t = np.clip(t, 0, 1)
                proj = p2_start + t * v_vec
                dist = np.linalg.norm(p1 - proj)
            dists.append(dist)
            
            # Also distance from p2_start to segment p1-v1[i+1]? 
            # Symmetric check is needed. 
            # Actually, min distance between two convex polygons is the min of:
            # 1. Vertex of A to Edge of B
            # 2. Vertex of B to Edge of A
            # So we need to do both.
            
            p2 = v2[j]
            p1_start = v1[i]
            p1_end = v1[(i+1)%6]
            v_vec2 = p1_end - p1_start
            w2 = p2 - p1_start
            vv2 = np.dot(v_vec2, v_vec2)
            if vv2 < 1e-12:
                dist2 = np.linalg.norm(w2)
            else:
                t2 = np.dot(w2, v_vec2) / vv2
                t2 = np.clip(t2, 0, 1)
                proj2 = p1_start + t2 * v_vec2
                dist2 = np.linalg.norm(p2 - proj2)
            dists.append(dist2)
            
    dists = np.array(dists)
    
    # Soft min
    m = np.min(dists)
    # If m is large, soft min ~ m. If m is small (overlap), soft min ~ m.
    # We want a penalty that is 0 if m > 0 and large if m < 0.
    # Penalty = max(0, -m)^2
    # Smooth max(0, x) = 0.5 * (x + sqrt(x^2 + eps))
    # So Smooth max(0, -m) = 0.5 * (-m + sqrt(m^2 + eps))
    eps = 1e-6
    smooth_neg_min = 0.5 * (-m + np.sqrt(m*m + eps))
    
    return smooth_neg_min ** 2

def _objective(params, outer_center, outer_angle):
    """
    Objective function: Side length + Overlap Penalties + Containment Penalties.
    params: flat array of [x1, y1, a1, x2, y2, a2, ...]
    """
    positions = np.array(params).reshape(N, 3)
    # Unwrap angles for smoothness? No, sin/cos are periodic.
    # But _hex_vertices uses radians.
    
    # 1. Calculate Side Length
    side = _min_enclosing_side_smooth(positions, outer_center, outer_angle)
    
    # 2. Calculate Overlap Penalties
    penalty = 0.0
    verts_list = []
    for i in range(N):
        x, y, a = positions[i]
        verts = _hex_vertices(x, y, 1.0, a)
        verts_list.append(verts)
        
    for i in range(N):
        for j in range(i+1, N):
            # If centers are far apart, skip detailed check to save time?
            # c1 = positions[i][:2], c2 = positions[j][:2]
            # if np.linalg.norm(c1-c2) > 4.0: continue
            
            d_pen = _min_distance_hexagons_smooth(verts_list[i], verts_list[j])
            penalty += d_pen
    
    # 3. Calculate Containment Penalties
    # For each vertex of each inner hex, check if it is inside the outer hex.
    # If not, add penalty proportional to the distance outside.
    # Outer hex is defined by center, side, angle.
    # We can reuse the support function logic.
    # A point p is inside if p . n_k <= s * sqrt(3)/2 for all k.
    # Violation = max(0, p . n_k - s * sqrt(3)/2)
    
    cx, cy = outer_center
    outer_angle_rad = np.radians(outer_angle)
    face_normals = []
    for k in range(6):
        a = np.radians(30 + k * 60)
        face_normals.append([np.cos(a), np.sin(a)])
    face_normals = np.array(face_normals)
    ca = np.cos(outer_angle_rad)
    sa = np.sin(outer_angle_rad)
    rot = np.array([[ca, -sa], [sa, ca]])
    face_normals = face_normals @ rot.T
    
    inradius_required = side * np.sqrt(3) / 2
    
    for i in range(N):
        verts = verts_list[i]
        verts_rel = verts - np.array(outer_center)
        projs = verts_rel @ face_normals.T
        # projs shape (6, 6)
        # max projection for each vertex
        max_projs = np.max(projs, axis=1)
        violations = np.maximum(0, max_projs - inradius_required)
        penalty += np.sum(violations ** 2)
        
    return side + 100.0 * penalty

def _gradient(params, outer_center, outer_angle):
    """Numerical gradient calculation for L-BFGS-B."""
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = _objective(params, outer_center, outer_angle)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = _objective(params_plus, outer_center, outer_angle)
        grad[i] = (f_plus - f0) / eps
    return grad

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """Strict validity check for final result."""
    cx, cy = outer_center
    
    # Check containment
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    # We need to check if inner verts are inside outer hex.
    # Use point in polygon or support functions.
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            # Check if v is inside outer hex
            # Rotate v into outer hex frame
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(outer_angle)
            ca = np.cos(-angle_rad)
            sa = np.sin(-angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            inradius = outer_side * np.sqrt(3) / 2
            inside = True
            for k in range(6):
                a = np.radians(30 + k * 60)
                nx = np.cos(a)
                ny = np.sin(a)
                if lx * nx + ly * ny > inradius + tol:
                    inside = False
                    break
            if not inside:
                return False
                
    # Check disjointness
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            # SAT check
            v1 = all_verts[i]
            v2 = all_verts[j]
            # Get normals
            n1 = []
            n2 = []
            for k in range(6):
                e1 = v1[(k+1)%6] - v1[k]
                n1.append([-e1[1], e1[0]])
                e2 = v2[(k+1)%6] - v2[k]
                n2.append([-e2[1], e2[0]])
            
            all_normals = n1 + n2
            disjoint = False
            for n in all_normals:
                norm = np.linalg.norm(n)
                if norm < 1e-9: continue
                n = n / norm
                proj1 = v1 @ n
                proj2 = v2 @ n
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    disjoint = True
                    break
            if not disjoint:
                return False
                
    return True

def _initial_lattice():
    """Create initial hexagonal lattice arrangement."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3) 
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions

def _initial_random():
    """Create a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 3.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def construct():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    outer_angles = [0.0, 30.0, 15.0]
    outer_center = (0.0, 0.0)
    
    for oa in outer_angles:
        if _remaining_time() < 30.0:
            break
            
        initial_configs = [
            _initial_lattice(),
            _initial_random(),
            _initial_random(),
            _initial_random()
        ]
        
        for init_pos in initial_configs:
            if _remaining_time() < 10.0:
                break
                
            # Flatten positions
            x0 = np.array(init_pos).flatten()
            
            # Run Optimization
            # Use L-BFGS-B with numerical gradient
            # Bounds: x, y in [-5, 5], angle in [0, 360]
            bounds = []
            for i in range(N):
                bounds.append((-5.0, 5.0))
                bounds.append((-5.0, 5.0))
                bounds.append((0.0, 360.0))
                
            try:
                result = minimize(
                    _objective, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
                )
            except Exception:
                continue
                
            if result.success or result.fun < best_score:
                # Reconstruct positions
                pos_opt = result.x.reshape(N, 3)
                # Calculate the actual side length (not the penalized one)
                side = _min_enclosing_side_smooth(pos_opt, outer_center, oa)
                
                # Verify validity
                if _check_validity(pos_opt, outer_center, side, oa, tol=1e-4):
                    if side < best_score:
                        best_score = side
                        best_result = (pos_opt.tolist(), outer_center, side, oa)
                        
    if best_result is not None:
        return best_result
    
    # Fallback: known valid configuration (loose)
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 3.5
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = 6.0
    for k in range(4):
        a = np.radians(30 + 60 * k)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    side = _min_enclosing_side_smooth(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0

if __name__ == "__main__":
    res = construct()
    print(res)
