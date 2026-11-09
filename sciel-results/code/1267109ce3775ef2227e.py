<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hexagon_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        x = center[0] + side * np.cos(theta)
        y = center[1] + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _hex_edge_normals(angle_deg):
    """Return unit normals for the 6 edges of a hexagon with given rotation."""
    angle = np.radians(angle_deg)
    normals = []
    for i in range(6):
        # Edge i is between vertices i and i+1
        # The outward normal points at angle + 30 + i*60 degrees (for flat-top)
        # For pointy-top (vertex at 0), edge between vertex i and i+1 has normal at angle + 30 + i*60
        theta = angle + np.pi / 6 + np.pi * i / 3.0
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _hex_inradius(side):
    """Inradius (apothem) of regular hexagon with side length."""
    return side * np.sqrt(3) / 2.0

def _min_distance_to_hex_boundary(px, py, center, side, angle_deg):
    """Minimum signed distance from point to hexagon boundary (positive inside, negative outside)."""
    cx, cy = center
    dx = px - cx
    dy = py - cy
    # Transform to local frame
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # For a regular hexagon with circumradius R centered at origin, vertex at angle 0:
    # The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
    # Each edge is at distance inradius = R*sqrt(3)/2 from center
    r = _hex_inradius(side)
    
    # Compute signed distance to each edge (positive = inside)
    min_dist = float('inf')
    for i in range(6):
        theta = np.pi / 6 + np.pi * i / 3.0
        nx, ny = np.cos(theta), np.sin(theta)
        # Signed distance from point to this edge line
        dist = lx * nx + ly * ny - r
        # For the edge between vertices, the constraint is dist >= 0 for inside
        # But we need to handle the corners properly
        min_dist = min(min_dist, dist)
    
    # For points near corners, the above gives correct signed distance for convex polygon
    # The minimum over all half-plane distances gives the signed distance for convex polygons
    return min_dist

def _hexagon_overlap_distance(v1_center, v1_angle, v2_center, v2_angle, side=1.0):
    """Return negative overlap depth if two hexagons overlap, 0 if touching, positive if separated.
    Uses SAT (Separating Axis Theorem)."""
    verts1 = _hexagon_vertices(v1_center, side, v1_angle)
    verts2 = _hexagon_vertices(v2_center, side, v2_angle)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            norm = np.sqrt(ex**2 + ey**2)
            if norm > 1e-12:
                axes.append((-ey / norm, ex / norm))
        return axes
    
    axes = get_axes(verts1) + get_axes(verts2)
    min_overlap = float('inf')
    
    for ax, ay in axes:
        proj1 = np.array([vx * ax + vy * ay for vx, vy in verts1])
        proj2 = np.array([vx * ax + vy * ay for vx, vy in verts2])
        max1, min1 = proj1.max(), proj1.min()
        max2, min2 = proj2.max(), proj2.min()
        
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return 0.0  # Separated, no overlap
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return -min_overlap  # Negative means overlap depth

def _compute_objective(centers, angles, outer_center, outer_side, outer_angle):
    """Compute objective: minimize outer_side subject to constraints.
    Returns (objective_value, valid, min_clearance, max_overlap)."""
    min_clearance = float('inf')
    max_overlap = -float('inf')
    
    # Check inner hexagons inside outer
    for i in range(N):
        cx, cy = centers[i]
        # Check all vertices of inner hexagon
        inner_verts = _hexagon_vertices((cx, cy), 1.0, angles[i])
        for vx, vy in inner_verts:
            dist = _min_distance_to_hex_boundary(vx, vy, outer_center, outer_side, outer_angle)
            min_clearance = min(min_clearance, dist)
    
    # Check pairwise overlaps
    for i in range(N):
        for j in range(i + 1, N):
            od = _hexagon_overlap_distance(centers[i], angles[i], centers[j], angles[j])
            max_overlap = max(max_overlap, od)
    
    valid = (min_clearance >= -1e-7) and (max_overlap <= 1e-7)
    
    # Objective: outer_side + penalty for constraint violations
    penalty = 0.0
    if min_clearance < 0:
        penalty += 10.0 * (-min_clearance) ** 2
    if max_overlap > 0:
        penalty += 10.0 * max_overlap ** 2
    
    return outer_side + penalty, valid, min_clearance, max_overlap

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    def make_hex_packing_config():
        """Create a hexagonal close-packed arrangement of 11 unit hexagons.
        Center + 6 in first ring + 4 in second ring."""
        centers = np.zeros((N, 2))
        angles = np.zeros(N)
        
        # Center hexagon
        centers[0] = [0.0, 0.0]
        
        # First ring: 6 hexagons
        r1 = np.sqrt(3)  # Distance between centers of adjacent hexagons
        for i in range(6):
            angle = np.pi / 3 * i
            centers[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        
        # Second ring: 4 more hexagons (positions 7, 8, 9, 10)
        r2 = 2 * np.sqrt(3)
        # Place them at angles 30, 90, 150, 210 degrees (offset by 30 from first ring)
        second_angles = [np.pi / 6 + np.pi / 3 * i for i in range(4)]
        for i in range(4):
            centers[7 + i] = [r2 * np.cos(second_angles[i]), r2 * np.sin(second_angles[i])]
        
        return centers, angles
    
    def optimize_from(centers0, angles0, outer_center0, outer_side0, outer_angle0):
        """Local optimization using gradient descent with penalty method."""
        centers = centers0.copy().astype(float)
        angles = angles0.copy().astype(float)
        outer_center = np.array(outer_center0, dtype=float)
        outer_side = float(outer_side0)
        outer_angle = float(outer_angle0)
        
        best_local_obj = float('inf')
        best_local_state = None
        
        lr = 0.05
        patience = 0
        max_patience = 50
        
        for step in range(3000):
            if time.time() > DEADLINE:
                break
            
            # Compute current objective
            obj, valid, min_clear, max_ov = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
            
            if valid and obj < best_local_obj:
                best_local_obj = obj
                best_local_state = (centers.copy(), angles.copy(), outer_center.copy(), outer_side, outer_angle)
                patience = 0
            else:
                patience += 1
            
            if patience > max_patience:
                break
            
            # Numerical gradient computation
            eps = 1e-4
            grad_c = np.zeros_like(centers)
            grad_a = np.zeros_like(angles)
            grad_oc = np.zeros(2)
            grad_s = 0.0
            grad_oa = 0.0
            
            # Gradient w.r.t. centers
            for i in range(N):
                for d in range(2):
                    orig = centers[i, d]
                    centers[i, d] = orig + eps
                    obj_plus, _, _, _ = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
                    centers[i, d] = orig - eps
                    obj_minus, _, _, _ = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
                    centers[i, d] = orig
                    grad_c[i, d] = (obj_plus - obj_minus) / (2 * eps)
            
            # Gradient w.r.t. angles
            for i in range(N):
                orig = angles[i]
                angles[i] = orig + eps
                obj_plus, _, _, _ = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
                angles[i] = orig - eps
                obj_minus, _, _, _ = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
                angles[i] = orig
                grad_a[i] = (obj_plus - obj_minus) / (2 * eps)
            
            # Gradient w.r.t. outer center
            for d in range(2):
                orig = outer_center[d]
                outer_center[d] = orig + eps
                obj_plus, _, _, _ = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
                outer_center[d] = orig - eps
                obj_minus, _, _, _ = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
                outer_center[d] = orig
                grad_oc[d] = (obj_plus - obj_minus) / (2 * eps)
            
            # Gradient w.r.t. outer side
            orig_s = outer_side
            outer_side = orig_s + eps
            obj_plus, _, _, _ = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
            outer_side = orig_s - eps
            obj_minus, _, _, _ = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
            outer_side = orig_s
            grad_s = (obj_plus - obj_minus) / (2 * eps)
            
            # Gradient w.r.t. outer angle
            orig_oa = outer_angle
            outer_angle = orig_oa + eps
            obj_plus, _, _, _ = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
            outer_angle = orig_oa - eps
            obj_minus, _, _, _ = _compute_objective(centers, angles, outer_center, outer_side, outer_angle)
            outer_angle = orig_oa
            grad_oa = (obj_plus - obj_minus) / (2 * eps)
            
            # Gradient descent step
            centers -= lr * grad_c
            angles -= lr * grad_a
            outer_center -= lr * grad_oc
            outer_side = max(2.0, outer_side - lr * grad_s)
            outer_angle -= lr * grad_oa
            
            # Adaptive learning rate
            if obj < best_local_obj:
                lr = min(lr * 1.05, 0.2)
            else:
                lr *= 0.95
            
            if lr < 1e-6:
                break
        
        return best_local_obj, best_local_state
    
    # Try multiple initial configurations
    rng = np.random.RandomState(42)
    
    # Configuration 1: Hexagonal packing
    centers1, angles1 = make_hex_packing_config()
    obj1, state1 = optimize_from(centers1, angles1, (0.0, 0.0), 5.0, 0.0)
    if state1 is not None and obj1 < best_score:
        best_score = obj1
        best_result = state1
    
    if time.time() > DEADLINE:
        pass
    
    # Configuration 2: Perturbed hexagonal packing
    centers2, angles2 = make_hex_packing_config()
    centers2 += rng.normal(0, 0.1, (N, 2))
    angles2 += rng.normal(0, 5.0, N)
    obj2, state2 = optimize_from(centers2, angles2, (0.0, 0.0), 5.0, 0.0)
    if state2 is not None and obj2 < best_score:
        best_score = obj2
        best_result = state2
    
    if time.time() > DEADLINE:
        pass
    
    # Configuration 3: Different rotation for outer hexagon
    centers3, angles3 = make_hex_packing_config()
    obj3, state3 = optimize_from(centers3, angles3, (0.0, 0.0), 5.0, 30.0)
    if state3 is not None and obj3 < best_score:
        best_score = obj3
        best_result = state3
    
    if time.time() > DEADLINE:
        pass
    
    # Configuration 4: Random restart
    centers4 = rng.uniform(-3, 3, (N, 2))
    angles4 = rng.uniform(-180, 180, N)
    obj4, state4 = optimize_from(centers4, angles4, (0.0, 0.0), 6.0, 0.0)
    if state4 is not None and obj4 < best_score:
        best_score = obj4
        best_result = state4
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    centers, angles, outer_center, outer_side, outer_angle = best_result
    inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
=======
"""Improved program: force-directed relaxation + outer hexagon optimization."""
import numpy as np
import time
from itertools import combinations

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    return np.array([
        center[0] + side * np.cos(angle + np.pi * i / 3.0),
        center[1] + side * np.sin(angle + np.pi * i / 3.0)
    ] for i in range(6)).reshape(6, 2)

def _hex_edge_normals(angle_deg):
    """Return unit outward normals for the 6 edges."""
    angle = np.radians(angle_deg)
    return np.array([
        np.cos(angle + np.pi / 6 + np.pi * i / 3.0),
        np.sin(angle + np.pi / 6 + np.pi * i / 3.0)
    ] for i in range(6)).reshape(6, 2)

def _min_enclosing_side_lp(centers, angles, outer_center, outer_angle):
    """Find minimum outer side length such that all inner hexagons fit inside.
    Uses LP-like approach: for each inner hex vertex, compute required outer side."""
    outer_normals = _hex_edge_normals(outer_angle)
    outer_inradius = None  # will compute
    
    # For each inner hexagon vertex, project onto each outer edge normal
    # The constraint is: dot(vertex - outer_center, normal) <= inradius
    # => inradius >= dot(vertex - outer_center, normal)
    # => side >= 2 * max_required_inradius / sqrt(3)
    
    max_required_inradius = 0.0
    for i in range(N):
        verts = _hex_vertices(centers[i], 1.0, angles[i])
        for vx, vy in verts:
            dx = vx - outer_center[0]
            dy = vy - outer_center[1]
            for nx, ny in outer_normals:
                dist = dx * nx + dy * ny
                max_required_inradius = max(max_required_inradius, dist)
    
    if max_required_inradius <= 0:
        return 0.0
    
    # inradius = side * sqrt(3) / 2 => side = 2 * inradius / sqrt(3)
    return 2.0 * max_required_inradius / np.sqrt(3.0)

def _hex_overlap_depth(c1, a1, c2, a2, side=1.0):
    """Return positive overlap depth if two hexagons overlap, 0 if separated/touching."""
    v1 = _hex_vertices(c1, side, a1)
    v2 = _hex_vertices(c2, side, a2)
    
    # Get all edge normals from both hexagons
    axes = []
    for verts in (v1, v2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm > 1e-12:
                normal = np.array([-edge[1], edge[0]]) / norm
                axes.append(normal)
    
    min_overlap = float('inf')
    for ax in axes:
        proj1 = v1 @ ax
        proj2 = v2 @ ax
        max1, min1 = proj1.max(), proj1.min()
        max2, min2 = proj2.max(), proj2.min()
        
        if max1 <= min2 + 1e-9 or max2 <= min1 + 1e-9:
            return 0.0  # Separated
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    return max(0.0, min_overlap)

def _check_validity(centers, angles, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: no overlaps, all inside outer."""
    # Check pairwise overlaps
    for i, j in combinations(range(N), 2):
        ov = _hex_overlap_depth(centers[i], angles[i], centers[j], angles[j])
        if ov > tol:
            return False
    
    # Check all inner hexagons inside outer
    outer_normals = _hex_edge_normals(outer_angle)
    outer_inradius = outer_side * np.sqrt(3) / 2.0
    
    for i in range(N):
        verts = _hex_vertices(centers[i], 1.0, angles[i])
        for vx, vy in verts:
            dx = vx - outer_center[0]
            dy = vy - outer_center[1]
            for nx, ny in outer_normals:
                if dx * nx + dy * ny > outer_inradius + tol:
                    return False
    
    return True

def _optimize_outer(centers, angles, outer_angle):
    """Optimize outer hexagon center and side for given inner configuration."""
    best_side = float('inf')
    best_center = None
    
    # Try multiple outer center positions
    # First: centroid of inner hexagons
    centroid = np.mean(centers, axis=0)
    
    # Try a grid of outer centers around centroid
    for dx in np.linspace(-2, 2, 9):
        for dy in np.linspace(-2, 2, 9):
            oc = centroid + np.array([dx, dy])
            side = _min_enclosing_side_lp(centers, angles, oc, outer_angle)
            if side < best_side:
                best_side = side
                best_center = oc
    
    # Refine with local search
    for _ in range(50):
        improved = False
        for dx in np.linspace(-0.5, 0.5, 5):
            for dy in np.linspace(-0.5, 0.5, 5):
                oc = best_center + np.array([dx, dy])
                side = _min_enclosing_side_lp(centers, angles, oc, outer_angle)
                if side < best_side - 1e-6:
                    best_side = side
                    best_center = oc
                    improved = True
        if not improved:
            break
    
    return best_side, best_center

def _force_directed_relax(centers0, angles0, max_steps=5000, tol=1e-6):
    """Force-directed relaxation to find a valid, dense packing."""
    centers = centers0.copy().astype(float)
    angles = angles0.copy().astype(float)
    
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_score = float('inf')
    
    lr = 0.1
    cooling = 0.999
    
    for step in range(max_steps):
        if time.time() > DEADLINE:
            break
        
        # Compute forces
        forces = np.zeros_like(centers)
        torques = np.zeros_like(angles)
        
        # Repulsion between overlapping hexagons
        for i, j in combinations(range(N), 2):
            c1, a1 = centers[i], angles[i]
            c2, a2 = centers[j], angles[j]
            
            overlap = _hex_overlap_depth(c1, a1, c2, a2)
            if overlap > 0:
                # Push apart along center-to-center direction
                d = c2 - c1
                dist = np.linalg.norm(d)
                if dist > 1e-6:
                    d = d / dist
                else:
                    d = np.array([1.0, 0.0])
                
                # Force proportional to overlap
                f = 2.0 * overlap
                forces[i] -= f * d
                forces[j] += f * d
                
                # Torque: try to rotate to reduce overlap
                # Simple heuristic: rotate to align edges
                angle_diff = a2 - a1
                # Normalize to [-180, 180]
                angle_diff = ((angle_diff + 180) % 360) - 180
                torque = 0.1 * overlap * angle_diff
                torques[i] += torque
                torques[j] -= torque
        
        # Attraction toward center (to keep packing compact)
        for i in range(N):
            d_to_center = -centers[i]
            dist = np.linalg.norm(d_to_center)
            if dist > 1e-6:
                forces[i] += 0.01 * d_to_center / dist
        
        # Apply forces with velocity damping
        centers += lr * forces
        angles += lr * torques
        
        # Cool down
        lr *= cooling
        
        # Evaluate current configuration
        outer_side, outer_center = _optimize_outer(centers, angles, 0.0)
        score = outer_side
        
        # Check validity
        if _check_validity(centers, angles, outer_center, outer_side, 0.0):
            if score < best_score:
                best_score = score
                best_centers = centers.copy()
                best_angles = angles.copy()
        
        # Early stopping if converged
        if lr < 1e-6:
            break
    
    return best_score, best_centers, best_angles

def _get_valid_fallback():
    """Return a valid fallback configuration."""
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Hexagonal packing
    centers[0] = [0.0, 0.0]
    r1 = np.sqrt(3)
    for i in range(6):
        angle = np.pi / 3 * i
        centers[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    
    r2 = 2 * np.sqrt(3)
    second_angles = [np.pi / 6 + np.pi / 3 * i for i in range(4)]
    for i in range(4):
        centers[7 + i] = [r2 * np.cos(second_angles[i]), r2 * np.sin(second_angles[i])]
    
    outer_side, outer_center = _optimize_outer(centers, angles, 0.0)
    inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), 0.0

def _generate_initial_configurations(rng):
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Standard hexagonal packing
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    centers[0] = [0.0, 0.0]
    r1 = np.sqrt(3)
    for i in range(6):
        angle = np.pi / 3 * i
        centers[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    r2 = 2 * np.sqrt(3)
    second_angles = [np.pi / 6 + np.pi / 3 * i for i in range(4)]
    for i in range(4):
        centers[7 + i] = [r2 * np.cos(second_angles[i]), r2 * np.sin(second_angles[i])]
    configs.append((centers, angles))
    
    # 2. Perturbed hexagonal
    centers2 = centers + rng.normal(0, 0.2, (N, 2))
    angles2 = angles + rng.normal(0, 10.0, N)
    configs.append((centers2, angles2))
    
    # 3. Different rotation
    centers3 = centers.copy()
    angles3 = angles + 30.0
    configs.append((centers3, angles3))
    
    # 4. Random
    centers4 = rng.uniform(-3, 3, (N, 2))
    angles4 = rng.uniform(-180, 180, N)
    configs.append((centers4, angles4))
    
    # 5. Cluster
    centers5 = np.zeros((N, 2))
    for i in range(N):
        centers5[i] = rng.uniform(-2, 2, 2)
    angles5 = rng.uniform(-180, 180, N)
    configs.append((centers5, angles5))
    
    return configs

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    rng = np.random.RandomState(42)
    
    # Generate initial configurations
    configs = _generate_initial_configurations(rng)
    
    for idx, (centers0, angles0) in enumerate(configs):
        if time.time() > DEADLINE - 30:  # Leave time for final evaluation
            break
        
        # Force-directed relaxation
        score, centers, angles = _force_directed_relax(centers0, angles0)
        
        if np.isfinite(score) and score < best_score:
            # Optimize outer hexagon
            outer_side, outer_center = _optimize_outer(centers, angles, 0.0)
            
            # Verify validity
            if _check_validity(centers, angles, outer_center, outer_side, 0.0):
                best_score = outer_side
                inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
                best_result = (inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), 0.0)
    
    # Also try different outer angles for the best configuration
    if best_result is not None:
        inner, outer_center, outer_side, outer_angle = best_result
        centers = np.array([[x, y] for x, y, a in inner])
        angles = np.array([a for x, y, a in inner])
        
        for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
            if time.time() > DEADLINE - 10:
                break
            side, oc = _optimize_outer(centers, angles, oa)
            if side < best_score and _check_validity(centers, angles, oc, side, oa):
                best_score = side
                best_result = (inner, (float(oc[0]), float(oc[1])), float(side), float(oa))
    
    if best_result is None:
        return _get_valid_fallback()
    
    return best_result
>>>>>>> REPLACE
