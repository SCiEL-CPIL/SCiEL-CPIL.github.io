import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)
TWO_PI = 2 * math.pi
TWO_THIRD_PI = 2 * math.pi / 3
ONE_THIRD_PI = math.pi / 3

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def get_hex_verts_rad(cx, cy, side, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def outer_normals(oa):
        """
        Returns 6 unit outward normals for a regular hexagon with rotation oa (radians).
        For a pointy-top hexagon (vertices at 0, 60, ...), the edge normals are at 30, 90, 150, ...
        So normal k is at angle: oa + pi/6 + k * pi/3
        """
        normals = []
        for k in range(6):
            ang_n = oa + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        return np.array(normals)

    def enclosing_hexagon_side(pos, ang, oa, oc):
        """
        Compute the minimum side length of an outer hexagon (center oc, angle oa)
        that contains all inner hexagons defined by pos (N,2) and ang (N,) radians.
        """
        normals = outer_normals(oa)
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts_rad(pos[i][0], pos[i][1], 1.0, ang[i])
            verts_rel = verts - np.array(oc)
            projs = verts_rel @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        # Distance from center to edge is apothem = side * sqrt(3)/2
        # So side = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / SQRT3

    def sat_overlap_depth(p1, p2):
        """
        Returns the minimum separation between two polygons p1 and p2.
        Positive if separated, negative if overlapping (overlap depth).
        """
        # Quick bounding box check
        min_x1, min_y1 = np.min(p1, axis=0)
        max_x1, max_y1 = np.max(p1, axis=0)
        min_x2, min_y2 = np.min(p2, axis=0)
        max_x2, max_y2 = np.max(p2, axis=0)
        if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
            return float('inf')
        
        min_sep = float('inf')
        axes = []
        for k in range(6):
            edge = p1[(k+1)%6] - p1[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9: axes.append(n/norm)
        for k in range(6):
            edge = p2[(k+1)%6] - p2[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9: axes.append(n/norm)
        
        for axis in axes:
            proj1 = np.dot(p1, axis)
            proj2 = np.dot(p2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            sep = max(min1, min2) - min(max1, max2)
            if sep < min_sep:
                min_sep = sep
        return min_sep

    def cluster_objective(params):
        """
        Objective for Phase 1: Minimize the radius of the enclosing circle of the cluster.
        This encourages a compact arrangement.
        params: [x0..x11, y0..y11, a0..a11]
        """
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:]
        
        # Compute enclosing circle radius
        # Approximate by max distance from centroid
        centroid = np.mean(pos, axis=0)
        max_dist = 0.0
        for i in range(N):
            d = np.linalg.norm(pos[i] - centroid)
            if d > max_dist:
                max_dist = d
        
        # Add penalty for overlaps to ensure feasibility
        penalty = 0.0
        polys = [get_hex_verts_rad(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                sep = sat_overlap_depth(polys[i], polys[j])
                if sep < 0:
                    penalty += (-sep) ** 2 * 50.0
        
        return max_dist + penalty

    def phase1_optimize(x0):
        """
        Optimize the cluster configuration to be compact and non-overlapping.
        Uses SLSQP with a simple overlap penalty (since SLSQP constraints are hard, 
        and defining 66 SAT constraints is complex, we use a heavy penalty in the objective 
        and verify later, or we can define simple distance constraints).
        Actually, let's use Nelder-Mead for robustness in Phase 1, then SLSQP for Phase 2.
        """
        res = minimize(
            cluster_objective, 
            x0, 
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        return res.x

    def identify_active_vertices(pos, ang, oa, oc):
        """
        Identify which vertices of the inner hexagons are "active" (i.e., on the boundary 
        of the minimum enclosing hexagon) for the given outer configuration.
        Returns a list of tuples: (inner_idx, vertex_idx)
        """
        normals = outer_normals(oa)
        max_proj = -float('inf')
        # First pass: find max projection
        for i in range(N):
            verts = get_hex_verts_rad(pos[i][0], pos[i][1], 1.0, ang[i])
            verts_rel = verts - np.array(oc)
            projs = verts_rel @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        active = []
        tol = 1e-6
        for i in range(N):
            verts = get_hex_verts_rad(pos[i][0], pos[i][1], 1.0, ang[i])
            verts_rel = verts - np.array(oc)
            projs = verts_rel @ normals.T
            for v in range(6):
                for k in range(6):
                    if abs(projs[v, k] - max_proj) < tol:
                        active.append((i, v))
        
        # If no active vertices found (should not happen), return all
        if not active:
            return [(i, v) for i in range(N) for v in range(6)]
            
        return active

    def phase2_optimize(pos, ang, oa, oc, active_verts):
        """
        Optimize the outer hexagon (center, angle) and potentially adjust inner hexagons slightly
        to minimize the side length, subject to the active vertices touching the corresponding edges.
        
        We will optimize only the outer hexagon parameters (oc_x, oc_y, oa) here,
        keeping inner hexagons fixed, because the inner arrangement is already compact.
        Then, we can do a joint optimization if needed.
        
        Let's define the objective as the maximum projection of the active vertices.
        side = max_proj * 2 / sqrt(3)
        We want to minimize max_proj.
        
        Variables: [oc_x, oc_y, oa]
        """
        def outer_obj(outer_params):
            oc_x, oc_y, oa_val = outer_params
            oc = np.array([oc_x, oc_y])
            normals = outer_normals(oa_val)
            
            max_proj = -float('inf')
            for i, v in active_verts:
                # Get vertex v of inner hexagon i
                # We need to compute the vertex position
                verts = get_hex_verts_rad(pos[i][0], pos[i][1], 1.0, ang[i])
                vert = verts[v]
                proj = np.dot(vert - oc, normals.T)
                # The projection onto the normal of the edge that this vertex is supposed to touch
                # Wait, the vertex might not touch the same edge if the outer hexagon rotates.
                # The "active" set is defined for the current oa.
                # In the optimization, as oa changes, the active edges might change.
                # This is the non-smoothness.
                # However, if we assume the active set is constant locally, we can minimize the max projection.
                # Let's just compute the max projection over ALL vertices of ALL hexagons for safety,
                # but weight the active ones? No, just use the global max.
                # Actually, if we use the global max, we are back to the original problem.
                # The key is that the gradient is well-defined if we pick the specific active constraints.
                # Let's try to minimize the maximum projection of the *active* vertices only.
                # This is valid if the active set remains the binding constraint.
                # If another vertex becomes more prominent, the objective will increase, 
                # but the gradient might be wrong.
                # For now, let's use the max over active vertices.
                if proj > max_proj:
                    max_proj = proj
            return max_proj * 2.0 / SQRT3

        x0_outer = np.array([oc[0], oc[1], oa])
        
        try:
            res_outer = minimize(
                outer_obj, 
                x0_outer, 
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res_outer.success:
                return res_outer.x
        except:
            pass
        return x0_outer

    def full_joint_optimize(pos, ang, oa, oc, active_verts):
        """
        Jointly optimize inner positions/angles and outer hexagon parameters.
        This is the most powerful step.
        Variables: [x0..x11, y0..y11, a0..a11, oc_x, oc_y, oa]
        Objective: side length (max projection)
        Constraints: Non-overlap (SAT >= 0)
        """
        def joint_obj(params):
            p_pos = params[:2*N].reshape(N, 2)
            p_ang = params[2*N:3*N]
            oc_x, oc_y, oa_val = params[3*N:3*N+3]
            oc = np.array([oc_x, oc_y])
            
            normals = outer_normals(oa_val)
            max_proj = -float('inf')
            
            # Compute max projection over active vertices (for smoothness)
            # But we must ensure no other vertex exceeds this.
            # So we take the max over ALL vertices.
            # This makes the objective non-smooth again.
            # To handle this, we can use the "active set" trick:
            # Only consider the vertices in active_verts for the objective,
            # but add a penalty if any other vertex exceeds the current max_proj.
            
            # Let's stick to the global max for correctness, and rely on SLSQP's
            # ability to handle piecewise linear functions (it uses gradients, 
            # so it might struggle at kinks).
            # Alternative: Use the max of the active set, and add a large penalty
            # for any violation of "active vertex projection >= other vertex projection".
            
            # Let's try the simple global max first.
            for i in range(N):
                verts = get_hex_verts_rad(p_pos[i][0], p_pos[i][1], 1.0, p_ang[i])
                verts_rel = verts - oc
                projs = verts_rel @ normals.T
                max_proj_i = np.max(projs)
                if max_proj_i > max_proj:
                    max_proj = max_proj_i
            
            side = max_proj * 2.0 / SQRT3
            
            # Overlap penalty
            penalty = 0.0
            polys = [get_hex_verts_rad(p_pos[i][0], p_pos[i][1], 1.0, p_ang[i]) for i in range(N)]
            for i in range(N):
                for j in range(i+1, N):
                    sep = sat_overlap_depth(polys[i], polys[j])
                    if sep < 0:
                        penalty += (-sep) ** 2 * 100.0
            
            return side + penalty

        x0 = np.concatenate([pos.flatten(), ang, [oc[0], oc[1], oa]])
        
        try:
            res = minimize(
                joint_obj, 
                x0, 
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if res.success:
                return res.x
        except:
            pass
        return x0

    # Initial configurations
    inits = []
    
    # 1. Honeycomb lattice (standard)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * SQRT3 / 2.0 + j * SQRT3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_ang = np.zeros(N)
    inits.append(np.concatenate([init_pos.flatten(), init_ang]))
    
    # 2. Ring configuration
    ring_pos = []
    ring_ang = []
    r = 2.0
    for i in range(N):
        a = 2 * math.pi * i / N
        ring_pos.append([r * math.cos(a), r * math.sin(a)])
        ring_ang.append(a)
    ring_pos = np.array(ring_pos)
    ring_pos -= np.mean(ring_pos, axis=0)
    inits.append(np.concatenate([ring_pos.flatten(), np.array(ring_ang)]))
    
    # 3. Random starts
    np.random.seed(42)
    for _ in range(5):
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(0, TWO_PI, N)
        inits.append(np.concatenate([pos.flatten(), ang]))

    best_side = float('inf')
    best_params = None
    best_outer_center = [0.0, 0.0]
    best_outer_angle = 0.0

    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        # Phase 1: Compact the cluster
        try:
            x1 = phase1_optimize(x0)
        except:
            x1 = x0
        
        pos1 = x1[:2*N].reshape(N, 2)
        ang1 = x1[2*N:]
        
        # Check if Phase 1 result is valid (no overlaps)
        polys1 = [get_hex_verts_rad(pos1[i][0], pos1[i][1], 1.0, ang1[i]) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                sep = sat_overlap_depth(polys1[i], polys1[j])
                if sep < -1e-5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Try to repair with a few steps of local search
            # Or just skip this init
            continue

        # Phase 2: Optimize outer hexagon
        # Start with outer hexagon centered at centroid, angle 0
        centroid = np.mean(pos1, axis=0)
        oc0 = centroid
        oa0 = 0.0
        
        # Identify active vertices
        active_verts = identify_active_vertices(pos1, ang1, oa0, oc0)
        
        # Optimize outer hexagon only
        outer_params = phase2_optimize(pos1, ang1, oa0, oc0, active_verts)
        oc1 = np.array(outer_params[:2])
        oa1 = outer_params[2]
        
        side1 = enclosing_hexagon_side(pos1, ang1, oa1, oc1)
        
        # Phase 3: Joint optimization
        x2 = full_joint_optimize(pos1, ang1, oa1, oc1, active_verts)
        
        pos2 = x2[:2*N].reshape(N, 2)
        ang2 = x2[2*N:3*N]
        oc2 = np.array(x2[3*N:3*N+2])
        oa2 = x2[3*N+2]
        
        side2 = enclosing_hexagon_side(pos2, ang2, oa2, oc2)
        
        # Check validity of joint result
        polys2 = [get_hex_verts_rad(pos2[i][0], pos2[i][1], 1.0, ang2[i]) for i in range(N)]
        valid2 = True
        for i in range(N):
            for j in range(i+1, N):
                sep = sat_overlap_depth(polys2[i], polys2[j])
                if sep < -1e-5:
                    valid2 = False
                    break
            if not valid2:
                break
        
        if valid2 and side2 < best_side:
            best_side = side2
            best_params = x2.copy()
            best_outer_center = [oc2[0], oc2[1]]
            best_outer_angle = np.degrees(oa2)
        
        # Also check Phase 2 result if it's better and valid
        if side1 < best_side:
            # Re-check validity for Phase 2 (should be same as Phase 1)
            if valid:
                best_side = side1
                # Construct params for Phase 2 result
                x_temp = np.concatenate([pos1.flatten(), ang1, [oc1[0], oc1[1], oa1]])
                best_params = x_temp
                best_outer_center = [oc1[0], oc1[1]]
                best_outer_angle = np.degrees(oa1)

    # Fallback
    if best_params is None:
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        best_params = np.concatenate([pos.flatten(), ang, [0,0,0]])
        best_side = enclosing_hexagon_side(pos, ang, 0.0, np.array([0,0]))
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0

    # Construct return values
    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:3*N]
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(final_ang[i])])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle
