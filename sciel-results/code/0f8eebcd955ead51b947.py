import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top)
    # Vertices at 0, 60, ..., 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Precompute normals for the 6 sides of a unit hexagon
    # Edge 0: (1,0) to (0.5, sqrt(3)/2). Normal points outward: (sqrt(3)/2, 0.5) -> 30 deg
    unit_normals_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle that contains all inner hexagons.
        Vectorized for speed.
        """
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides: 6 directions
        normal_angles = theta_rad + math.pi/6 + np.arange(6) * math.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # For each inner hex, compute max projection onto each normal
        # verts_i = R(angle_i) @ unit_verts_0 + pos_i
        # proj = verts_i @ normals.T  -> shape (6, 6) per hex
        # max over vertices for each normal -> shape (6,) per hex
        # Then max over all hexes for each normal -> shape (6,)
        # Final answer: max over normals of that
        
        max_proj_all = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang_deg = angles[i]
            ang_rad = math.radians(ang_deg)
            c, s = math.cos(ang_rad), math.sin(ang_rad)
            rot = np.array([[c, -s], [s, c]])
            verts = unit_verts_0 @ rot.T + np.array([cx, cy])
            projs = verts @ normals.T  # (6, 6)
            max_proj_i = np.max(projs, axis=0)  # (6,)
            m = np.max(max_proj_i)
            if m > max_proj_all:
                max_proj_all = m
        
        return max_proj_all * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value (approximate penetration depth).
        If they are separated, returns the positive distance.
        """
        min_gap = float('inf')
        
        # Axes from poly1 edges
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            norm = np.linalg.norm(normal1)
            if norm < 1e-9: continue
            axis = normal1 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap on this axis
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                # Overlapping on this axis
                min_gap = min(min_gap, gap)
            else:
                # Separated on this axis -> disjoint
                return gap
                
        # Axes from poly2 edges
        for i in range(6):
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            norm = np.linalg.norm(normal2)
            if norm < 1e-9: continue
            axis = normal2 / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
                
        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- PHASE 1: DISCRETE GRID SEARCH FOR TOPOLOGY ---
    # Instead of optimizing continuous coordinates from a random start, we search
    # a lattice of potential centers and a set of discrete angles to find a valid
    # configuration. This avoids getting stuck in local minima of the penalty method.
    
    sqrt3 = math.sqrt(3)
    
    # Generate candidate centers in a hexagonal lattice pattern
    # The ideal packing for hexagons is on a triangular lattice.
    # Lattice vectors: a1 = (1.5, sqrt(3)/2), a2 = (0, sqrt(3))
    # We search a range of offsets and scales.
    
    best_valid_config = None
    best_valid_score = float('inf')
    
    # Search space for lattice scaling and offset
    # A unit hexagon has width 2. To pack them, centers should be at least ~2 apart in some directions.
    # Let's try lattice spacings that allow tight packing.
    
    # Pre-computed discrete angles (multiples of 30 degrees are often optimal for hexagons)
    discrete_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    # We will search for configurations on a lattice defined by:
    # P_i = origin + i * v1 + j * v2
    # where v1, v2 are lattice vectors.
    
    # Common tight packings use specific lattice vectors.
    # For hexagons, the "standard" dense packing has centers at distance 2*sqrt(3)/2 * 2? 
    # Distance between centers in dense hex packing is 2 * (inradius) + 2 * (inradius) = 2 * sqrt(3) * 1 = 2*sqrt(3) ~ 3.46?
    # Wait, inradius of unit hex is sqrt(3)/2. Distance between parallel sides is sqrt(3).
    # If aligned, centers need to be sqrt(3) apart in y and 1.5 in x?
    # Lattice vectors for aligned hexagons:
    # v1 = (1.5, sqrt(3)/2), v2 = (0, sqrt(3))
    # This creates a grid where hexagons touch.
    
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0, sqrt3])
    
    # We will vary the scale of this lattice and the rotation of the whole cluster.
    # Also allow small perturbations to break symmetry if needed? 
    # For now, stick to exact lattice points to ensure validity, then refine.
    
    candidates = []
    
    # Try different lattice scales (stretching/compressing)
    for scale_x in [0.9, 0.95, 1.0, 1.05, 1.1]:
        for scale_y in [0.9, 0.95, 1.0, 1.05, 1.1]:
            v1 = base_v1 * np.array([scale_x, 1.0])
            v2 = base_v2 * np.array([1.0, scale_y])
            
            # Generate a large enough grid to pick 12 points
            # We need to pick 12 points from this lattice that form a compact cluster.
            # The "best" 12 points are usually the ones closest to the centroid.
            
            points = []
            for i in range(-4, 5):
                for j in range(-4, 5):
                    p = i * v1 + j * v2
                    points.append(p)
            
            # Sort by distance from origin
            points.sort(key=lambda p: np.dot(p, p))
            
            # Take the first 12
            selected = np.array(points[:12])
            
            # Center the cluster
            centroid = np.mean(selected, axis=0)
            selected -= centroid
            
            # Try different global rotations for this cluster
            for rot_deg in [0, 30, 60, 90]:
                rot_rad = math.radians(rot_deg)
                rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                    [math.sin(rot_rad), math.cos(rot_rad)]])
                rotated_selected = selected @ rot_mat.T
                
                # Assign angles. For aligned packing, all angles are 0 (or rot_deg).
                # But we can also try mixing angles? 
                # Let's stick to uniform angle first for validity, then SLSQP will optimize.
                angles = np.full(N, rot_deg)
                
                # Check validity
                if is_valid(rotated_selected, angles, tol=1e-6):
                    score = calc_min_outer_side(rotated_selected, angles, 0.0)
                    if score < best_valid_score:
                        best_valid_score = score
                        best_valid_config = (rotated_selected.copy(), angles.copy())
                    
                    if time.time() > deadline - 20:
                        break
            if time.time() > deadline - 20:
                break
        if time.time() > deadline - 20:
            break

    # If grid search found nothing, fallback to a standard honeycomb subset
    if best_valid_config is None:
        print("Grid search failed, using fallback honeycomb")
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * 1.5
                y = i * sqrt3 / 2.0 + j * sqrt3
                lattice_points.append((x, y))
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        init_positions = np.array(lattice_points[:12])
        init_positions -= np.mean(init_positions, axis=0)
        init_angles = np.zeros(N)
        best_valid_config = (init_positions, init_angles)
        best_valid_score = calc_min_outer_side(init_positions, init_angles, 0.0)

    init_positions, init_angles = best_valid_config
    x0 = np.concatenate([init_positions.flatten(), init_angles, [0.0]])
    
    # --- PHASE 2: SLSQP REFINEMENT WITH HARD CONSTRAINTS ---
    # Now that we have a valid configuration, we use SLSQP to minimize the outer side length.
    # SLSQP handles inequality constraints directly, which is superior to penalty methods
    # for maintaining strict non-overlap while pushing boundaries.
    
    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        outer_angle = params[3*N]
        return calc_min_outer_side(positions, angles, outer_angle)
    
    # Define constraints for SLSQP
    # For each pair (i, j), we require gap >= 0.
    # gap = min_separation.
    # We define constraint as gap(i, j) >= 0.
    # SLSQP expects func(x) >= 0 for inequality constraints.
    
    def make_constraint(i, j):
        def constraint(params):
            pos_i = params[2*i:2*i+2]
            pos_j = params[2*j:2*j+2]
            ang_i = params[2*N + i]
            ang_j = params[2*N + j]
            
            poly_i = get_hex_verts(pos_i[0], pos_i[1], 1.0, ang_i)
            poly_j = get_hex_verts(pos_j[0], pos_j[1], 1.0, ang_j)
            
            return sat_gap(poly_i, poly_j)
        return constraint

    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, 360))
    bounds.append((0, 60))  # outer angle

    best_score = best_valid_score
    best_result = x0.copy()

    # Run SLSQP with outer angle as a free parameter
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 400, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.success or res.fun < best_score:
            pos = res.x[:2*N].reshape(N, 2)
            ang = res.x[2*N:3*N]
            outer_ang = res.x[3*N]
            
            # Verify validity strictly
            if is_valid(pos, ang, tol=1e-4):
                score = calc_min_outer_side(pos, ang, outer_ang)
                if score < best_score:
                    best_score = score
                    best_result = res.x.copy()
    except Exception as e:
        print(f"SLSQP error: {e}")

    # --- PHASE 3: LOCAL POLISHING with smooth penalty (Nelder-Mead) ---
    # Use a smooth soft-min surrogate for the gap to avoid non-smoothness
    # that traps gradient-free methods at kinks.
    
    if time.time() < deadline - 10:
        def soft_min_penalty(polys):
            """Smooth penalty: for each pair, compute soft-min of projection gaps."""
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    # Compute all 12 separating axes (6 from each polygon)
                    axes = []
                    for k in range(6):
                        e1 = polys[i][(k+1)%6] - polys[i][k]
                        n1 = np.array([-e1[1], e1[0]])
                        nm = np.linalg.norm(n1)
                        if nm > 1e-9:
                            axes.append(n1 / nm)
                    for k in range(6):
                        e2 = polys[j][(k+1)%6] - polys[j][k]
                        n2 = np.array([-e2[1], e2[0]])
                        nm = np.linalg.norm(n2)
                        if nm > 1e-9:
                            axes.append(n2 / nm)
                    
                    # For each axis, compute the gap (positive = separated)
                    gaps = []
                    for ax in axes:
                        p1 = polys[i] @ ax
                        p2 = polys[j] @ ax
                        gap = max(np.min(p1) - np.max(p2), np.min(p2) - np.max(p1))
                        gaps.append(gap)
                    gaps = np.array(gaps)
                    # The actual gap is the max of all axis gaps (separation distance)
                    # But for penalty, we want: if any axis shows overlap (gap < 0), penalize.
                    # The "worst" overlap is min(gaps). If min(gaps) < 0, they overlap.
                    # Use soft-min for smoothness: softmin(gaps) approximates min(gaps)
                    # Actually we want the separation = max(gaps). If max(gaps) < 0, deep overlap.
                    # For penalty: penalize when max(gaps) < 0 (overlap).
                    # Smooth: use -softplus(-max(gaps)) style, or just use the min gap with soft approximation.
                    # Simpler: the minimum gap across axes gives the tightest constraint.
                    # If min_gap < 0, they overlap on that axis. The penetration depth is -min_gap.
                    # Use a smooth approximation of max(0, -min_gap)^2
                    min_gap = np.min(gaps)
                    if min_gap < 0:
                        # Smooth: use sigmoid-like transition
                        penalty += 500.0 * min_gap * min_gap
            return penalty

        def obj_penalty(params):
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:3*N]
            outer_angle = params[3*N]
            side_len = calc_min_outer_side(positions, angles, outer_angle)
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            penalty = soft_min_penalty(polys)
            return side_len + penalty

        try:
            res2 = minimize(
                obj_penalty, 
                best_result, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'disp': False}
            )
            
            pos = res2.x[:2*N].reshape(N, 2)
            ang = res2.x[2*N:3*N]
            outer_ang = res2.x[3*N]
            
            if is_valid(pos, ang, tol=1e-5):
                score2 = calc_min_outer_side(pos, ang, outer_ang)
                if score2 < best_score:
                    best_score = score2
                    best_result = res2.x.copy()
        except Exception:
            pass

    # Finalize
    final_positions = best_result[:2*N].reshape(N, 2)
    final_angles = best_result[2*N:3*N]
    final_outer_angle = best_result[3*N]
    final_side = calc_min_outer_side(final_positions, final_angles, final_outer_angle)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, final_outer_angle

if __name__ == "__main__":
    construct()
