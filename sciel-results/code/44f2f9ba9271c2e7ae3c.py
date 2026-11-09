"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    SQRT3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        all_axes = axes1 + axes2
        for axis in all_axes:
            nx, ny = axis
            min1 = min(vx * nx + vy * ny for vx, vy in hex1_verts)
            max1 = max(vx * nx + vy * ny for vx, vy in hex1_verts)
            min2 = min(vx * nx + vy * ny for vx, vy in hex2_verts)
            max2 = max(vx * nx + vy * ny for vx, vy in hex2_verts)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        return 2.0 * max_proj / SQRT3

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    def pbd_step(centers, angles, outer_angle_deg, outer_side, iterations=10):
        """
        Perform Position-Based Dynamics relaxation.
        Constraints:
        1. Non-overlap between inner hexagons.
        2. Containment within outer hexagon.
        """
        # Precompute outer hexagon constraints
        # Outer hexagon is centered at (0,0) with side length `outer_side` and rotation `outer_angle_deg`.
        # The distance from center to edge (apothem) is:
        apothem = outer_side * SQRT3 / 2.0
        
        base_angles = [30, 90, 150, 210, 270, 330]
        outer_normals = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            nx = math.cos(rad)
            ny = math.sin(rad)
            outer_normals.append((nx, ny))
            
        # Precompute inner hexagon vertices and axes for current state
        # We update these in the loop as positions change
        
        for _ in range(iterations):
            # 1. Containment Constraint
            # For each hexagon, check if any vertex is outside the outer hexagon.
            # If so, move the hexagon center to satisfy the constraint.
            # Since hexagons are convex, if all vertices are inside, the hexagon is inside.
            # We project the center so that the "most violating" vertex lies on the boundary.
            
            for i in range(N):
                cx, cy = centers[i]
                a = angles[i]
                verts = get_hex_vertices(cx, cy, a)
                
                max_violation = 0.0
                best_normal = None
                best_vertex = None
                
                for nx, ny in outer_normals:
                    # The constraint is: dot(v, n) <= apothem
                    # Violation = max(dot(v, n)) - apothem
                    # Since the hexagon is centered at c, and n is a normal to the outer hex,
                    # the max dot product for the inner hexagon is roughly dot(c, n) + support_function(inner, n)
                    # For a unit hexagon, the support function in direction n depends on rotation.
                    # However, a simpler PBD approach: just check vertices.
                    # If a vertex v is outside (dot(v, n) > apothem), we need to move c.
                    # The correction is along -n.
                    # Amount = dot(v, n) - apothem.
                    # But moving c by -n * amount moves all vertices.
                    # We take the max violation across all normals.
                    
                    for vx, vy in verts:
                        proj = vx * nx + vy * ny
                        if proj > apothem:
                            violation = proj - apothem
                            if violation > max_violation:
                                max_violation = violation
                                best_normal = (nx, ny)
                                best_vertex = (vx, vy)
                
                if max_violation > 1e-9 and best_normal:
                    # Move center by -max_violation * normal
                    nx, ny = best_normal
                    centers[i] = (cx - nx * max_violation, cy - ny * max_violation)

            # 2. Non-overlap Constraint
            # For each pair, check for overlap. If overlapping, separate them.
            # We use SAT to find the Minimum Translation Vector (MTV).
            for i in range(N):
                for j in range(i+1, N):
                    ci = centers[i]
                    cj = centers[j]
                    ai = angles[i]
                    aj = angles[j]
                    
                    verts_i = get_hex_vertices(ci[0], ci[1], ai)
                    verts_j = get_hex_vertices(cj[0], cj[1], aj)
                    axes_i = get_hex_axes(ai)
                    axes_j = get_hex_axes(aj)
                    
                    all_axes = axes_i + axes_j
                    
                    min_overlap = float('inf')
                    best_axis = None
                    
                    for axis in all_axes:
                        nx, ny = axis
                        min_i = min(vx * nx + vy * ny for vx, vy in verts_i)
                        max_i = max(vx * nx + vy * ny for vx, vy in verts_i)
                        min_j = min(vx * nx + vy * ny for vx, vy in verts_j)
                        max_j = max(vx * nx + vy * ny for vx, vy in verts_j)
                        
                        if max_i < min_j or max_j < min_i:
                            # No overlap on this axis
                            min_overlap = -1.0
                            break
                        
                        overlap = min(max_i, max_j) - max(min_i, min_j)
                        if overlap < min_overlap:
                            min_overlap = overlap
                            best_axis = (nx, ny)
                            
                    if min_overlap > 0:
                        # Overlap exists. Separate along best_axis.
                        # Determine direction: from i to j or j to i?
                        # Project centers onto axis to determine relative position.
                        ci_proj = ci[0] * best_axis[0] + ci[1] * best_axis[1]
                        cj_proj = cj[0] * best_axis[0] + cj[1] * best_axis[1]
                        
                        if ci_proj < cj_proj:
                            # i is "behind" j, push i back, j forward?
                            # Usually we split the correction.
                            # Direction is from i to j.
                            dx = cj[0] - ci[0]
                            dy = cj[1] - ci[1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist < 1e-9:
                                dx, dy = best_axis
                                dist = 1.0
                            else:
                                dx /= dist
                                dy /= dist
                                
                            # Ensure direction aligns with axis normal
                            # If dot((dx,dy), best_axis) < 0, flip
                            if dx * best_axis[0] + dy * best_axis[1] < 0:
                                dx = -dx
                                dy = -dy
                                
                            # Apply correction
                            correction = min_overlap / 2.0
                            centers[i] = (ci[0] - dx * correction, ci[1] - dy * correction)
                            centers[j] = (cj[0] + dx * correction, cj[1] + dy * correction)
                        
                        else:
                            dx = ci[0] - cj[0]
                            dy = ci[1] - cj[1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist < 1e-9:
                                dx, dy = best_axis
                                dist = 1.0
                            else:
                                dx /= dist
                                dy /= dist
                                
                            if dx * best_axis[0] + dy * best_axis[1] < 0:
                                dx = -dx
                                dy = -dy
                                
                            correction = min_overlap / 2.0
                            centers[i] = (ci[0] + dx * correction, ci[1] + dy * correction)
                            centers[j] = (cj[0] - dx * correction, cj[1] - dy * correction)
                            
        return centers

    # Initial configurations
    coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    coords2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    coords3 = [(0,0), (1,0), (-1,0), (0,1), (1,1), (-1,1), (0,-1), (1,-1), (-1,-1), (2,0), (-2,0)]
    coords4 = [(0,0), (1,0), (-1,0), (0,1), (1,1), (-1,1), (0,-1), (1,-1), (-1,-1), (2,1), (-2,1)]
    coords5 = [(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1),(-1,-1),(0,-1),(1,-1)]

    def make_initial_params(coords):
        centers = []
        angles = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            centers.append((x, y))
            angles.append(0.0)
        return centers, angles

    best_centers = None
    best_angles = None
    best_side = float('inf')
    best_oa = 0.0

    configs = [coords1, coords2, coords3, coords4, coords5]

    for config in configs:
        if time.time() > deadline - 20.0:
            break
            
        centers, angles = make_initial_params(config)
        
        # Binary search on outer side length
        # Lower bound: Theoretical minimum for 11 hexagons?
        # A single hexagon has side 1. 
        # 7 hexagons in a flower pattern fit in side 3? 
        # Let's start with a generous bound and tighten.
        lo = 3.0
        hi = 10.0
        
        # First, check if hi is valid
        # We run PBD with a fixed outer side and check if it converges to a valid state.
        # If it does, hi is feasible.
        
        for _ in range(10): # Binary search steps
            mid = (lo + hi) / 2.0
            
            # Try to pack with outer side = mid
            test_centers, test_angles = make_initial_params(config)
            
            # Run PBD relaxation
            # We need to determine the best outer angle. 
            # PBD is sensitive to initial orientation. 
            # We'll try a few outer angles.
            feasible = False
            best_test_side = float('inf')
            best_test_centers = None
            best_test_angles = None
            best_test_oa = 0.0
            
            for oa_deg in [0, 15, 30, 45]:
                if time.time() > deadline - 5.0:
                    break
                    
                tc, ta = make_initial_params(config)
                # Relax
                relaxed_centers = pbd_step(tc, ta, oa_deg, mid, iterations=20)
                
                # Check validity
                hexagons = [get_hex_vertices(c[0], c[1], a) for c, a in zip(relaxed_centers, ta)]
                hex_axes = [get_hex_axes(a) for a in ta]
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    # Check containment
                    # Recalculate actual side length required
                    actual_side = calc_outer_side(hexagons, oa_deg)
                    if actual_side < best_test_side:
                        best_test_side = actual_side
                        best_test_centers = relaxed_centers
                        best_test_angles = ta
                        best_test_oa = oa_deg
                        feasible = True
            
            if feasible:
                hi = mid
                if best_test_side < best_side:
                    best_side = best_test_side
                    best_centers = best_test_centers
                    best_angles = best_test_angles
                    best_oa = best_test_oa
            else:
                lo = mid

    # Final Polish: Use the best found configuration and refine with local search on centers
    # to minimize the side length further.
    if best_centers is not None:
        # Local refinement
        # We can try small perturbations and re-run PBD with a tighter bound?
        # Or just accept the PBD result.
        # Let's do a quick Nelder-Mead on the centers to minimize the side length
        # using the exact objective, starting from the PBD result.
        
        def objective(p):
            centers = [(p[i*2], p[i*2+1]) for i in range(N)]
            angles = best_angles[:]
            
            # Run a few PBD steps to ensure validity? No, that's expensive.
            # Just calculate side length. If invalid, return large penalty.
            hexagons = [get_hex_vertices(c[0], c[1], a) for c, a in zip(centers, angles)]
            hex_axes = [get_hex_axes(a) for a in angles]
            
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                        valid = False
                        break
                if not valid: break
            
            if not valid:
                return 1000.0
                
            side = calc_outer_side(hexagons, best_oa)
            return side

        from scipy.optimize import minimize
        x0 = np.array([c[0] for c in best_centers] + [c[1] for c in best_centers])
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 200})
            if res.fun < best_side:
                # Update best
                new_centers = [(res.x[i*2], res.x[i*2+1]) for i in range(N)]
                hexagons = [get_hex_vertices(c[0], c[1], a) for c, a in zip(new_centers, best_angles)]
                hex_axes = [get_hex_axes(a) for a in best_angles]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                            valid = False
                            break
                    if not valid: break
                if valid:
                    best_side = res.fun
                    best_centers = new_centers
        except Exception:
            pass

    # Construct final output
    if best_centers is None:
        # Fallback
        centers, angles = make_initial_params(coords1)
        best_side = calc_outer_side([get_hex_vertices(c[0], c[1], a) for c, a in zip(centers, angles)], 0.0)
        best_oa = 0.0

    # Center the configuration at origin
    cx_avg = sum(c[0] for c in best_centers) / N
    cy_avg = sum(c[1] for c in best_centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in best_centers]
    
    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], best_angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
