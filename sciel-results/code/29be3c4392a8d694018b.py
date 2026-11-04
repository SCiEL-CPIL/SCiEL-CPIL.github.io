import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    random.seed(42)
    np.random.seed(42)

    # Precompute unit hexagon vertices at angle 0
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_center, outer_angle_deg):
        """
        Compute the minimum side length of a regular hexagon centered at outer_center
        with rotation outer_angle_deg that contains all hexagons at given positions/angles.
        
        For a regular hexagon with side length s, the apothem (distance from center to
        midpoint of each edge) is s * sqrt(3) / 2.
        The hexagon is the intersection of 6 half-planes. For each of the 6 edge normals,
        the maximum projection of any inner vertex onto that normal (relative to center)
        must be <= apothem = s * sqrt(3) / 2.
        So s >= max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3).
        """
        outer_angle_rad = math.radians(outer_angle_deg)
        cx, cy = outer_center
        
        # The 6 edge normal directions of the outer hexagon
        # For a regular hexagon, edges are at angles: outer_angle + k*60 degrees
        # The outward normals are perpendicular to edges, so at outer_angle + 30 + k*60
        max_proj = 0.0
        for k in range(6):
            normal_angle = outer_angle_rad + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            for i in range(len(positions)):
                px, py = positions[i]
                a = angles[i]
                verts = get_hex_verts(px, py, 1.0, a)
                # Project all vertices onto this normal, relative to outer center
                for vx, vy in verts:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    if proj > max_proj:
                        max_proj = proj
        
        return 2.0 * max_proj / math.sqrt(3)

    def sat_disjoint(hex1, hex2, eps=1e-9):
        """Check if two hexagons are disjoint using Separating Axis Theorem."""
        x1, y1, a1 = hex1
        x2, y2, a2 = hex2
        verts1 = get_hex_verts(x1, y1, 1.0, a1)
        verts2 = get_hex_verts(x2, y2, 1.0, a2)
        
        # Get all edge normals from both hexagons
        for verts in [verts1, verts2]:
            for i in range(6):
                x_a, y_a = verts[i]
                x_b, y_b = verts[(i+1)%6]
                dx = x_b - x_a
                dy = y_b - y_a
                length = math.sqrt(dx*dx + dy*dy)
                if length > 1e-10:
                    nx = -dy / length
                    ny = dx / length
                    
                    proj1 = verts1 @ np.array([nx, ny])
                    proj2 = verts2 @ np.array([nx, ny])
                    
                    if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                        return True
        return False

    def hexagons_disjoint(inner, eps=1e-9):
        """Check if all hexagons are pairwise disjoint."""
        n = len(inner)
        for i in range(n):
            for j in range(i+1, n):
                if not sat_disjoint(inner[i], inner[j], eps):
                    return False
        return True

    def hexagons_in_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-9):
        """Check if all inner hexagons are inside the outer hexagon."""
        cx, cy = outer_center
        outer_angle_rad = math.radians(outer_angle_deg)
        apothem = outer_side * math.sqrt(3) / 2.0
        
        for k in range(6):
            normal_angle = outer_angle_rad + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            for x, y, a in inner:
                verts = get_hex_verts(x, y, 1.0, a)
                for vx, vy in verts:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    if proj > apothem + eps:
                        return False
        return True

    def compute_gap(inner, outer_center, outer_angle_deg):
        """
        Compute the minimum gap between inner hexagons and outer boundary.
        Returns (min_gap, is_valid) where min_gap is the minimum distance
        from any inner hex vertex to the outer hexagon boundary (negative means outside).
        """
        cx, cy = outer_center
        outer_angle_rad = math.radians(outer_angle_deg)
        
        # For each inner hex vertex, compute the maximum projection onto each outer normal
        # The constraint is: proj <= apothem, so gap = apothem - proj
        # We want to find the minimum gap over all vertices and all normals
        
        # Actually, we need to find the outer side length such that all vertices fit.
        # The minimum side length is determined by the maximum projection.
        # gap = apothem - max_proj = s*sqrt(3)/2 - max_proj
        # So for a given s, gap = s*sqrt(3)/2 - max_proj
        
        # Let's just compute max_proj and return it
        max_proj = 0.0
        for k in range(6):
            normal_angle = outer_angle_rad + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            for x, y, a in inner:
                verts = get_hex_verts(x, y, 1.0, a)
                for vx, vy in verts:
                    proj = (vx - cx) * nx + (vy - cy) * ny
                    if proj > max_proj:
                        max_proj = proj
        
        return max_proj

    def objective(params):
        """
        Objective function for SLSQP optimization.
        params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx, cy, theta]
        Returns the outer side length.
        """
        if time.time() > deadline - 2.0:
            return 1e10
        
        positions = []
        angles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = math.degrees(params[3*i+2])
            positions.append((x, y))
            angles.append(a)
        
        cx = params[3*N]
        cy = params[3*N+1]
        theta = math.degrees(params[3*N+2])
        
        return calc_min_outer_side(positions, angles, (cx, cy), theta)

    def non_overlap_constraints(params):
        """
        Constraints for SLSQP: for each pair of hexagons, the minimum distance
        between them must be >= 0 (i.e., they are disjoint).
        
        We use the SAT-based check to compute the minimum gap.
        For two convex polygons, the minimum gap along separating axes can be computed.
        """
        constraints = []
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = math.degrees(params[3*i+2])
            inner.append((x, y, a))
        
        for i in range(N):
            for j in range(i+1, N):
                # Compute minimum gap between hex i and hex j
                x1, y1, a1 = inner[i]
                x2, y2, a2 = inner[j]
                verts1 = get_hex_verts(x1, y1, 1.0, a1)
                verts2 = get_hex_verts(x2, y2, 1.0, a2)
                
                min_gap = float('inf')
                for verts in [verts1, verts2]:
                    for k in range(6):
                        x_a, y_a = verts[k]
                        x_b, y_b = verts[(k+1)%6]
                        dx = x_b - x_a
                        dy = y_b - y_a
                        length = math.sqrt(dx*dx + dy*dy)
                        if length > 1e-10:
                            nx = -dy / length
                            ny = dx / length
                            
                            proj1 = verts1 @ np.array([nx, ny])
                            proj2 = verts2 @ np.array([nx, ny])
                            
                            # Gap along this axis
                            gap = min(proj2.max() - proj1.min(), proj1.max() - proj2.min())
                            if gap < min_gap:
                                min_gap = gap
                
                # Constraint: min_gap >= 0 (with small tolerance)
                constraints.append({'type': 'ineq', 'fun': lambda mg=min_gap: mg - 1e-8})
        
        return constraints

    def generate_ring12_topology():
        """Generate 1+6+5 ring topology."""
        eps = 0.02
        d = math.sqrt(3) + eps
        
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = [np.array([0.0, 0.0])]
        
        ring1_vectors = [
            u, v, v - u, -u, -v, u - v
        ]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
        ring2_candidates = []
        basis = [u, v, v-u, -u, -v, u-v]
        
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
                        
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
            
        return [[c[0], c[1], 30.0] for c in centers]

    def generate_hex6plus6_topology():
        """Generate two clusters of 6 hexagons side by side."""
        eps = 0.02
        d = math.sqrt(3) + eps
        
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = []
        
        # First cluster: center + 5 neighbors
        centers.append(np.array([0.0, 0.0]))
        ring1 = [u, v, v-u, -u, -v]
        for vec in ring1:
            centers.append(vec.copy())
        
        # Second cluster: offset
        offset = np.array([2.0*d, 0.0])
        centers.append(offset + np.array([0.0, 0.0]))
        ring1_2 = [u, v, v-u, -u, -v]
        for vec in ring1_2:
            centers.append(offset + vec.copy())
            
        return [[c[0], c[1], 30.0] for c in centers]

    def generate_rect3x4_topology():
        """Generate a 3x4 rectangular grid of hexagons."""
        eps = 0.02
        d = math.sqrt(3) + eps
        
        row_spacing = d * math.sqrt(3)
        col_spacing = d
        
        centers = []
        for row in range(3):
            for col in range(4):
                x = col * col_spacing
                y = row * row_spacing
                if row % 2 == 1:
                    x += col_spacing / 2.0
                centers.append(np.array([x, y]))
                
        return [[c[0], c[1], 30.0] for c in centers]

    def evaluate_topology(inner):
        """Evaluate a topology and return best configuration."""
        all_verts = []
        for x, y, a in inner:
            verts = get_hex_verts(x, y, 1.0, a)
            all_verts.extend(verts)
            
        cx = sum(p[0] for p in all_verts) / len(all_verts)
        cy = sum(p[1] for p in all_verts) / len(all_verts)
        
        best_side = float('inf')
        best_angle = 0.0
        
        for theta_deg in np.linspace(0, 60, 61):
            positions = [(x, y) for x, y, a in inner]
            angles = [a for x, y, a in inner]
            s = calc_min_outer_side(positions, angles, (cx, cy), theta_deg)
            if s < best_side:
                best_side = s
                best_angle = theta_deg
                
        return inner, cx, cy, best_angle, best_side

    # Generate multiple topologies
    topologies = [
        generate_ring12_topology(),
        generate_hex6plus6_topology(),
        generate_rect3x4_topology(),
    ]
    
    best_overall = None
    best_overall_side = float('inf')
    
    for inner in topologies:
        if time.time() > deadline - 30.0:
            break
        
        if not hexagons_disjoint(inner):
            continue
        
        inner_eval, cx_eval, cy_eval, angle_eval, side_eval = evaluate_topology(inner)
        
        if side_eval < best_overall_side:
            best_overall = (inner_eval, cx_eval, cy_eval, angle_eval, side_eval)
            best_overall_side = side_eval
    
    if best_overall is None:
        inner = generate_ring12_topology()
        best_overall = evaluate_topology(inner)
        best_overall_side = best_overall[4]
    
    inner, cx, cy, best_angle, best_side = best_overall
    
    # SLSQP optimization with multi-start
    def slsqp_optimize(initial_inner, cx0, cy0, theta0, max_time=25.0):
        """Run SLSQP optimization from a given initial configuration."""
        if time.time() > deadline - 5.0:
            return None
        
        params = []
        for x, y, a in initial_inner:
            params.extend([x, y, math.radians(a)])
        params.extend([cx0, cy0, math.radians(theta0)])
        
        best_result = None
        best_side = calc_min_outer_side(
            [(x, y) for x, y, a in initial_inner],
            [a for x, y, a in initial_inner],
            (cx0, cy0), theta0
        )
        
        try:
            constraints = non_overlap_constraints(params)
            
            result = minimize(
                objective,
                params,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result.success:
                opt_inner = []
                for i in range(N):
                    x = result.x[3*i]
                    y = result.x[3*i+1]
                    a = math.degrees(result.x[3*i+2])
                    opt_inner.append([x, y, a])
                opt_cx = result.x[3*N]
                opt_cy = result.x[3*N+1]
                opt_theta = math.degrees(result.x[3*N+2])
                
                # Verify disjointness
                if hexagons_disjoint(opt_inner):
                    opt_side = calc_min_outer_side(
                        [(x, y) for x, y, a in opt_inner],
                        [a for x, y, a in opt_inner],
                        (opt_cx, opt_cy), opt_theta
                    )
                    if opt_side < best_side:
                        best_result = (opt_inner, opt_cx, opt_cy, opt_theta, opt_side)
        
        except Exception:
            pass
        
        return best_result

    # Try SLSQP from the best topology
    if time.time() < deadline - 30.0:
        slsqp_result = slsqp_optimize(inner, cx, cy, best_angle, max_time=25.0)
        if slsqp_result is not None:
            inner_sc, cx_sc, cy_sc, theta_sc, side_sc = slsqp_result
            if side_sc < best_side:
                inner = inner_sc
                cx, cy = cx_sc, cy_sc
                best_angle = theta_sc
                best_side = side_sc
    
    # Local polish with random perturbations
    def polish_configuration(inner, cx, cy, theta_deg, max_iter=300, step_size=0.01):
        """High-precision polish pass."""
        best_side = calc_min_outer_side(
            [(x, y) for x, y, a in inner],
            [a for x, y, a in inner],
            (cx, cy), theta_deg
        )
        best_inner = [list(h) for h in inner]
        best_cx, best_cy, best_theta = cx, cy, theta_deg
        
        current_inner = [list(h) for h in inner]
        current_cx, current_cy, current_theta = cx, cy, theta_deg
        current_side = best_side
        
        for iteration in range(max_iter):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            for _ in range(20):
                new_inner = [list(h) for h in current_inner]
                new_cx, new_cy = current_cx, current_cy
                new_theta = current_theta
                
                r = random.random()
                if r < 0.6:
                    idx = random.randint(0, len(new_inner)-1)
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_inner[idx][0] += dx
                    new_inner[idx][1] += dy
                elif r < 0.75:
                    idx = random.randint(0, len(new_inner)-1)
                    da = random.uniform(-1.0, 1.0)
                    new_inner[idx][2] += da
                elif r < 0.875:
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_cx += dx
                    new_cy += dy
                else:
                    dtheta = random.uniform(-1.0, 1.0)
                    new_theta += dtheta
                
                if not hexagons_disjoint(new_inner):
                    continue
                
                new_side = calc_min_outer_side(
                    [(x, y) for x, y, a in new_inner],
                    [a for x, y, a in new_inner],
                    (new_cx, new_cy), new_theta
                )
                
                if new_side < current_side - 1e-10:
                    current_inner = new_inner
                    current_cx, current_cy, current_theta = new_cx, new_cy, new_theta
                    current_side = new_side
                    improved = True
                    step_size *= 0.97
                    break
            
            if not improved:
                step_size *= 0.85
                if step_size < 0.0005:
                    break
            
            if current_side < best_side - 1e-12:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_cx, best_cy, best_theta = current_cx, current_cy, current_theta
        
        return best_inner, best_cx, best_cy, best_theta, best_side

    # Final polish
    if time.time() < deadline - 5.0:
        inner_pol, cx_pol, cy_pol, theta_pol, side_pol = polish_configuration(
            inner, cx, cy, best_angle, max_iter=200, step_size=0.008
        )
        if side_pol < best_side:
            inner = inner_pol
            cx, cy = cx_pol, cy_pol
            best_angle = theta_pol
            best_side = side_pol

    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
