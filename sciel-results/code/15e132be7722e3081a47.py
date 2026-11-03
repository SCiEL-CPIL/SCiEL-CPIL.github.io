import math
import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def hexagons_disjoint(inner, eps=1e-6):
        """Check if all hexagons are disjoint using SAT."""
        n = len(inner)
        for i in range(n):
            for j in range(i+1, n):
                if not sat_disjoint(inner[i], inner[j], eps):
                    return False
        return True

    def sat_disjoint(hex1, hex2, eps=1e-6):
        """Check if two hexagons are disjoint using Separating Axis Theorem."""
        x1, y1, a1 = hex1
        x2, y2, a2 = hex2
        verts1 = get_hex_vertices(x1, y1, 1.0, a1)
        verts2 = get_hex_vertices(x2, y2, 1.0, a2)
        
        # Get all edge normals from both hexagons
        axes = []
        for verts in [verts1, verts2]:
            for i in range(6):
                x_a, y_a = verts[i]
                x_b, y_b = verts[(i+1)%6]
                dx = x_b - x_a
                dy = y_b - y_a
                length = math.sqrt(dx*dx + dy*dy)
                if length > 1e-10:
                    # Normal perpendicular to edge
                    nx = -dy / length
                    ny = dx / length
                    axes.append((nx, ny))
        
        # Check each axis
        for nx, ny in axes:
            min1, max1 = float('inf'), float('-inf')
            min2, max2 = float('inf'), float('-inf')
            for px, py in verts1:
                proj = px * nx + py * ny
                min1 = min(min1, proj)
                max1 = max(max1, proj)
            for px, py in verts2:
                proj = px * nx + py * ny
                min2 = min(min2, proj)
                max2 = max(max2, proj)
            # If there's a gap, they are disjoint
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        return False

    def compute_min_outer_side(inner, cx, cy, theta_deg):
        """Compute minimum outer hexagon side length for given center and angle."""
        theta = math.radians(theta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for x, y, a in inner:
                verts = get_hex_vertices(x, y, 1.0, a)
                for px, py in verts:
                    proj = (px - cx) * nx + (py - cy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def optimize_configuration(inner, cx, cy, theta_deg, max_iter=200, step_size=0.05):
        """Local optimization to reduce outer side length while maintaining disjointness."""
        best_side = compute_min_outer_side(inner, cx, cy, theta_deg)
        best_inner = [list(h) for h in inner]
        best_cx, best_cy, best_theta = cx, cy, theta_deg
        
        current_inner = [list(h) for h in inner]
        current_cx, current_cy, current_theta = cx, cy, theta_deg
        current_side = best_side
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try small perturbations
            for _ in range(20):
                new_inner = [list(h) for h in current_inner]
                new_cx, new_cy = current_cx, current_cy
                new_theta = current_theta
                
                # Randomly perturb one parameter
                r = random.random()
                if r < 0.7:
                    # Perturb a hexagon position
                    idx = random.randint(0, len(new_inner)-1)
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_inner[idx][0] += dx
                    new_inner[idx][1] += dy
                elif r < 0.85:
                    # Perturb outer center
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    new_cx += dx
                    new_cy += dy
                else:
                    # Perturb outer angle
                    dtheta = random.uniform(-2.0, 2.0)
                    new_theta += dtheta
                
                # Check disjointness
                if not hexagons_disjoint(new_inner):
                    continue
                
                # Compute new side length
                new_side = compute_min_outer_side(new_inner, new_cx, new_cy, new_theta)
                
                if new_side < current_side - 1e-8:
                    current_inner = new_inner
                    current_cx, current_cy, current_theta = new_cx, new_cy, new_theta
                    current_side = new_side
                    improved = True
                    step_size *= 0.95
                    break
            
            if not improved:
                step_size *= 0.8
                if step_size < 0.001:
                    break
            
            # Update best
            if current_side < best_side - 1e-10:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_cx, best_cy, best_theta = current_cx, current_cy, current_theta
        
        return best_inner, best_cx, best_cy, best_theta, best_side

    def get_hex_vertices_array(cx, cy, side, angle_rad):
        """Vectorized hex vertices."""
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def sat_disjoint_fast(hex1, hex2, eps=1e-6):
        """Fast SAT check for two hexagons."""
        x1, y1, a1 = hex1
        x2, y2, a2 = hex2
        verts1 = get_hex_vertices_array(x1, y1, 1.0, math.radians(a1))
        verts2 = get_hex_vertices_array(x2, y2, 1.0, math.radians(a2))
        
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

    def hexagons_disjoint_fast(inner, eps=1e-6):
        """Check if all hexagons are disjoint using fast SAT."""
        n = len(inner)
        for i in range(n):
            for j in range(i+1, n):
                if not sat_disjoint_fast(inner[i], inner[j], eps):
                    return False
        return True

    def smooth_barrier_objective(params, beta=50.0):
        """
        Smooth objective: outer side length + barrier penalty for overlaps.
        params: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, cx, cy, theta]
        """
        if time.time() > deadline - 2.0:
            return 1e10
        
        inner = []
        for i in range(N):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            inner.append([x, y, math.degrees(a)])
        cx, cy, theta = params[3*N], params[3*N+1], params[3*N+2]
        theta_deg = math.degrees(theta)
        
        # Compute outer side length
        side = compute_min_outer_side(inner, cx, cy, theta_deg)
        
        # Barrier penalty for overlaps (smooth approximation)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, a1 = inner[i]
                x2, y2, a2 = inner[j]
                verts1 = get_hex_vertices_array(x1, y1, 1.0, math.radians(a1))
                verts2 = get_hex_vertices_array(x2, y2, 1.0, math.radians(a2))
                
                # Compute minimum distance between edges using SAT-like approach
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
                            
                            gap = min(proj2.max() - proj1.min(), proj1.max() - proj2.min())
                            if gap < min_gap:
                                min_gap = gap
                
                # If min_gap < 0, they overlap. Use smooth penalty.
                if min_gap < 0.05:
                    # Smooth barrier: beta * log(1 + exp(-beta * (min_gap + margin)))
                    margin = 0.01
                    val = min_gap + margin
                    # Soft plus: max(0, x) approx
                    penalty += beta * np.logaddexp(0, -beta * val)
        
        return side + penalty

    def scipy_optimize(inner, cx, cy, theta_deg, max_time=30.0):
        """Use scipy L-BFGS-B to optimize the configuration."""
        if time.time() > deadline - 5.0:
            return None
        
        # Initial parameters
        params = []
        for x, y, a in inner:
            params.extend([x, y, math.radians(a)])
        params.extend([cx, cy, math.radians(theta_deg)])
        
        best_result = None
        best_side = compute_min_outer_side(inner, cx, cy, theta_deg)
        
        start_time = time.time()
        
        try:
            result = minimize(
                smooth_barrier_objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_side:
                # Extract optimized parameters
                opt_inner = []
                for i in range(N):
                    x, y, a = result.x[3*i], result.x[3*i+1], result.x[3*i+2]
                    opt_inner.append([x, y, math.degrees(a)])
                opt_cx, opt_cy = result.x[3*N], result.x[3*N+1]
                opt_theta = math.degrees(result.x[3*N+2])
                
                # Verify disjointness
                if hexagons_disjoint_fast(opt_inner):
                    opt_side = compute_min_outer_side(opt_inner, opt_cx, opt_cy, opt_theta)
                    if opt_side < best_side:
                        best_result = (opt_inner, opt_cx, opt_cy, opt_theta, opt_side)
                        best_side = opt_side
        
        except Exception:
            pass
        
        return best_result

    def generate_ring12_topology():
        """Generate 1+6+5 ring topology."""
        eps = 0.01
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
        eps = 0.01
        d = math.sqrt(3) + eps
        
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = []
        
        # First cluster: center + 5 neighbors (one side open)
        centers.append(np.array([0.0, 0.0]))
        ring1 = [u, v, v-u, -u, -v]  # 5 neighbors, leaving u-v side open
        for vec in ring1:
            centers.append(vec.copy())
        
        # Second cluster: offset by 2*d in x direction
        offset = np.array([2.0*d, 0.0])
        centers.append(offset + np.array([0.0, 0.0]))
        ring1_2 = [u, v, v-u, -u, -v]
        for vec in ring1_2:
            centers.append(offset + vec.copy())
            
        return [[c[0], c[1], 30.0] for c in centers]

    def generate_rect3x4_topology():
        """Generate a 3x4 rectangular grid of hexagons."""
        eps = 0.01
        d = math.sqrt(3) + eps
        
        # For a grid, we need to stagger rows
        # Row spacing: d*sqrt(3)/2 * 2 = d*sqrt(3) for touching
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
            verts = get_hex_vertices(x, y, 1.0, a)
            all_verts.extend(verts)
            
        cx = sum(p[0] for p in all_verts) / len(all_verts)
        cy = sum(p[1] for p in all_verts) / len(all_verts)
        
        best_side = float('inf')
        best_angle = 0.0
        
        # Scan outer hexagon angle
        for theta_deg in np.linspace(0, 60, 61):
            s = compute_min_outer_side(inner, cx, cy, theta_deg)
            if s < best_side:
                best_side = s
                best_angle = theta_deg
                
        return inner, cx, cy, best_angle, best_side

    # Generate multiple topologies and find the best
    topologies = [
        ("ring12", generate_ring12_topology()),
        ("hex6plus6", generate_hex6plus6_topology()),
        ("rect3x4", generate_rect3x4_topology()),
    ]
    
    best_overall = None
    best_overall_side = float('inf')
    
    for name, inner in topologies:
        if time.time() > deadline - 20.0:
            break
            
        # Check if topology is valid (disjoint)
        if not hexagons_disjoint_fast(inner):
            # Try to fix by adding more epsilon
            continue
            
        inner_eval, cx_eval, cy_eval, angle_eval, side_eval = evaluate_topology(inner)
        
        if side_eval < best_overall_side:
            best_overall = (inner_eval, cx_eval, cy_eval, angle_eval, side_eval)
            best_overall_side = side_eval
    
    if best_overall is None:
        # Fallback: use ring12 with larger epsilon
        eps = 0.05
        d = math.sqrt(3) + eps
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
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
            
        inner = [[c[0], c[1], 30.0] for c in centers]
        best_overall = evaluate_topology(inner)
        best_overall_side = best_overall[4]
    
    inner, cx, cy, best_angle, best_side = best_overall
    
    # Local optimization with random search
    if time.time() < deadline - 15.0:
        inner_opt, cx_opt, cy_opt, theta_opt, side_opt = optimize_configuration(
            inner, cx, cy, best_angle, max_iter=100, step_size=0.03
        )
        if side_opt < best_side:
            inner = inner_opt
            cx, cy = cx_opt, cy_opt
            best_angle = theta_opt
            best_side = side_opt
    
    # Scipy optimization
    if time.time() < deadline - 10.0:
        scipy_result = scipy_optimize(inner, cx, cy, best_angle, max_time=25.0)
        if scipy_result is not None:
            inner_sc, cx_sc, cy_sc, theta_sc, side_sc = scipy_result
            if side_sc < best_side:
                inner = inner_sc
                cx, cy = cx_sc, cy_sc
                best_angle = theta_sc
                best_side = side_sc
    
    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
