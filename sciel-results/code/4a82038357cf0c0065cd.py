import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# Precompute hexagon vertices for unit hexagon at angle 0
_HEX_VERTS_0 = np.array([[math.cos(i * math.pi/3), math.sin(i * math.pi/3)] for i in range(6)])

def _rotate_vertices(verts, angle_rad):
    """Rotate vertices array by angle_rad."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return verts @ R.T

def _get_hex_verts_fast(cx, cy, side, angle_deg):
    """Fast hexagon vertices using precomputed template."""
    a = math.radians(angle_deg)
    return _rotate_vertices(_HEX_VERTS_0, a) * side + np.array([cx, cy])

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        verts = _get_hex_verts_fast(cx, cy, side, angle_deg)
        return [(verts[i][0], verts[i][1]) for i in range(6)]

    def sat_disjoint(hex1, hex2, eps=1e-6):
        """Check if two hexagons are disjoint using Separating Axis Theorem."""
        x1, y1, a1 = hex1
        x2, y2, a2 = hex2
        verts1 = _get_hex_verts_fast(x1, y1, 1.0, a1)
        verts2 = _get_hex_verts_fast(x2, y2, 1.0, a2)
        
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
                    n_vec = np.array([nx, ny])
                    proj1 = verts1 @ n_vec
                    proj2 = verts2 @ n_vec
                    if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                        return True
        return False

    def hexagons_disjoint(inner, eps=1e-6):
        """Check if all hexagons are disjoint using SAT."""
        n = len(inner)
        for i in range(n):
            for j in range(i+1, n):
                if not sat_disjoint(inner[i], inner[j], eps):
                    return False
        return True

    def compute_min_outer_side_fast(inner, cx, cy, theta_deg):
        """Fast computation of minimum outer hexagon side length using vectorized ops.
        
        A regular hexagon with center (cx,cy), side s, rotation theta contains point p
        iff for each of the 6 edge-normals n_k, |dot(p - center, n_k)| <= s * sqrt(3)/2.
        So the minimal s = max over all points and all 6 axes of |proj| / (sqrt(3)/2).
        """
        theta = math.radians(theta_deg)
        max_abs_proj = 0.0
        for k in range(6):
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            n_vec = np.array([nx, ny])
            for x, y, a in inner:
                verts = _get_hex_verts_fast(x, y, 1.0, a)
                projs = (verts - np.array([cx, cy])) @ n_vec
                m = abs(projs).max()
                if m > max_abs_proj:
                    max_abs_proj = m
        return max_abs_proj / (math.sqrt(3) / 2.0)

    def structured_local_search(inner, cx, cy, theta_deg, max_iter=500, step_size=0.05):
        """Structured local search with coordinate descent and line search."""
        best_side = compute_min_outer_side_fast(inner, cx, cy, theta_deg)
        best_inner = [list(h) for h in inner]
        best_cx, best_cy, best_theta = cx, cy, theta_deg
        
        current_inner = [list(h) for h in inner]
        current_cx, current_cy, current_theta = cx, cy, theta_deg
        current_side = best_side
        
        # Precompute directions for each hexagon
        directions = []
        for _ in range(N):
            dirs = []
            for ang in np.linspace(0, 2*np.pi, 12, endpoint=False):
                dirs.append((math.cos(ang), math.sin(ang)))
            directions.append(dirs)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            step = step_size
            
            # Try each hexagon in turn
            for i in range(N):
                if time.time() > deadline - 10.0:
                    break
                
                # Try each direction for this hexagon
                for dx_dir, dy_dir in directions[i]:
                    new_inner = [list(h) for h in current_inner]
                    new_inner[i][0] += step * dx_dir
                    new_inner[i][1] += step * dy_dir
                    
                    if not hexagons_disjoint(new_inner):
                        continue
                    
                    new_side = compute_min_outer_side_fast(new_inner, current_cx, current_cy, current_theta)
                    
                    if new_side < current_side - 1e-10:
                        current_inner = new_inner
                        current_side = new_side
                        improved = True
                        
                        # Line search: try to go further in same direction
                        for _ in range(5):
                            new_inner2 = [list(h) for h in current_inner]
                            new_inner2[i][0] += step * dx_dir * 0.5
                            new_inner2[i][1] += step * dy_dir * 0.5
                            
                            if not hexagons_disjoint(new_inner2):
                                break
                            
                            new_side2 = compute_min_outer_side_fast(new_inner2, current_cx, current_cy, current_theta)
                            if new_side2 < current_side - 1e-10:
                                current_inner = new_inner2
                                current_side = new_side2
                            else:
                                break
            
            # Try outer center perturbation
            for _ in range(6):
                dx = random.uniform(-step, step)
                dy = random.uniform(-step, step)
                new_cx, new_cy = current_cx + dx, current_cy + dy
                new_side = compute_min_outer_side_fast(current_inner, new_cx, new_cy, current_theta)
                if new_side < current_side - 1e-10:
                    current_cx, current_cy = new_cx, new_cy
                    current_side = new_side
                    improved = True
            
            # Try outer angle perturbation
            for dtheta in [-1.0, -0.5, 0.5, 1.0]:
                new_theta = current_theta + dtheta
                new_side = compute_min_outer_side_fast(current_inner, current_cx, current_cy, new_theta)
                if new_side < current_side - 1e-10:
                    current_theta = new_theta
                    current_side = new_side
                    improved = True
            
            if not improved:
                step *= 0.7
                if step < 0.001:
                    break
            else:
                step *= 0.95
            
            if current_side < best_side - 1e-12:
                best_side = current_side
                best_inner = [list(h) for h in current_inner]
                best_cx, best_cy, best_theta = current_cx, current_cy, current_theta
        
        return best_inner, best_cx, best_cy, best_theta, best_side

    def scipy_polish(inner, cx, cy, theta_deg):
        """Use scipy.optimize.minimize for a smooth local refinement with SLSQP."""
        # Flatten inner hexagons to 1D vector: [x1,y1,a1, x2,y2,a2, ..., cx, cy, theta]
        def flatten(inner, cx, cy, theta):
            vec = []
            for x, y, a in inner:
                vec.extend([x, y, a])
            vec.extend([cx, cy, theta])
            return np.array(vec)
        
        def unflatten(vec):
            inner = []
            for i in range(N):
                inner.append([vec[3*i], vec[3*i+1], vec[3*i+2]])
            return inner, vec[3*N], vec[3*N+1], vec[3*N+2]
        
        x0 = flatten(inner, cx, cy, theta_deg)
        
        # Compute initial disjointness
        def check_disjoint(inner):
            for i in range(N):
                for j in range(i+1, N):
                    if not sat_disjoint(inner[i], inner[j], 1e-7):
                        return False
            return True
        
        # Objective function
        def objective(x):
            inner_tmp, cx_t, cy_t, theta_t = unflatten(x)
            if not check_disjoint(inner_tmp):
                return 1e6
            return compute_min_outer_side_fast(inner_tmp, cx_t, cy_t, theta_t)
        
        # Use Nelder-Mead for derivative-free optimization
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < objective(x0):
                inner_opt, cx_opt, cy_opt, theta_opt = unflatten(result.x)
                if check_disjoint(inner_opt):
                    side_opt = compute_min_outer_side_fast(inner_opt, cx_opt, cy_opt, theta_opt)
                    return inner_opt, cx_opt, cy_opt, theta_opt, side_opt
        except Exception:
            pass
        
        return inner, cx, cy, theta_deg, compute_min_outer_side_fast(inner, cx, cy, theta_deg)

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

    def generate_compact_cluster_topology():
        """Generate a compact 3x4 arrangement with tighter spacing."""
        eps = 0.005
        d = math.sqrt(3) + eps
        
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = []
        
        # 4 rows of 3 hexagons each, staggered
        for row in range(4):
            for col in range(3):
                x = col * u[0]
                y = row * v[1]
                if row % 2 == 1:
                    x += u[0] / 2.0
                centers.append(np.array([x, y]))
        
        return [[c[0], c[1], 30.0] for c in centers]

    def generate_linear_topology():
        """Generate a linear chain of 12 hexagons."""
        eps = 0.01
        d = math.sqrt(3) + eps
        
        u = np.array([d, 0.0])
        
        centers = []
        for i in range(12):
            centers.append(np.array([i * u[0], 0.0]))
        
        return [[c[0], c[1], 30.0] for c in centers]

    def generate_2x6_topology():
        """Generate a 2x6 arrangement."""
        eps = 0.01
        d = math.sqrt(3) + eps
        
        u = np.array([d, 0.0])
        v = np.array([0.0, d * math.sqrt(3)])
        
        centers = []
        for row in range(2):
            for col in range(6):
                x = col * u[0]
                y = row * v[1]
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
        
        # Scan outer hexagon angle with finer resolution
        for theta_deg in np.linspace(0, 60, 241):
            s = compute_min_outer_side_fast(inner, cx, cy, theta_deg)
            if s < best_side:
                best_side = s
                best_angle = theta_deg
                
        return inner, cx, cy, best_angle, best_side

    # Generate multiple topologies and find the best
    topologies = [
        ("ring12", generate_ring12_topology()),
        ("hex6plus6", generate_hex6plus6_topology()),
        ("rect3x4", generate_rect3x4_topology()),
        ("compact", generate_compact_cluster_topology()),
        ("linear", generate_linear_topology()),
        ("2x6", generate_2x6_topology()),
    ]
    
    best_overall = None
    best_overall_side = float('inf')
    
    for name, inner in topologies:
        if time.time() > deadline - 60.0:
            break
            
        # Check if topology is valid (disjoint)
        if not hexagons_disjoint(inner):
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
    
    # Structured local search
    if time.time() < deadline - 50.0:
        inner_opt, cx_opt, cy_opt, theta_opt, side_opt = structured_local_search(
            inner, cx, cy, best_angle, max_iter=400, step_size=0.03
        )
        if side_opt < best_side:
            inner = inner_opt
            cx, cy = cx_opt, cy_opt
            best_angle = theta_opt
            best_side = side_opt
    
    # Scipy-based polish
    if time.time() < deadline - 20.0:
        inner_scipy, cx_scipy, cy_scipy, theta_scipy, side_scipy = scipy_polish(
            inner, cx, cy, best_angle
        )
        if side_scipy < best_side:
            inner = inner_scipy
            cx, cy = cx_scipy, cy_scipy
            best_angle = theta_scipy
            best_side = side_scipy
    
    # Final quick polish with very small steps + outer parameter tuning
    if time.time() < deadline - 5.0:
        step = 0.002
        for _ in range(300):
            if time.time() > deadline - 3.0:
                break
            improved = False
            # Try moving each hexagon
            for i in range(N):
                for dx in [-step, step]:
                    for dy in [-step, step]:
                        new_inner = [list(h) for h in inner]
                        new_inner[i][0] += dx
                        new_inner[i][1] += dy
                        if not hexagons_disjoint(new_inner):
                            continue
                        new_side = compute_min_outer_side_fast(new_inner, cx, cy, best_angle)
                        if new_side < best_side - 1e-12:
                            inner = new_inner
                            best_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # Try adjusting outer center
            if not improved:
                for dcx in [-step, step]:
                    for dcy in [-step, step]:
                        new_side = compute_min_outer_side_fast(inner, cx + dcx, cy + dcy, best_angle)
                        if new_side < best_side - 1e-12:
                            cx += dcx
                            cy += dcy
                            best_side = new_side
                            improved = True
                            break
                    if improved:
                        break
            # Try adjusting outer angle
            if not improved:
                for dtheta in [-0.5, 0.5]:
                    new_side = compute_min_outer_side_fast(inner, cx, cy, best_angle + dtheta)
                    if new_side < best_side - 1e-12:
                        best_angle += dtheta
                        best_side = new_side
                        improved = True
            if not improved:
                step *= 0.7
                if step < 1e-5:
                    break
    
    outer_center = (cx, cy)
    outer_side = best_side
    outer_angle = best_angle
    
    # Final safety check
    if not hexagons_disjoint(inner, 1e-5):
        # Fallback to a known valid configuration
        eps = 0.1
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
        all_verts = []
        for x, y, a in inner:
            verts = get_hex_vertices(x, y, 1.0, a)
            all_verts.extend(verts)
        cx = sum(p[0] for p in all_verts) / len(all_verts)
        cy = sum(p[1] for p in all_verts) / len(all_verts)
        best_angle = 0.0
        best_side = float('inf')
        for theta_deg in np.linspace(0, 60, 61):
            s = compute_min_outer_side_fast(inner, cx, cy, theta_deg)
            if s < best_side:
                best_side = s
                best_angle = theta_deg
        outer_center = (cx, cy)
        outer_side = best_side
        outer_angle = best_angle
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
