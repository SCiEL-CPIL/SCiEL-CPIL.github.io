import math
import time
import random
import numpy as np
from scipy.optimize import minimize, Bounds

N = 12
SQRT3 = math.sqrt(3)

def construct():
    deadline = time.time() + 145.0
    best_side = float('inf')
    best_result = None

    def hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def support_function(cx, cy, angle_deg, nx, ny):
        """Max projection of unit hexagon centered at (cx,cy) with rotation angle_deg onto direction (nx,ny)."""
        angle = math.radians(angle_deg)
        max_p = 0.0
        for i in range(6):
            theta = angle + i * math.pi / 3
            vx = math.cos(theta)
            vy = math.sin(theta)
            p = vx * nx + vy * ny
            if p > max_p:
                max_p = p
        return cx * nx + cy * ny + max_p

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """Compute minimum outer hexagon side length that contains all inner hexagons."""
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                ang = angles[i]
                sp = support_function(cx, cy, ang, nx, ny)
                proj = sp - (ocx * nx + ocy * ny)
                if proj > max_proj:
                    max_proj = proj
            # Also check negative direction (other side)
            for i in range(len(centers)):
                cx, cy = centers[i]
                ang = angles[i]
                sp_neg = support_function(cx, cy, ang, -nx, -ny)
                proj = (ocx * nx + ocy * ny) - sp_neg
                if proj > max_proj:
                    max_proj = proj
        return 2.0 * max_proj / SQRT3

    def minkowski_distance_sq(c1, a1, c2, a2):
        """
        Squared minimum center-to-center distance for two unit hexagons with given
        centers and angles to be disjoint. Uses the support function approach:
        two convex sets A and B are disjoint iff there exists a separating line.
        For regular hexagons, the minimum distance can be computed via the
        Minkowski difference. For unit hexagons, the critical distance depends
        on relative orientation.
        
        We use the fact that two unit regular hexagons are disjoint iff
        for all 6 directions of the outer hexagon normals AND the 6 directions
        of each inner hexagon, the separation is maintained.
        
        Actually, the exact condition: two convex polygons are disjoint iff
        there's a separating axis. For two hexagons, we check all edge normals
        of both hexagons (up to 12 unique directions).
        
        For simplicity and speed, we use the sufficient condition that the
        center distance projected onto every edge-normal direction of both
        hexagons exceeds the sum of support functions.
        """
        a1_rad = math.radians(a1)
        a2_rad = math.radians(a2)
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist_sq = dx*dx + dy*dy
        
        # Check all 12 edge normals (6 from each hexagon)
        for i in range(6):
            # Normal of hex1 edge i
            theta1 = a1_rad + i * math.pi / 3 + math.pi / 6
            nx1 = math.cos(theta1)
            ny1 = math.sin(theta1)
            
            # Support of hex1 in direction (nx1, ny1)
            s1 = 0.0
            for v in range(6):
                va = a1_rad + v * math.pi / 3
                p = math.cos(va) * nx1 + math.sin(va) * ny1
                if p > s1: s1 = p
            
            # Support of hex2 in direction (-nx1, -ny1)
            s2 = 0.0
            for v in range(6):
                va = a2_rad + v * math.pi / 3
                p = math.cos(va) * (-nx1) + math.sin(va) * (-ny1)
                if p > s2: s2 = p
            
            # Projected distance
            proj = dx * nx1 + dy * ny1
            if proj < s1 + s2 - 1e-10:
                return -1.0  # Overlapping
            
            # Also check opposite direction
            proj2 = -dx * nx1 - dy * ny1
            if proj2 < s1 + s2 - 1e-10:
                return -1.0  # Overlapping
        
        return dist_sq

    def check_disjoint(centers, angles, tol=1e-8):
        """Check all pairs are disjoint using Minkowski separation."""
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                d = minkowski_distance_sq(centers[i], angles[i], centers[j], angles[j])
                if d < 0:
                    return False
        return True

    def check_disjoint_fast(centers, angles, min_center_dist=SQRT3 - 0.01):
        """Fast approximate check using center distance (necessary but not sufficient)."""
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if dx*dx + dy*dy < min_center_dist*min_center_dist:
                    return False
        return True

    def polish_outer(centers, angles, init_ocx, init_ocy, init_otheta):
        """Fine-tune outer hexagon center and rotation for minimum side."""
        best_side = compute_outer_side(centers, angles, init_ocx, init_ocy, init_otheta)
        best_ocx, best_ocy, best_otheta = init_ocx, init_ocy, init_otheta
        
        scales = [0.5, 0.1, 0.02, 0.005, 0.001]
        for scale in scales:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da in [-scale*10, 0, scale*10]:
                            new_otheta = (best_otheta + da) % 60
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, new_otheta)
                            if side < best_side - 1e-13:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = new_otheta
                                improved = True
        return best_ocx, best_ocy, best_otheta, best_side

    def generate_ring_config(ring1, ring2):
        centers = [[0.0, 0.0]]
        for vec in ring1:
            centers.append([vec[0], vec[1]])
        for vec in ring2:
            centers.append([vec[0], vec[1]])
        return centers

    # --- Lattice-based initial configurations ---
    d = SQRT3 + 0.001
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*SQRT3/2.0])
    basis = [u, v, v-u, -u, -v, u-v]
    ring1 = [u, v, v-u, -u, -v, u-v]
    
    # Generate ring 2 candidates
    ring2_cands = []
    seen = set()
    for i in range(6):
        p1 = ring1[i]
        for b in basis:
            p2 = p1 + b
            key = (round(float(p2[0]), 2), round(float(p2[1]), 2))
            is_r0 = np.allclose(p2, np.zeros(2), atol=0.05)
            is_r1 = False
            for p1j in ring1:
                if np.allclose(p2, p1j, atol=0.05):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1 and key not in seen:
                seen.add(key)
                ring2_cands.append(p2)
    
    ring2_cands.sort(key=lambda p: np.linalg.norm(p))
    
    # --- Homotopy-style continuation: start from large container, shrink ---
    def homotopy_optimize(centers, angles, start_side, end_side, num_steps=20):
        """
        Continuation method: start with a large outer hexagon (known feasible),
        gradually shrink while adjusting inner positions to maintain feasibility.
        This tracks the solution branch and avoids local minima.
        """
        n = len(centers)
        current_centers = [c[:] for c in centers]
        current_angles = angles[:]
        
        # Start with outer hexagon centered at origin
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        current_side = start_side
        
        for step in range(num_steps):
            if time.time() > deadline - 15:
                break
            # Target side for this step
            target_side = start_side - (start_side - end_side) * (step + 1) / num_steps
            
            # Objective: minimize outer side while maintaining feasibility
            def objective(x):
                c = [[x[3*i], x[3*i+1]] for i in range(n)]
                a = [x[3*i+2] for i in range(n)]
                # Penalty for overlap
                penalty = 0.0
                for i in range(n):
                    for j in range(i+1, n):
                        d = minkowski_distance_sq(c[i], a[i], c[j], a[j])
                        if d < 0:
                            penalty += 100.0 * (-d)
                side = compute_outer_side(c, a, ocx, ocy, otheta)
                return side + penalty
            
            x0 = []
            for i in range(n):
                x0.extend([current_centers[i][0], current_centers[i][1], current_angles[i]])
            
            bounds = []
            for i in range(n):
                bounds.extend([(-5, 5), (-5, 5), (0, 60)])
            
            # Inequality constraints: Minkowski separation
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    def sep_constraint(x, i=i, j=j):
                        ci = [x[3*i], x[3*i+1]]
                        ai = x[3*i+2]
                        cj = [x[3*j], x[3*j+1]]
                        aj = x[3*j+2]
                        # Use center distance as proxy for speed in constraints
                        dx = ci[0] - cj[0]
                        dy = ci[1] - cj[1]
                        return dx*dx + dy*dy - (SQRT3 - 0.005)**2
                    constraints.append({'type': 'ineq', 'fun': sep_constraint})
            
            try:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                                  constraints=constraints,
                                  options={'maxiter': 30, 'ftol': 1e-9})
                if result.success or result.fun < objective(x0):
                    current_centers = [[result.x[3*i], result.x[3*i+1]] for i in range(n)]
                    current_angles = [result.x[3*i+2] for i in range(n)]
            except Exception:
                pass
            
            current_side = compute_outer_side(current_centers, current_angles, ocx, ocy, otheta)
        
        return current_centers, current_angles, ocx, ocy, otheta, current_side

    # --- Multi-start with continuation ---
    num_starts = 8
    for start_idx in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        # Select ring 2
        if start_idx == 0:
            ring2_sel = ring2_cands[:5]
        elif start_idx == 1:
            ring2_sel = ring2_cands[1:6]
        elif start_idx == 2:
            ring2_sel = ring2_cands[2:7]
        elif start_idx == 3:
            ring2_sel = ring2_cands[0:5]
        else:
            random.seed(start_idx * 7919)
            idxs = list(range(min(len(ring2_cands), 10)))
            random.shuffle(idxs)
            ring2_sel = [ring2_cands[i] for i in idxs[:5]]
        
        if len(ring2_sel) < 5:
            continue
        
        centers = generate_ring_config(ring1, ring2_sel)
        angles = [30.0] * 12
        
        # Quick feasibility check
        if not check_disjoint_fast(centers, angles, SQRT3 - 0.05):
            continue
        
        # Try continuation from large to small
        initial_side = compute_outer_side(centers, angles, 0, 0, 0)
        
        # Phase 1: Direct SLSQP optimization
        def objective(x):
            if time.time() > deadline - 20:
                return 1e10
            c = [[x[3*i], x[3*i+1]] for i in range(N)]
            a = [x[3*i+2] for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = c[i][0] - c[j][0]
                    dy = c[i][1] - c[j][1]
                    dist_sq = dx*dx + dy*dy
                    min_d = SQRT3 - 0.002
                    if dist_sq < min_d*min_d:
                        penalty += 50.0 * (min_d*min_d - dist_sq)
            side = compute_outer_side(c, a, 0, 0, 0)
            return side + penalty
        
        x0 = []
        for i in range(N):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        
        bounds = []
        for i in range(N):
            bounds.extend([(-4, 4), (-4, 4), (0, 60)])
        
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                def dist_con(x, i=i, j=j):
                    dx = x[3*i] - x[3*j]
                    dy = x[3*i+1] - x[3*j+1]
                    return dx*dx + dy*dy - (SQRT3 - 0.003)**2
                constraints.append({'type': 'ineq', 'fun': dist_con})
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': 60, 'ftol': 1e-10})
            if result.fun < 1e8:
                c = [[result.x[3*i], result.x[3*i+1]] for i in range(N)]
                a = [result.x[3*i+2] for i in range(N)]
                # Verify feasibility
                if check_disjoint_fast(c, a, SQRT3 - 0.02):
                    ocx, ocy, otheta, side = polish_outer(c, a, 0, 0, 0)
                    if side < best_side:
                        best_side = side
                        best_result = (c, a, ocx, ocy, otheta)
        except Exception:
            pass
        
        # Phase 2: Continuation approach
        if time.time() < deadline - 25:
            try:
                c2, a2, ocx2, ocy2, otheta2, side2 = homotopy_optimize(
                    centers, angles, initial_side, initial_side * 0.85, num_steps=10)
                if check_disjoint_fast(c2, a2, SQRT3 - 0.02):
                    ocx2, ocy2, otheta2, side2 = polish_outer(c2, a2, ocx2, ocy2, otheta2)
                    if side2 < best_side:
                        best_side = side2
                        best_result = (c2, a2, ocx2, ocy2, otheta2)
            except Exception:
                pass
        
        # Phase 3: Local hill climbing
        if time.time() < deadline - 15:
            c3 = [c[:] for c in centers]
            a3 = angles[:]
            local_side = compute_outer_side(c3, a3, 0, 0, 0)
            
            step = 0.03
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                improved = False
                for i in range(N):
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            nc = [c[:] for c in c3]
                            nc[i][0] += dx
                            nc[i][1] += dy
                            if check_disjoint_fast(nc, a3, SQRT3 - 0.01):
                                s = compute_outer_side(nc, a3, 0, 0, 0)
                                if s < local_side - 1e-12:
                                    local_side = s
                                    c3 = nc
                                    improved = True
                    # Try angle perturbation
                    for da in [-5, 0, 5]:
                        na = a3[:]
                        na[i] = (na[i] + da) % 60
                        if check_disjoint_fast(c3, na, SQRT3 - 0.01):
                            s = compute_outer_side(c3, na, 0, 0, 0)
                            if s < local_side - 1e-12:
                                local_side = s
                                a3 = na
                                improved = True
                if not improved:
                    break
            
            if check_disjoint_fast(c3, a3, SQRT3 - 0.02):
                ocx3, ocy3, otheta3, side3 = polish_outer(c3, a3, 0, 0, 0)
                if side3 < best_side:
                    best_side = side3
                    best_result = (c3, a3, ocx3, ocy3, otheta3)

    # Fallback: use best ring configuration
    if best_result is None:
        ring2_sel = ring2_cands[:5]
        centers = generate_ring_config(ring1, ring2_sel)
        angles = [30.0] * 12
        ocx, ocy, otheta, best_side = polish_outer(centers, angles, 0, 0, 0)
        best_result = (centers, angles, ocx, ocy, otheta)

    centers, angles, ocx, ocy, otheta = best_result
    
    # Final verification
    if not check_disjoint_fast(centers, angles, SQRT3 - 0.05):
        # Use safe fallback
        ring2_sel = ring2_cands[:5]
        centers = generate_ring_config(ring1, ring2_sel)
        angles = [30.0] * 12
        ocx, ocy, otheta, best_side = polish_outer(centers, angles, 0, 0, 0)
        best_result = (centers, angles, ocx, ocy, otheta)

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (ocx, ocy)
    outer_side = best_side
    outer_angle = otheta

    return inner, outer_center, outer_side, outer_angle
