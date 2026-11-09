import numpy as np
import math
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = np.zeros((6, 2))
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts[k, 0] = cx + side * math.cos(theta)
            verts[k, 1] = cy + side * math.sin(theta)
        return verts

    def get_axes(verts):
        axes = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                normal /= norm
            axes.append(normal)
        return axes

    def sat_separation(verts1, verts2):
        """
        Returns minimum separation distance (positive if separated, negative if overlapping).
        Uses SAT to find the minimum penetration depth.
        """
        axes1 = get_axes(verts1)
        axes2 = get_axes(verts2)
        all_axes = axes1 + axes2
        
        min_sep = float('inf')
        
        for axis in all_axes:
            proj1 = np.dot(verts1, axis)
            proj2 = np.dot(verts2, axis)
            
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            # Separation on this axis
            sep = min(max1, max2) - max(min1, min2)
            
            if sep < min_sep:
                min_sep = sep
        
        return min_sep

    def separation_constraint(i, j, x):
        """
        Constraint: hexagon i and j must be separated.
        x contains [x1,y1,a1, x2,y2,a2, ...] for all N hexagons.
        Returns negative overlap (should be >= 0).
        """
        x1, y1, a1 = x[3*i], x[3*i+1], math.radians(x[3*i+2])
        x2, y2, a2 = x[3*j], x[3*j+1], math.radians(x[3*j+2])
        
        v1 = hexagon_vertices(x1, y1, 1.0, math.degrees(a1))
        v2 = hexagon_vertices(x2, y2, 1.0, math.degrees(a2))
        
        sep = sat_separation(v1, v2)
        return sep  # Should be >= 0 for valid packing

    def objective(x):
        """
        Objective: minimize the maximum distance from center (proxy for outer hexagon size).
        x contains [x1,y1,a1, x2,y2,a2, ...]
        """
        max_dist = 0.0
        for i in range(N):
            xi, yi = x[3*i], x[3*i+1]
            dist = xi*xi + yi*yi
            if dist > max_dist:
                max_dist = dist
        return max_dist

    def objective_grad(x):
        """Gradient of objective."""
        grad = np.zeros_like(x)
        for i in range(N):
            xi, yi = x[3*i], x[3*i+1]
            dist2 = xi*xi + yi*yi
            if dist2 > 1e-10:
                grad[3*i] = 2*xi
                grad[3*i+1] = 2*yi
        return grad

    def constraints_func(x):
        """All pairwise separation constraints."""
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                def make_constraint(i, j):
                    def constraint(x):
                        return separation_constraint(i, j, x)
                    constraint.name = f"sep_{i}_{j}"
                    return constraint
                cons.append({'type': 'ineq', 'fun': make_constraint(i, j)})
        return cons

    def generate_hex_ring(n):
        positions = [(0, 0)]
        ring = 1
        while len(positions) < n:
            q, r = ring, 0
            for side_idx in range(6):
                for step in range(ring):
                    positions.append((q, r))
                    dq = [1, 0, -1, -1, 0, 1][side_idx]
                    dr = [0, -1, -1, 0, 1, 1][side_idx]
                    q += dq
                    r += dr
            ring += 1
        return positions[:n]

    def axial_to_cartesian(q, r, spacing):
        x = spacing * (2.0 * q + r)
        y = spacing * sqrt3 * r
        return x, y

    def evaluate_packing(inner_list, outer_center=(0.0, 0.0), outer_angle=0.0):
        """Evaluate packing: return (side, valid)."""
        # Check disjointness
        for i in range(len(inner_list)):
            for j in range(i+1, len(inner_list)):
                x1, y1, a1 = inner_list[i]
                x2, y2, a2 = inner_list[j]
                v1 = hexagon_vertices(x1, y1, 1.0, a1)
                v2 = hexagon_vertices(x2, y2, 1.0, a2)
                sep = sat_separation(v1, v2)
                if sep < -1e-6:
                    return None, False
        
        # Find minimum outer side
        cx, cy = outer_center
        lo = 1.0
        hi = 25.0
        
        def fits(side):
            for ix, iy, iang in inner_list:
                verts = hexagon_vertices(ix, iy, 1.0, iang)
                for vx, vy in verts:
                    dx = vx - cx
                    dy = vy - cy
                    angle_rad = math.radians(outer_angle)
                    cos_a = math.cos(-angle_rad)
                    sin_a = math.sin(-angle_rad)
                    rx = dx * cos_a - dy * sin_a
                    ry = dx * sin_a + dy * cos_a
                    
                    apothem = side * sqrt3 / 2
                    for k in range(6):
                        theta = math.radians(30 + k * 60)
                        nx = math.cos(theta)
                        ny = math.sin(theta)
                        if rx * nx + ry * ny > apothem + 1e-9:
                            return False
            return True
        
        if not fits(hi):
            return None, False
        if fits(lo):
            return lo
        
        for _ in range(30):
            mid = (lo + hi) / 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        return hi + 1e-10, True

    best_side = float('inf')
    best_result = None

    # Strategy 1: NLP optimization from hexagonal lattice starts
    axial_pos = generate_hex_ring(N)
    
    for spacing in [1.7, 1.6, 1.5, 1.4]:
        if time.time() > deadline - 30:
            break
        
        init_inner = []
        for q, r in axial_pos:
            x, y = axial_to_cartesian(q, r, spacing)
            init_inner.append([x, y, 0.0])
        
        # Center cluster
        cx_mean = sum(p[0] for p in init_inner) / len(init_inner)
        cy_mean = sum(p[1] for p in init_inner) / len(init_inner)
        centered_init = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in init_inner]
        
        # Convert to flat array
        x0 = np.zeros(3*N)
        for i, (xi, yi, ai) in enumerate(centered_init):
            x0[3*i] = xi
            x0[3*i+1] = yi
            x0[3*i+2] = ai
        
        # Run NLP optimization
        try:
            constraints = constraints_func(x0)
            result = minimize(
                objective, 
                x0, 
                method='trust-constr',
                jac=objective_grad,
                constraints=constraints,
                options={'maxiter': 200, 'gtol': 1e-8, 'xtol': 1e-8}
            )
            
            if result.success or result.fun < float('inf'):
                # Extract result
                opt_inner = []
                for i in range(N):
                    xi, yi, ai = result.x[3*i], result.x[3*i+1], result.x[3*i+2]
                    opt_inner.append([xi, yi, ai])
                
                side, valid = evaluate_packing(opt_inner)
                if valid and side < best_side:
                    best_side = side
                    best_result = (opt_inner, (0.0, 0.0), side, 0.0)
                    
        except Exception:
            pass

    # Strategy 2: Simulated Annealing
    if time.time() < deadline - 40:
        rng = np.random.default_rng(42)
        
        # Start from best NLP result or hex lattice
        if best_result is not None:
            current_inner = [list(p) for p in best_result[0]]
            current_side = best_side
        else:
            init_inner = []
            for q, r in axial_pos:
                x, y = axial_to_cartesian(q, r, 1.5)
                init_inner.append([x, y, 0.0])
            cx_mean = sum(p[0] for p in init_inner) / len(init_inner)
            cy_mean = sum(p[1] for p in init_inner) / len(init_inner)
            current_inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in init_inner]
            current_side, _ = evaluate_packing(current_inner)
            if current_side is None:
                current_side = float('inf')
        
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        
        for step in range(500):
            if time.time() > deadline - 20:
                break
            
            # Create perturbation
            candidate = [list(p) for p in current_inner]
            idx = rng.integers(0, N)
            candidate[idx][0] += rng.normal(0, 0.3)
            candidate[idx][1] += rng.normal(0, 0.3)
            candidate[idx][2] += rng.normal(0, 10)
            
            # Evaluate candidate
            cand_side, cand_valid = evaluate_packing(candidate)
            
            if not cand_valid:
                cand_score = float('inf')
            else:
                cand_score = cand_side
            
            cur_score = current_side if current_side < float('inf') else float('inf')
            
            # Accept if better or with probability exp(-delta/T)
            if cand_score < cur_score:
                current_inner = candidate
                current_side = cand_side
                if current_side < best_side:
                    best_side = current_side
                    best_result = (current_inner, (0.0, 0.0), current_side, 0.0)
            else:
                delta = cand_score - cur_score
                if delta < float('inf') and rng.random() < math.exp(-delta / max(T, 1e-10)):
                    current_inner = candidate
                    current_side = cand_side
            
            T *= cooling
            if T < T_min:
                T = T_min

    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
