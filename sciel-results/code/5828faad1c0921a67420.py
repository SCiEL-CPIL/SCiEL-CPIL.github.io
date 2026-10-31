import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0  # Safety margin before 150s kill


def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts


def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3


def hex_distance(a, b):
    """
    Compute the minimum distance between two unit hexagons.
    a, b: (x, y, angle_deg)
    Returns positive if separated, 0 if touching, negative if overlapping.
    Uses SAT for accuracy.
    """
    va = get_hex_vertices(a[0], a[1], 1.0, math.radians(a[2]))
    vb = get_hex_vertices(b[0], b[1], 1.0, math.radians(b[2]))
    
    # Get all edge normals
    axes = []
    for poly in [va, vb]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
            if norm > 1e-9:
                axes.append((normal[0] / norm, normal[1] / norm))
    
    min_gap = float('inf')
    for axis in axes:
        min_a = min(p[0] * axis[0] + p[1] * axis[1] for p in va)
        max_a = max(p[0] * axis[0] + p[1] * axis[1] for p in va)
        min_b = min(p[0] * axis[0] + p[1] * axis[1] for p in vb)
        max_b = max(p[0] * axis[0] + p[1] * axis[1] for p in vb)
        
        gap = max(min_a - max_b, min_b - max_a)
        if gap > min_gap:
            min_gap = gap
    
    return -min_gap  # Negative means overlapping


def smooth_overlap_penalty(inner, beta=20.0):
    """
    Smooth approximation of overlap penalty using log-sum-exp.
    Returns a smooth measure of total overlap.
    """
    n = len(inner)
    total_penalty = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            d = hex_distance(inner[i], inner[j])
            # If d < 0, they overlap; penalize exp(-beta * d) for d < 0
            if d < 0.01:
                # Smooth transition: use logistic-like function
                penalty = math.exp(-beta * d)
                total_penalty += penalty
    
    return total_penalty


def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            d = hex_distance(inner[i], inner[j])
            if d < -eps:
                return False
    return True


def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    # Initial configuration: hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]

    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    epsilon = 1e-6
    scale = 1.0 + epsilon

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # --- Phase 1: Quick outer hexagon optimization ---
    def objective_outer(params, inner_list):
        cx, cy, angle_deg = params
        return compute_required_side(inner_list, (cx, cy), angle_deg)

    best_outer_val = float('inf')
    best_outer_params = [cx_avg, cy_avg, 0.0]

    for init_angle in [0, 30, 60, 90, 120, 150]:
        if time.time() - start_time > DEADLINE - 60:
            break

        try:
            res = minimize(
                lambda p: objective_outer(p, initial_inner),
                [cx_avg, cy_avg, float(init_angle)],
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 500}
            )
            if res.fun < best_outer_val:
                best_outer_val = res.fun
                best_outer_params = res.x.copy()
        except Exception:
            pass

    if best_outer_val < float('inf'):
        best_outer_center = (best_outer_params[0], best_outer_params[1])
        best_outer_angle = best_outer_params[2]
        best_side = best_outer_val
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # --- Phase 2: Multi-start Simulated Annealing ---
    def evaluate_layout(inner_list, outer_cx, outer_cy, outer_angle):
        """Evaluate a layout: return (side_length, overlap_penalty, is_valid)"""
        side = compute_required_side(inner_list, (outer_cx, outer_cy), outer_angle)
        
        # Check for overlaps
        has_overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                d = hex_distance(inner_list[i], inner_list[j])
                if d < -1e-6:
                    has_overlap = True
                    break
            if has_overlap:
                break
        
        # Check inside outer
        if not has_overlap:
            if not check_inside_outer(inner_list, (outer_cx, outer_cy), side, outer_angle):
                has_overlap = True
        
        return side, has_overlap

    def simulated_annealing(inner_init, outer_cx_init, outer_cy_init, outer_angle_init, 
                           t_start, t_end, steps, seed=42):
        """Run simulated annealing from given initial state."""
        local_rng = np.random.default_rng(seed)
        
        # State: positions and angles of all hexagons + outer center and angle
        # Total variables: 3*N + 3
        state = []
        for x, y, a in inner_init:
            state.extend([x, y, a])
        state.extend([outer_cx_init, outer_cy_init, outer_angle_init])
        
        current_side, _ = evaluate_layout(
            [[state[i*3], state[i*3+1], state[i*3+2]] for i in range(N)],
            state[N*3], state[N*3+1], state[N*3+2]
        )
        
        best_side_sa = current_side
        best_state = state.copy()
        
        for step in range(steps):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            temp = t_start * (t_end / t_start) ** (step / steps)
            
            # Perturb: move one hexagon or adjust outer
            if local_rng.random() < 0.85:
                # Perturb an inner hexagon
                idx = local_rng.integers(0, N)
                # Small random perturbation
                dx = local_rng.normal(0, 0.1)
                dy = local_rng.normal(0, 0.1)
                da = local_rng.normal(0, 5.0)  # degrees
                
                new_state = state.copy()
                new_state[idx*3] += dx
                new_state[idx*3+1] += dy
                new_state[idx*3+2] += da
            else:
                # Perturb outer
                new_state = state.copy()
                new_state[N*3] += local_rng.normal(0, 0.05)
                new_state[N*3+1] += local_rng.normal(0, 0.05)
                new_state[N*3+2] += local_rng.normal(0, 2.0)
            
            new_side, new_overlap = evaluate_layout(
                [[new_state[i*3], new_state[i*3+1], new_state[i*3+2]] for i in range(N)],
                new_state[N*3], new_state[N*3+1], new_state[N*3+2]
            )
            
            # Acceptance criterion
            if not new_overlap and new_side < current_side:
                state = new_state
                current_side = new_side
                if new_side < best_side_sa:
                    best_side_sa = new_side
                    best_state = state.copy()
            elif new_overlap and current_side < float('inf'):
                # Reject overlapping states unless very hot
                if temp > 0.5 and local_rng.random() < math.exp(-10.0 / max(temp, 0.01)):
                    state = new_state
                    current_side = new_side
            else:
                # Metropolis acceptance for non-overlapping states
                delta = new_side - current_side
                if delta < 0 or local_rng.random() < math.exp(-delta / max(temp, 0.01)):
                    state = new_state
                    current_side = new_side
                    if new_side < best_side_sa:
                        best_side_sa = new_side
                        best_state = state.copy()
        
        return best_side_sa, best_state

    # Run multiple SA starts
    sa_starts = [
        (initial_inner, best_outer_center[0], best_outer_center[1], best_outer_angle, 1.0, 0.01, 2000, 42),
        (initial_inner, best_outer_center[0], best_outer_center[1], best_outer_angle + 15, 0.8, 0.01, 1500, 123),
        (initial_inner, best_outer_center[0], best_outer_center[1], best_outer_angle - 15, 0.8, 0.01, 1500, 456),
    ]
    
    # Add perturbed starts
    for _ in range(2):
        perturbed = []
        for x, y, a in initial_inner:
            perturbed.append([x + rng.normal(0, 0.2), y + rng.normal(0, 0.2), a + rng.normal(0, 10)])
        sa_starts.append((perturbed, best_outer_center[0], best_outer_center[1], best_outer_angle, 0.6, 0.01, 1000, int(rng.integers(0, 10000))))
    
    for start_idx, (inner_init, ocx, ocy, oangle, t_s, t_e, steps, seed) in enumerate(sa_starts):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        try:
            sa_side, sa_state = simulated_annealing(inner_init, ocx, ocy, oangle, t_s, t_e, steps, seed)
            
            if sa_side < best_side:
                # Validate and update
                new_inner = [[sa_state[i*3], sa_state[i*3+1], sa_state[i*3+2]] for i in range(N)]
                new_cx, new_cy, new_angle = sa_state[N*3], sa_state[N*3+1], sa_state[N*3+2]
                
                if check_all_disjoint(new_inner) and check_inside_outer(new_inner, (new_cx, new_cy), sa_side, new_angle):
                    best_side = sa_side
                    best_inner = new_inner
                    best_outer_center = (new_cx, new_cy)
                    best_outer_angle = new_angle
        except Exception:
            pass

    # --- Phase 3: Final polish with Nelder-Mead ---
    if time.time() - start_time < DEADLINE - 15 and best_side < float('inf'):
        def objective_joint(params):
            if time.time() - start_time > DEADLINE - 5:
                return 1e10

            inner_list = []
            for i in range(N):
                x = params[i * 3]
                y = params[i * 3 + 1]
                a = params[i * 3 + 2]
                inner_list.append([x, y, a])

            cx = params[N * 3]
            cy = params[N * 3 + 1]
            angle = params[N * 3 + 2]

            side = compute_required_side(inner_list, (cx, cy), angle)

            # Penalty for overlaps
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = hex_distance(inner_list[i], inner_list[j])
                    if d < 0:
                        penalty += (-d) ** 2 * 100.0

            return side + penalty

        init_params_joint = []
        for x, y, a in best_inner:
            init_params_joint.extend([x, y, a])
        init_params_joint.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

        try:
            res_joint = minimize(
                objective_joint,
                init_params_joint,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1000}
            )

            new_inner = [[res_joint.x[i*3], res_joint.x[i*3+1], res_joint.x[i*3+2]] for i in range(N)]
            new_cx, new_cy, new_angle = res_joint.x[N*3], res_joint.x[N*3+1], res_joint.x[N*3+2]

            if check_all_disjoint(new_inner):
                new_side = compute_required_side(new_inner, (new_cx, new_cy), new_angle)
                if check_inside_outer(new_inner, (new_cx, new_cy), new_side, new_angle) and new_side < best_side:
                    best_side = new_side
                    best_inner = new_inner
                    best_outer_center = (new_cx, new_cy)
                    best_outer_angle = new_angle
        except Exception:
            pass

    # Final safety check
    if best_side == float('inf'):
        best_inner = initial_inner
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle
