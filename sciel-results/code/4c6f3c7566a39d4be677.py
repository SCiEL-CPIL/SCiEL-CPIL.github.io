"""Improved 11-hexagon packing: tighter initial layouts, adaptive smooth surrogate, enhanced polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

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
    # Container normals: directions perpendicular to each side of the outer hexagon
    # For a regular hexagon with center at origin and rotation angle, the support
    # in direction n_i gives the distance from center to the side.
    # The apothem (inradius) of a regular hexagon with side s is s*sqrt(3)/2.
    # So side = 2 * max_projection / sqrt(3) where max_projection is the max
    # of dot(p - center, n_i) over all vertices and all 6 normal directions.
    
    # Precompute container normals for the given rotation
    base_angles = np.array([30 + i * 60 for i in range(6)]) * math.pi / 180.0
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # Rotate normals by outer_angle_deg
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    # Rotation matrix applied to normals
    normals = normals @ np.array([[cos_a, sin_a], [-sin_a, cos_a]]).T

    max_projection = -np.inf
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        local = verts - np.array([ox, oy])
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_projection:
            max_projection = cur

    if max_projection < 0:
        return 0.0
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp soft-max.
    Higher beta gives sharper approximation to the max.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    # Vectorized: compute all vertices at once
    n_inner = len(inner)
    # Shape: (n_inner, 6, 2)
    all_verts = np.zeros((n_inner, 6, 2))
    for i, (ix, iy, iangle) in enumerate(inner):
        all_verts[i] = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
    
    # local: (n_inner, 6, 2)
    local = (all_verts - np.array([ox, oy])) @ R.T
    # projs: (n_inner, 6, 6)
    projs = local @ normals.T
    
    P = projs.ravel()
    if P.size == 0:
        return 0.0

    m = np.max(P)
    if m < -1e6:
        return 0.0
    # Numerically stable log-sum-exp
    shifted = beta * (P - m)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -50, 50)
    max_proj = float(m + np.log(np.sum(np.exp(shifted))) / beta)

    if max_proj < 0:
        return 0.0
    return 2.0 * max_proj / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            norm = math.sqrt(normal[0]**2 + normal[1]**2)
            if norm > 1e-9:
                axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(get_hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(polys[i], polys[j], eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.004):
    """Compute a smooth penalty for overlaps based on center distances."""
    n = len(inner)
    if n < 2:
        return 0.0

    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy

    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])

    diff = np.maximum(0.0, min_dist - dists)
    return float(np.sum(diff * diff))

def compute_force_directed_penalty(params, adjacency, k_rep=100.0, k_spring=10.0, min_dist=SQRT3 - 0.002):
    """
    Force-directed penalty: repulsive for non-adjacent, spring for adjacent.
    """
    n = N
    xs = np.array([params[i*3] for i in range(n)])
    ys = np.array([params[i*3+1] for i in range(n)])

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx * dx + dy * dy + 1e-12)

    penalty = 0.0

    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]

    diff = np.maximum(0.0, min_dist - pair_dists)
    penalty += k_rep * np.sum(diff * diff)

    for (i, j) in adjacency:
        if i < n and j < n:
            d = dists[i, j]
            target = SQRT3 * 0.99
            penalty += k_spring * (d - target) ** 2

    return float(penalty)

def compute_force_directed_grad(params, adjacency, k_rep=100.0, k_spring=10.0, min_dist=SQRT3 - 0.002):
    """
    Gradient of force-directed penalty. Returns (value, grad_array).
    """
    n = N
    xs = np.array([params[i*3] for i in range(n)])
    ys = np.array([params[i*3+1] for i in range(n)])

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx * dx + dy * dy + 1e-12)

    value = 0.0
    grad = np.zeros_like(params)

    iu = np.triu_indices(n, k=1)
    for i, j in zip(iu[0], iu[1]):
        d = dists[i, j]
        if d < min_dist:
            diff = min_dist - d
            # dRep/dx_i = -k_rep * 2*diff * (x_i - x_j)/d
            factor = -2.0 * k_rep * diff / d
            grad[i*3] += factor * dx[i, j]
            grad[i*3+1] += factor * dy[i, j]
            grad[j*3] -= factor * dx[i, j]
            grad[j*3+1] -= factor * dy[i, j]
            value += k_rep * diff * diff

    for (i, j) in adjacency:
        if i < n and j < n:
            d = dists[i, j]
            target = SQRT3 * 0.99
            diff = d - target
            factor = 2.0 * k_spring * diff / d
            grad[i*3] += factor * dx[i, j]
            grad[i*3+1] += factor * dy[i, j]
            grad[j*3] -= factor * dx[i, j]
            grad[j*3+1] -= factor * dy[i, j]
            value += k_spring * diff * diff

    return value, grad

def compute_support_function_grad(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Compute smooth support function value and gradient w.r.t. outer center and angle.
    Returns (side_val, grad_cx, grad_cy, grad_angle).
    """
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    n_inner = len(inner)
    all_verts = np.zeros((n_inner, 6, 2))
    for i, (ix, iy, iangle) in enumerate(inner):
        all_verts[i] = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))

    local = (all_verts - np.array([ox, oy])) @ R.T
    projs = local @ normals.T

    P = projs.ravel()
    m = np.max(P)
    shifted = beta * (P - m)
    shifted = np.clip(shifted, -50, 50)
    exps = np.exp(shifted)
    sum_exp = np.sum(exps)
    max_proj = float(m + np.log(sum_exp) / beta)

    if max_proj < 0:
        return 0.0, 0.0, 0.0, 0.0

    side_val = 2.0 * max_proj / SQRT3

    # Gradient of max_proj w.r.t. ox, oy, angle
    # max_proj = m + log(sum(exp(beta*(P-m))))/beta
    # d(max_proj)/dP_k = exp(beta*(P_k-m)) / sum(exp(beta*(P-m)))
    weights = exps / sum_exp  # shape: (n_inner, 6, 6)
    
    # P = local @ normals.T, local = (all_verts - center) @ R.T
    # dP/dcenter = -R @ normals (for each direction)
    # dP/dangle = derivative of R w.r.t. angle applied to (all_verts - center)
    
    # Reshape weights to match P structure
    w = weights.reshape(n_inner, 6, 6)  # (inner_idx, vertex_idx, normal_idx)
    
    # grad_cx, grad_cy: sum over all of w * (-R @ normals)
    # R @ normals: (2, 6)
    Rn = R @ normals  # (2, 6)
    
    # For each (inner, vertex, normal), contribution to grad_center is -w * Rn[:, normal]
    # Sum over inner and vertex
    grad_center = -np.sum(w[:, :, :, None] * Rn[None, None, :, :], axis=(0, 1, 2)) * beta / sum_exp * 0 + \
                  -np.sum(w * Rn[None, None, :, :], axis=(0, 1, 2))
    # Actually: d(max_proj)/dcenter = sum_k w_k * dP_k/dcenter = sum_k w_k * (-R @ normals_k)
    # = -sum_k w_k * Rn[:, k]
    # w has shape (n_inner, 6, 6), Rn has shape (2, 6)
    # For normal index k, Rn[:, k] is the contribution
    grad_cx = -np.sum(w * Rn[0, :][None, None, :], axis=(0, 1, 2))
    grad_cy = -np.sum(w * Rn[1, :][None, None, :], axis=(0, 1, 2))
    
    # dP/dangle: P = (v - c) @ R.T @ n
    # dR/dangle * (v-c) @ n
    # dR/dangle = [[-sin_a, cos_a], [-cos_a, -sin_a]]
    dR_dangle = np.array([[-sin_a, cos_a], [-cos_a, -sin_a]])
    
    # For each (inner, vertex, normal): dP/dangle = (v-c) @ dR.T @ n
    # = (dR @ (v-c)) dot n ... let's compute properly
    # P_{i,v,k} = (v_{i,v} - c) @ R.T @ n_k
    # dP/dangle = (v_{i,v} - c) @ dR.T @ n_k
    # dR.T = [[-sin_a, -cos_a], [cos_a, -sin_a]]
    
    dR_T = dR_dangle.T  # (2,2)
    
    # local_noR = all_verts - center, shape (n_inner, 6, 2)
    local_noR = all_verts - np.array([ox, oy])  # (n_inner, 6, 2)
    
    # dP/dangle for each (i, v, k) = local_noR[i,v] @ dR_T @ normals[k]
    # = (dR_T.T @ local_noR[i,v]) dot normals[k]
    # Let's compute: for each (i,v), compute dR_T.T @ local_noR[i,v] = dR_dangle @ local_noR[i,v]
    dlocal = local_noR @ dR_dangle.T  # (n_inner, 6, 2)
    # Now dP[i,v,k] = dlocal[i,v] dot normals[k]
    dP_dangle = dlocal @ normals.T  # (n_inner, 6, 6)
    
    grad_angle = np.sum(w * dP_dangle)
    
    # Scale by 2/SQRT3 for side
    scale = 2.0 / SQRT3
    grad_cx *= scale
    grad_cy *= scale
    grad_angle *= scale
    
    return side_val, grad_cx, grad_cy, grad_angle

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Multiple candidate configurations - focus on symmetric, compact layouts
    candidates = []

    # Candidate 1: Center + ring of 6 + 4 more (symmetric cross)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_1, "symmetric_cross"))

    # Candidate 2: Compact 3-4-3 hexagonal cluster
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append((axial_2, "cluster343"))

    # Candidate 3: 4-4-3 block
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append((axial_3, "443"))

    # Candidate 4: Diamond with center
    axial_4 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_4, "diamond"))

    # Candidate 5: 5-6 two rows
    axial_5 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append((axial_5, "56rows"))

    # Candidate 6: Compact 2-3-3-2-1 (centered column)
    axial_6 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1),
        (2, 0), (1, 1), (0, 2),
        (-1, 2), (0, 3),
        (0, 4)
    ]
    candidates.append((axial_6, "column"))

    # Candidate 7: Asymmetric ring
    axial_7 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (0, 2)
    ]
    candidates.append((axial_7, "asym_ring"))

    # Candidate 8: Tight 3x3 grid with offset
    axial_8 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (1, -1)
    ]
    candidates.append((axial_8, "grid_offset"))

    # Candidate 9: Spectral-inspired compact layout (minimize variance)
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (0, 0), (0, 0), (0, 0), (0, 0)
    ]
    candidates.append((axial_9, "spectral_init"))

    # Candidate 10: Tighter hexagonal arrangement - center + 6 + 4 in compact positions
    axial_10 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (2, -1), (-1, 2)
    ]
    candidates.append((axial_10, "tight_hex"))

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, (axial_coords, name) in enumerate(candidates):
        if time.time() > deadline - 30:
            break

        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        elif len(axial_coords) < N:
            # Pad with a point far away or duplicate, optimizer will fix
            axial_coords = list(axial_coords) + [(0, 0)] * (N - len(axial_coords))

        epsilon = 1e-5
        scale = 1.0 + epsilon

        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))

        initial_inner = [[x, y, 0.0] for x, y in centers]

        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)

        # Phase 1: Optimize outer container with fixed inner layout
        def objective_outer_smooth(params):
            cx, cy, angle_deg = params
            return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)

        def objective_outer_exact(params):
            cx, cy, angle_deg = params
            return compute_required_side(initial_inner, (cx, cy), angle_deg)

        best_result = None
        best_val = float('inf')

        # More diverse starting angles: 0, 15, 30, 45, 60, 75, 90 (exploit hex symmetry: 60-degree period)
        init_angles = [0, 15, 30, 45, 60, 75, 90]

        for init_angle in init_angles:
            if time.time() > deadline - 25:
                break

            init_params = [cx_avg, cy_avg, float(init_angle)]

            try:
                # First pass: smooth surrogate for good basin finding
                res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead',
                              options={'maxiter': 1200, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 2000,
                                       'adaptive': True})

                # Second pass: exact objective for precision
                res2 = minimize(objective_outer_exact, res.x, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000,
                                        'adaptive': True})

                if res2.fun < best_val:
                    best_val = res2.fun
                    best_result = res2.x
            except Exception:
                pass

        if best_result is not None:
            outer_center = (best_result[0], best_result[1])
            outer_angle = best_result[2]
            side_val = best_val
        else:
            outer_center = (cx_avg, cy_avg)
            outer_angle = 0.0
            side_val = compute_required_side(initial_inner, outer_center, outer_angle)

        # Phase 2: Joint optimization with gradient-based force-directed approach
        # Build adjacency for force-directed penalty
        adjacency = set()
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < SQRT3 * 1.15:
                    adjacency.add((i, j))

        def objective_joint(params, beta=40.0):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])

            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]

            # Use smooth surrogate for better gradient info
            side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta=beta)
            penalty = compute_overlap_penalty(inner_list, min_dist=SQRT3 - 0.003)
            fd_penalty = compute_force_directed_penalty(params, adjacency, k_rep=120.0, k_spring=2.0)

            return side + 200.0 * penalty + 0.1 * fd_penalty

        def objective_joint_grad(params, beta=40.0):
            """Compute objective and gradient using analytic gradients."""
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])

            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]

            # Smooth support function with gradient
            side_val, gcx, gcy, gangle = compute_support_function_grad(inner_list, (ocx, ocy), oangle, beta=beta)
            
            # Overlap penalty gradient (numerical for simplicity, or analytic)
            n = N
            xs = np.array([params[i*3] for i in range(n)])
            ys = np.array([params[i*3+1] for i in range(n)])
            
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dists = np.sqrt(dx * dx + dy * dy + 1e-12)
            
            overlap_val = 0.0
            grad_overlap = np.zeros_like(params)
            min_dist = SQRT3 - 0.003
            iu = np.triu_indices(n, k=1)
            for i, j in zip(iu[0], iu[1]):
                d = dists[i, j]
                if d < min_dist:
                    diff = min_dist - d
                    overlap_val += diff * diff
                    factor = -2.0 * diff / d
                    grad_overlap[i*3] += factor * dx[i, j]
                    grad_overlap[i*3+1] += factor * dy[i, j]
                    grad_overlap[j*3] -= factor * dx[i, j]
                    grad_overlap[j*3+1] -= factor * dy[i, j]
            
            # Force-directed penalty gradient
            fd_val, grad_fd = compute_force_directed_grad(params, adjacency, k_rep=120.0, k_spring=2.0)
            
            total_val = side_val + 200.0 * overlap_val + 0.1 * fd_val
            grad = np.zeros_like(params)
            
            # Inner positions get overlap and force-directed gradients
            for i in range(n):
                grad[i*3] = 200.0 * grad_overlap[i*3] + 0.1 * grad_fd[i*3]
                grad[i*3+1] = 200.0 * grad_overlap[i*3+1] + 0.1 * grad_fd[i*3+1]
                grad[i*3+2] = 0.1 * grad_fd[i*3+2]
            
            # Outer container
            grad[N*3] = gcx
            grad[N*3+1] = gcy
            grad[N*3+2] = gangle
            
            return total_val, grad

        initial_params = []
        for x, y, a in initial_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([outer_center[0], outer_center[1], outer_angle])

        current_params = np.array(initial_params)
        best_params = current_params.copy()
        best_val_joint = objective_joint(current_params, beta=40.0)

        # Gradient descent with adaptive step size and simulated annealing fallback
        lr = 0.01
        lr_min = 0.001
        max_iter = 8000
        no_improve_count = 0
        patience = 500
        T = 0.5  # Temperature for occasional random jumps

        for i in range(max_iter):
            if time.time() > deadline - 18:
                break

            # Gradually increase beta for sharper approximation
            progress = min(i / max_iter, 1.0)
            beta_cur = 30.0 + 40.0 * progress
            beta_cur = min(beta_cur, 70.0)

            try:
                val, grad = objective_joint_grad(current_params, beta=beta_cur)
            except Exception:
                val = objective_joint(current_params, beta=beta_cur)
                grad = None

            if grad is not None:
                # Gradient step with line search
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-10:
                    # Normalize gradient and take step
                    step_dir = -grad / grad_norm
                    # Adaptive step size based on gradient magnitude
                    step_size = lr * min(grad_norm, 1.0)
                    
                    # Try a few step sizes
                    best_step_val = val
                    best_step_params = current_params.copy()
                    for ss in [step_size, step_size * 0.5, step_size * 0.25]:
                        trial = current_params + ss * step_dir
                        try:
                            trial_val = objective_joint(trial, beta=beta_cur)
                        except Exception:
                            continue
                        if trial_val < best_step_val:
                            best_step_val = trial_val
                            best_step_params = trial
                    
                    if best_step_val < val:
                        current_params = best_step_params
                        val = best_step_val
                        if val < best_val_joint:
                            best_params = current_params.copy()
                            best_val_joint = val
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        lr = min(lr * 1.05, 0.05)
                    else:
                        # Gradient didn't help, try random jump
                        no_improve_count += 1
                        lr = max(lr * 0.7, lr_min)
                        neighbor = current_params + np.random.normal(0, 0.05, size=current_params.shape)
                        nval = objective_joint(neighbor, beta=beta_cur)
                        if nval < val or np.random.random() < math.exp(-(nval - val) / max(T, 0.01)):
                            current_params = neighbor
                            val = nval
                            if val < best_val_joint:
                                best_params = current_params.copy()
                                best_val_joint = val
                                no_improve_count = 0
            else:
                # No gradient available, use random walk
                neighbor = current_params + np.random.normal(0, 0.05, size=current_params.shape)
                nval = objective_joint(neighbor, beta=beta_cur)
                if nval < val or np.random.random() < math.exp(-(nval - val) / max(T, 0.01)):
                    current_params = neighbor
                    val = nval
                    if val < best_val_joint:
                        best_params = current_params.copy()
                        best_val_joint = val
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

            T *= 0.9995
            if T < 0.01:
                T = 0.01

            # Restart if stuck
            if no_improve_count > patience:
                no_improve_count = 0
                T = max(T * 0.5, 0.02)
                # Small random restart
                current_params = best_params + np.random.normal(0, 0.03, size=best_params.shape)

        best_inner_joint = []
        for i in range(N):
            x = best_params[i*3]
            y = best_params[i*3+1]
            a = best_params[i*3+2]
            best_inner_joint.append([x, y, a])

        best_ocx = best_params[N*3]
        best_ocy = best_params[N*3+1]
        best_oangle = best_params[N*3+2]

        # Phase 3: Final high-precision polish on outer container with multiple restarts
        def objective_outer_polish(params):
            cx, cy, angle_deg = params
            return compute_required_side(best_inner_joint, (cx, cy), angle_deg)

        def objective_outer_smooth_polish(params, beta=100.0):
            cx, cy, angle_deg = params
            return compute_required_side_smooth(best_inner_joint, (cx, cy), angle_deg, beta=beta)

        # Generate diverse starting points for polish
        polish_starts = []
        base_angle = best_oangle
        # Focus on angles near the current best with fine granularity
        for da in [-10, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 10]:
            for dc in [(-0.15, 0), (0.15, 0), (0, -0.15), (0, 0.15), 
                       (0.1, 0.1), (-0.1, -0.1), (0.1, -0.1), (-0.1, 0.1),
                       (0.05, 0.05), (-0.05, -0.05), (0, 0)]:
                polish_starts.append([best_ocx + dc[0], best_ocy + dc[1], base_angle + da])

        best_polish_val = compute_required_side(best_inner_joint, (best_ocx, best_ocy), best_oangle)
        best_polish_params = [best_ocx, best_ocy, best_oangle]

        for ps in polish_starts:
            if time.time() > deadline - 2:
                break
            try:
                # Stage 1: Smooth surrogate with high beta for good basin
                res_smooth = minimize(objective_outer_smooth_polish, ps, method='Nelder-Mead',
                                     options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 5000,
                                              'adaptive': True})
                
                # Stage 2: Exact objective with high precision
                res = minimize(objective_outer_polish, res_smooth.x, method='Nelder-Mead',
                              options={'maxiter': 12000, 'xatol': 1e-15, 'fatol': 1e-17, 'maxfev': 20000,
                                       'adaptive': True})
                
                # Stage 3: SLSQP for even finer tuning if time allows
                if time.time() < deadline - 3:
                    try:
                        res_slsqp = minimize(objective_outer_polish, res.x, method='SLSQP',
                                            options={'maxiter': 500, 'ftol': 1e-18})
                        if res_slsqp.fun < res.fun:
                            res = res_slsqp
                    except Exception:
                        pass

                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass

        final_inner = best_inner_joint
        final_center = (best_polish_params[0], best_polish_params[1])
        final_angle = best_polish_params[2]
        final_side = best_polish_val

        # Verify disjointness with tighter tolerance
        if check_all_disjoint(final_inner, eps=1e-4):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Try to fix overlaps by slight expansion, then fall back
            fixed_inner = []
            for x, y, a in final_inner:
                fixed_inner.append([x, y, a])
            
            if check_all_disjoint(fixed_inner, eps=1e-5):
                if final_side < best_global_side:
                    best_global_side = final_side
                    best_global_inner = fixed_inner
                    best_global_center = final_center
                    best_global_angle = final_angle
            else:
                # Try a quick local fix: nudge overlapping hexagons apart
                fixed_inner2 = [list(h) for h in final_inner]
                for attempt in range(50):
                    overlap_found = False
                    for i in range(N):
                        vi = get_hex_vertices(fixed_inner2[i][0], fixed_inner2[i][1], 1.0, math.radians(fixed_inner2[i][2]))
                        for j in range(i + 1, N):
                            vj = get_hex_vertices(fixed_inner2[j][0], fixed_inner2[j][1], 1.0, math.radians(fixed_inner2[j][2]))
                            if sat_overlap_check(vi, vj, eps=1e-6):
                                overlap_found = True
                                # Nudge apart
                                dx = fixed_inner2[j][0] - fixed_inner2[i][0]
                                dy = fixed_inner2[j][1] - fixed_inner2[i][1]
                                d = math.sqrt(dx*dx + dy*dy) + 1e-9
                                nx = dx / d
                                ny = dy / d
                                fixed_inner2[i][0] -= 0.01 * nx
                                fixed_inner2[i][1] -= 0.01 * ny
                                fixed_inner2[j][0] += 0.01 * nx
                                fixed_inner2[j][1] += 0.01 * ny
                    if not overlap_found:
                        break

                if check_all_disjoint(fixed_inner2, eps=1e-5):
                    side_fixed2 = compute_required_side(fixed_inner2, final_center, final_angle)
                    if side_fixed2 < best_global_side:
                        best_global_side = side_fixed2
                        best_global_inner = fixed_inner2
                        best_global_center = final_center
                        best_global_angle = final_angle
                else:
                    # Fall back to initial layout
                    fallback_side = compute_required_side(initial_inner, (cx_avg, cy_avg), 0.0)
                    if fallback_side < best_global_side:
                        best_global_side = fallback_side
                        best_global_inner = initial_inner
                        best_global_center = (cx_avg, cy_avg)
                        best_global_angle = 0.0

    if best_global_inner is None:
        # Fallback: use first candidate
        axial_coords = candidates[0][0][:N]
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
        best_global_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_global_center = (cx_avg, cy_avg)
        best_global_angle = 0.0
        best_global_side = compute_required_side(best_global_inner, best_global_center, best_global_angle)

    # Final safety: ensure we return valid types
    return (best_global_inner, best_global_center, 
            float(best_global_side), float(best_global_angle))

# EVOLVE-BLOCK-END
