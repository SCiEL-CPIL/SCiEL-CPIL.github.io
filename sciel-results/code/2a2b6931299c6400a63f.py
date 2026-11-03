import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

RING_COUNTS = [1, 6, 12, 18, 13]

def compute_radius(points):
    """Compute the maximum radius r such that circles of radius r centered at points
    are inside the unit circle and do not overlap."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r = min(min_dist / 2.0, min_boundary)
    return r if r > 0 else 0.0


def build_points_from_params(params):
    """
    Build 50 points from 10 parameters:
    params[0:5] = ring radii (r0, r1, r2, r3, r4)
    params[5:10] = ring rotation angles (a0, a1, a2, a3, a4)
    
    Ring 0: 1 point at center (radius 0, angle irrelevant)
    Ring 1: 6 points at radius r1, starting angle a1, spaced 60 deg
    Ring 2: 12 points at radius r2, starting angle a2, spaced 30 deg
    Ring 3: 18 points at radius r3, starting angle a3, spaced 20 deg
    Ring 4: 13 points at radius r4, starting angle a4, spaced 360/13 deg
    """
    points = np.zeros((N, 2))
    idx = 0
    
    # Ring 0: center
    points[idx] = [0.0, 0.0]
    idx += 1
    
    for ring_i in range(1, 5):
        radius = params[ring_i - 1]  # params[0] is r0 (center, unused for radius)
        # Actually let me re-index: params[0]=r0(center radius, should be 0), params[1]=r1, etc.
        # Let's use: params[0:5] = radii for rings 0-4 (ring 0 radius should be ~0)
        # params[5:10] = angles for rings 0-4 (ring 0 angle irrelevant)
        
        r = params[ring_i - 1]  # This is wrong indexing. Let me fix.
        # params[0] = radius ring 0 (center, ~0)
        # params[1] = radius ring 1
        # params[2] = radius ring 2
        # params[3] = radius ring 3
        # params[4] = radius ring 4
        # params[5] = angle ring 0 (irrelevant)
        # params[6] = angle ring 1
        # params[7] = angle ring 2
        # params[8] = angle ring 3
        # params[9] = angle ring 4
        
        radius = params[ring_i]  # ring 1 -> params[1], ring 2 -> params[2], etc.
        angle_offset = params[5 + ring_i]  # ring 1 -> params[6], ring 2 -> params[7], etc.
        
        n_points = RING_COUNTS[ring_i]
        for k in range(n_points):
            angle = angle_offset + 2 * np.pi * k / n_points
            points[idx, 0] = radius * np.cos(angle)
            points[idx, 1] = radius * np.sin(angle)
            idx += 1
    
    return points


def structured_objective(params):
    """
    Objective to minimize: negative of the soft-min radius.
    params: 10 values (5 radii, 5 angles)
    """
    points = build_points_from_params(params)
    
    # Compute margins
    norms = np.linalg.norm(points, axis=1)
    f = 1.0 - norms  # boundary margins, N values
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    g = dists / 2.0  # pairwise margins, N*N values
    
    all_margins = np.concatenate([f.ravel(), g.ravel()])
    
    # Soft min with adaptive beta
    beta = 100.0
    
    max_margin = np.max(all_margins)
    shifted = beta * (max_margin - all_margins)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    log_sum_exp = np.log(np.sum(exp_shifted))
    
    soft_min = max_margin - (1.0 / beta) * log_sum_exp
    
    return -soft_min


def structured_objective_and_grad(params):
    """
    Objective and gradient for structured optimization.
    Uses finite differences for gradient since analytic gradient through
    the structured parameterization is complex but feasible.
    Actually, let's use analytic gradient.
    """
    points = build_points_from_params(params)
    
    norms = np.linalg.norm(points, axis=1)
    f = 1.0 - norms
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    g = dists / 2.0
    
    all_margins = np.concatenate([f.ravel(), g.ravel()])
    
    beta = 100.0
    max_margin = np.max(all_margins)
    shifted = beta * (max_margin - all_margins)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    p = exp_shifted / sum_exp  # weights
    
    soft_min = max_margin - (1.0 / beta) * np.log(sum_exp)
    obj = -soft_min
    
    # Gradient via chain rule
    # d(soft_min)/d(params) = sum_k p_k * d(margin_k)/d(params)
    # We need d(margin_k)/d(params) for each margin
    
    grad = np.zeros(10)
    
    # Helper: get point index for a given ring and position
    # Ring 0: point 0
    # Ring 1: points 1-6
    # Ring 2: points 7-18
    # Ring 3: points 19-36
    # Ring 4: points 37-49
    
    def point_index(ring, pos_in_ring):
        if ring == 0:
            return 0
        offset = 1
        for r in range(1, ring):
            offset += RING_COUNTS[r]
        return offset + pos_in_ring
    
    # For each point i, compute d(point_i)/d(params)
    # point_i depends on radius and angle of its ring
    
    # Precompute for each point which ring it belongs to and its position
    point_ring = np.zeros(N, dtype=int)
    point_pos = np.zeros(N, dtype=int)
    
    idx = 0
    point_ring[idx] = 0
    point_pos[idx] = 0
    idx += 1
    for ring in range(1, 5):
        n = RING_COUNTS[ring]
        for pos in range(n):
            point_ring[idx] = ring
            point_pos[idx] = pos
            idx += 1
    
    # d(point_i)/d(radius_r) and d(point_i)/d(angle_r)
    # For point i in ring r:
    #   x_i = r * cos(a_r + 2*pi*pos/n)
    #   y_i = r * sin(a_r + 2*pi*pos/n)
    #   dx_i/dr = cos(angle_i), dy_i/dr = sin(angle_i)
    #   dx_i/da = -r * sin(angle_i), dy_i/da = -r * cos(angle_i)
    
    # Boundary margins: f_i = 1 - |c_i|
    # d(f_i)/d(params) = -d(|c_i|)/d(params)
    # For point i in ring r (r > 0):
    #   |c_i| = r (the ring radius)
    #   d(f_i)/d(r_r) = -1
    #   d(f_i)/d(a_r) = 0
    # For center point (ring 0): f_0 = 1, constant, grad = 0
    
    # Pairwise margins: g_ij = |c_i - c_j| / 2
    # d(g_ij)/d(params) = (1/2) * d|c_i - c_j|/d(params)
    # = (1/2) * ((c_i - c_j) / |c_i - c_j|) · d(c_i - c_j)/d(params)
    # = (1/2) * ((c_i - c_j) / d_ij) · (d(c_i)/d(params) - d(c_j)/d(params))
    
    # Let's compute gradients for each point w.r.t. each parameter
    # For point i in ring r > 0:
    #   angle_i = a_r + 2*pi*pos/n
    #   d(c_i)/d(r_r) = [cos(angle_i), sin(angle_i)]
    #   d(c_i)/d(a_r) = [-r*sin(angle_i), -r*cos(angle_i)]
    # For point 0 (center): d(c_0)/d(anything) = [0, 0]
    
    # Precompute per-point gradients w.r.t. the 10 params
    # grad_point[i, param_idx] = d(c_i)/d(param_idx)
    grad_point = np.zeros((N, 10, 2))
    
    for i in range(N):
        ring = point_ring[i]
        if ring == 0:
            continue  # center, no dependence
        pos = point_pos[i]
        n = RING_COUNTS[ring]
        radius = params[ring]  # ring 1 -> params[1], etc.
        angle_offset = params[5 + ring]  # ring 1 -> params[6], etc.
        angle_i = angle_offset + 2 * np.pi * pos / n
        
        cos_a = np.cos(angle_i)
        sin_a = np.sin(angle_i)
        
        # d(c_i)/d(radius_r) where r = ring
        grad_point[i, ring, 0] = cos_a
        grad_point[i, ring, 1] = sin_a
        
        # d(c_i)/d(angle_r) where r = ring
        grad_point[i, 5 + ring, 0] = -radius * sin_a
        grad_point[i, 5 + ring, 1] = -radius * cos_a
    
    # Now compute gradient of soft_min w.r.t. params
    # grad(soft_min) = sum_k p_k * d(margin_k)/d(params)
    
    # Boundary margins: indices 0..N-1 in all_margins
    for i in range(N):
        w = p[i]
        if w < 1e-15:
            continue
        ring = point_ring[i]
        if ring == 0:
            continue
        # d(f_i)/d(radius_r) = -1, d(f_i)/d(angle_r) = 0
        grad[ring] += w * (-1.0)
    
    # Pairwise margins: indices N..N+N*N-1 in all_margins
    # g_ij is at index N + i*N + j
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = p[N + i * N + j]
            if w < 1e-15:
                continue
            d_ij = dists[i, j]
            if d_ij < 1e-10:
                continue
            vec = points[i] - points[j]
            unit_vec = vec / d_ij
            
            # d(g_ij)/d(params) = (1/2) * unit_vec · (d(c_i)/d(params) - d(c_j)/d(params))
            for param_idx in range(10):
                dc_i = grad_point[i, param_idx, :]
                dc_j = grad_point[j, param_idx, :]
                dot = np.dot(unit_vec, dc_i - dc_j)
                grad[param_idx] += w * 0.5 * dot
    
    # Objective is -soft_min, so gradient is -grad
    return obj, -grad


def initial_params():
    """Generate initial parameters based on known optimal packing structure."""
    # Known optimal r ~ 0.1258
    # Ring radii should be roughly: 0, 2r, 4r, 6r, 8r with some adjustments
    # But the outer ring needs to fit: r_4 + r < 1, so r_4 < 1 - r
    # For r = 0.1258, r_4 < 0.874
    # Let's estimate: rings at approximately 0, 0.25, 0.5, 0.7, 0.87
    
    params = np.zeros(10)
    # Radii (params[0:5])
    params[0] = 0.0  # center
    params[1] = 0.25  # ring 1
    params[2] = 0.50  # ring 2
    params[3] = 0.70  # ring 3
    params[4] = 0.87  # ring 4
    
    # Angles (params[5:10]) - staggered for better packing
    params[5] = 0.0  # center, irrelevant
    params[6] = 0.0  # ring 1
    params[7] = np.pi / 12  # ring 2, offset by 15 deg
    params[8] = 0.0  # ring 3
    params[9] = np.pi / 26  # ring 4, small offset
    
    return params


def optimize_structured(params0, time_limit=20.0, maxiter=200):
    """Run L-BFGS-B on the structured parameter space."""
    start_time = time.time()
    
    def fun_and_grad(params):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        obj, grad = structured_objective_and_grad(params)
        if not np.all(np.isfinite(grad)):
            grad = np.zeros(10)
        return obj, grad
    
    try:
        result = minimize(fun_and_grad, params0, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10,
                                  'maxfun': 10000})
        if result.x is not None and np.all(np.isfinite(result.x)):
            return result.x
    except Exception:
        pass
    
    return params0


def construct():
    deadline = time.time() + DEADLINE
    
    np.random.seed(42)
    
    best_r = 0.0
    best_points = None
    best_params = None
    
    # Generate multiple initializations with different angle offsets
    angle_offsets = [
        [0.0, 0.0, np.pi/12, 0.0, np.pi/26],
        [0.0, np.pi/6, 0.0, np.pi/18, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, np.pi/12, np.pi/24, np.pi/36, np.pi/48],
        [0.0, 0.0, np.pi/6, np.pi/9, np.pi/13],
    ]
    
    radius_scales = [
        [0.0, 0.25, 0.50, 0.70, 0.87],
        [0.0, 0.26, 0.51, 0.72, 0.88],
        [0.0, 0.24, 0.49, 0.69, 0.86],
    ]
    
    # Multi-start: try different combinations
    starts = []
    for radii in radius_scales:
        for angles in angle_offsets:
            params = np.array(radii + angles)
            starts.append(params)
    
    # Add some random perturbations
    for _ in range(5):
        base = initial_params()
        base[1:] += np.random.normal(0, 0.02, 9)
        starts.append(base)
    
    for start_idx, params0 in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        time_left = deadline - time.time() - 10
        if time_left < 5:
            break
        
        # Run optimization with time limit proportional to remaining time
        t_limit = min(time_left * 0.7, 25.0)
        
        params_opt = optimize_structured(params0, time_limit=t_limit, maxiter=150)
        
        points = build_points_from_params(params_opt)
        r = compute_radius(points)
        
        if r > best_r:
            best_r = r
            best_points = points.copy()
            best_params = params_opt.copy()
    
    # If we have a good solution, try to refine with a local search
    if best_params is not None and time.time() < deadline - 15:
        time_left = deadline - time.time() - 5
        if time_left > 10:
            # Try a few more optimization runs from the best solution
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                params_perturbed = best_params + np.random.normal(0, 0.01, 10)
                t_limit = min(deadline - time.time() - 5, 15.0)
                params_opt = optimize_structured(params_perturbed, time_limit=t_limit, maxiter=100)
                points = build_points_from_params(params_opt)
                r = compute_radius(points)
                if r > best_r:
                    best_r = r
                    best_points = points.copy()
                    best_params = params_opt.copy()
    
    # Final: if we haven't found anything good, use a fallback
    if best_points is None or best_r < 0.1:
        # Fallback: use a simple ring placement
        best_params = initial_params()
        best_points = build_points_from_params(best_params)
        best_r = compute_radius(best_points)
    
    # Ensure validity: check all points are inside unit circle
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] > 0.999:
            best_points[i] *= 0.999 / norms[i]
    
    return best_points


if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    r = compute_radius(pts)
    print(f"Radius: {r:.10f}")
    print(f"Shape: {pts.shape}")
    print(f"Time: {elapsed:.1f}s")
