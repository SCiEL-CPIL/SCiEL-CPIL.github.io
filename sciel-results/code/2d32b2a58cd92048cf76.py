import numpy as np
import time
import warnings
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _fourier_convex_boundary(fourier_params, n_samples=200):
    """
    Generate a convex boundary from Fourier series parameters.
    fourier_params: [a0, a1, b1, a2, b2] - coefficients for cos/sin terms
    Returns: array of (n_samples, 2) points on the boundary
    """
    a0, a1, b1, a2, b2 = fourier_params
    t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    r = a0 + a1 * np.cos(t) + b1 * np.sin(t) + a2 * np.cos(2 * t) + b2 * np.sin(2 * t)
    # Ensure positive radius
    r = np.maximum(r, 0.01)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])

def _point_in_convex_polygon(point, polygon):
    """Check if a point is inside a convex polygon using cross products."""
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        v1 = point - p1
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < -1e-10:  # Allow small tolerance
            return False
    return True

def _clip_point_to_convex_polygon(point, polygon, max_iter=100):
    """
    Clip a point to be inside a convex polygon by projecting onto edges.
    Returns the clipped point.
    """
    for _ in range(max_iter):
        if _point_in_convex_polygon(point, polygon):
            return point
        # Find the most violated edge
        n = len(polygon)
        min_cross = np.inf
        min_edge = 0
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n]
            v1 = point - p1
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if cross < min_cross:
                min_cross = cross
                min_edge = i
        
        # Project onto the violated edge
        p1 = polygon[min_edge]
        p2 = polygon[(min_edge + 1) % n]
        edge_vec = p2 - p1
        edge_len_sq = np.dot(edge_vec, edge_vec)
        if edge_len_sq < 1e-20:
            continue
        t = np.dot(point - p1, edge_vec) / edge_len_sq
        t = np.clip(t, 0, 1)
        point = p1 + t * edge_vec
        # Push slightly inward
        point = point * 0.999
    return point

def _optimize_points_in_convex_container(container_params, point_angles, n_boundary_samples=100):
    """
    Given convex container parameters and point angular positions,
    find optimal radial distances for each point.
    """
    # Generate boundary
    boundary = _fourier_convex_boundary(container_params, n_boundary_samples)
    
    # Get centroid of boundary
    centroid = np.mean(boundary, axis=0)
    boundary_centered = boundary - centroid
    
    # For each point angle, find the maximum radius that keeps it inside
    points = np.zeros((N, 2))
    for i in range(N):
        angle = point_angles[i]
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Binary search for maximum radius
        lo, hi = 0.0, 2.0
        for _ in range(30):
            mid = (lo + hi) / 2
            test_point = centroid + mid * direction
            if _point_in_convex_polygon(test_point, boundary_centered):
                lo = mid
            else:
                hi = mid
        radius = lo * 0.999  # Slight margin
        points[i] = centroid + radius * direction
    
    return points

def _fourier_point_objective(params):
    """
    Objective for Fourier-parameterized optimization.
    params: [a0, a1, b1, a2, b2, angles_0, ..., angles_12]
    First 5: Fourier coefficients for convex boundary
    Next 13: angles for each point
    """
    container_params = params[:5]
    point_angles = params[5:]
    
    try:
        points = _optimize_points_in_convex_container(container_params, point_angles)
        points = points - np.mean(points, axis=0)
        s = _score(points)
        if s <= 0:
            return 1e10
        return -s
    except Exception:
        return 1e10

def _fourier_de_objective(params):
    """DE-friendly wrapper with bounds checking."""
    a0, a1, b1, a2, b2 = params[:5]
    # Ensure reasonable values
    if a0 < 0.1 or a0 > 3.0:
        return 1e10
    if abs(a1) + abs(b1) > a0 * 0.8:
        return 1e10
    if abs(a2) + abs(b2) > a0 * 0.5:
        return 1e10
    return _fourier_point_objective(params)

def _smooth_objective(x_flat, p=50):
    """Smooth surrogate using log-sum-exp based soft-min for better gradients."""
    points = x_flat.reshape(N, 2)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Log-sum-exp based soft-min: -log(sum(exp(-p * areas))) / p
    # This gives a smooth approximation to min(areas)
    neg_p_areas = -p * areas
    max_neg = np.max(neg_p_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_p_areas - max_neg)))
    soft_min = -log_sum / p
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _coordinate_descent_polish(points, deadline, max_rounds=200):
    """Coordinate descent: optimize each point's x and y separately."""
    pts = points.copy()
    best_s = _score(pts)
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 2:
            break
        improved = False
        for i in range(N):
            for dim in range(2):
                # Save current value
                old_val = pts[i, dim]
                
                # Try to optimize this coordinate using golden section or small search
                # Use a small grid search around current value
                best_local = old_val
                best_local_s = best_s
                
                # Try perturbations of decreasing size
                for step in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
                    for delta in [-step, step]:
                        cand = pts.copy()
                        cand[i, dim] = old_val + delta
                        s = _score(cand)
                        if s > best_local_s:
                            best_local_s = s
                            best_local = old_val + delta
                
                if best_local != old_val:
                    pts[i, dim] = best_local
                    if best_local_s > best_s:
                        best_s = best_local_s
                        improved = True
        
        if not improved:
            break
    
    return pts, best_s

def _objective_full(x_flat):
    """Full objective: minimize negative score."""
    points = x_flat.reshape(N, 2)
    points = points - np.mean(points, axis=0)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    # Phase 1: Coordinate descent for quick local improvements
    if time.time() < deadline - 15:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(points, deadline)
            if cd_s > best_s:
                best_s = cd_s
                best_x = cd_pts.flatten().copy()
        except Exception:
            pass
    
    # Phase 2: Powell on smooth objective
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=50),
                best_x,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead on full objective
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final coordinate descent
    if time.time() < deadline - 2:
        try:
            cd_pts2, cd_s2 = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
            if cd_s2 > best_s:
                best_s = cd_s2
                best_x = cd_pts2.flatten().copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_s

def _simulated_annealing(points, deadline):
    """Simulated annealing in Cartesian space with adaptive step size."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_s = current_score
    
    time_budget = deadline - 3.0
    total_time = max(0.5, time_budget - time.time())
    est_iters = int(total_time * 500)
    est_iters = max(300, min(est_iters, 40000))
    
    T0 = 0.02
    T_end = 0.0003
    alpha = (T_end / T0) ** (1.0 / max(1, est_iters - 1))
    
    rng = np.random.default_rng(42)
    step_scale = 1.0
    
    for it in range(est_iters):
        if time.time() > time_budget:
            break
        
        T = T0 * (alpha ** it)
        # Adaptive step size: decrease over time
        noise = rng.normal(0, T * step_scale, size=(N, 2))
        candidate = current + noise
        candidate = np.clip(candidate, -5, 5)
        
        cand_score = _score(candidate)
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            if current_score > best_s:
                best_s = current_score
                best = current.copy()
            # Occasionally increase step scale if we're making progress
            if it % 50 == 0 and it > 0:
                step_scale = min(1.0, step_scale * 1.1)
        else:
            delta = current_score - cand_score
            if delta < 1e-12:
                current = candidate
                current_score = cand_score
            else:
                prob = np.exp(-delta / max(T, 1e-10))
                if rng.random() < prob:
                    current = candidate
                    current_score = cand_score
                    if current_score > best_s:
                        best_s = current_score
                        best = current.copy()
            # Decrease step scale if stuck
            if it % 100 == 0:
                step_scale = max(0.3, step_scale * 0.9)
    
    return best, best_s

def _fourier_optimize(deadline):
    """
    Optimize using Fourier-parameterized convex container.
    Returns best points found.
    """
    # Fourier parameters: [a0, a1, b1, a2, b2]
    # Point angles: 13 angles
    # Total: 18 parameters
    
    bounds = [
        (0.5, 2.0),  # a0 - base radius
        (-0.3, 0.3),  # a1 - cos(t) amplitude
        (-0.3, 0.3),  # b1 - sin(t) amplitude
        (-0.2, 0.2),  # a2 - cos(2t) amplitude
        (-0.2, 0.2),  # b2 - sin(2t) amplitude
    ] + [(0, 2 * np.pi)] * N  # 13 point angles
    
    try:
        de_result = differential_evolution(
            _fourier_de_objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=False,
            updating='deferred',
            workers=1,
            callback=lambda x, convergence: None,
            time_budget=deadline - time.time() - 10
        )
        
        if de_result.fun < 1e9:
            params = de_result.x
            container_params = params[:5]
            point_angles = params[5:]
            points = _optimize_points_in_convex_container(container_params, point_angles)
            points = points - np.mean(points, axis=0)
            return points
    except Exception:
        pass
    
    # Fallback: try a few random Fourier configurations
    rng = np.random.default_rng(999)
    best_points = None
    best_s = 0
    
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        a0 = rng.uniform(0.8, 1.5)
        a1 = rng.uniform(-0.2, 0.2)
        b1 = rng.uniform(-0.2, 0.2)
        a2 = rng.uniform(-0.15, 0.15)
        b2 = rng.uniform(-0.15, 0.15)
        container_params = np.array([a0, a1, b1, a2, b2])
        point_angles = rng.uniform(0, 2 * np.pi, N)
        
        try:
            points = _optimize_points_in_convex_container(container_params, point_angles)
            points = points - np.mean(points, axis=0)
            s = _score(points)
            if s > best_s:
                best_s = s
                best_points = points.copy()
        except Exception:
            continue
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    # === Phase 1: Fourier-parameterized convex container optimization ===
    if time.time() < deadline - 40:
        try:
            fourier_points = _fourier_optimize(deadline)
            if fourier_points is not None:
                _update_best(fourier_points)
        except Exception:
            pass
    
    # === Phase 2: Generate diverse initial configurations ===
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Two-ring configurations
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        if n_in < 3:
            continue
        for r_in in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            for offset in [0, np.pi / n_out, np.pi / (2 * n_out), 2*np.pi/(3*n_out)]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                inits.append(pts)
    
    # 3. Three-ring configurations
    for n1, n2, n3 in [(5, 4, 4), (6, 4, 3), (7, 3, 3), (4, 4, 5), (6, 5, 2), (5, 5, 3)]:
        if n1 + n2 + n3 != N:
            continue
        for r2 in [0.4, 0.5, 0.6]:
            for r3 in [0.15, 0.25, 0.35]:
                a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n1
                a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / n2
                pts = np.vstack([
                    np.column_stack([np.cos(a1), np.sin(a1)]),
                    np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
                    np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
                ])
                inits.append(pts)
    
    # 4. Random starts
    rng = np.random.default_rng(12345)
    for _ in range(15):
        r = np.sqrt(rng.random(N)) * 1.5
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)
    
    # 5. Elliptical configurations
    for aspect in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x = aspect * np.cos(angles)
        y = np.sin(angles)
        inits.append(np.column_stack([x, y]))
    
    # 6. If Fourier found something good, add its neighborhood
    if best_points is not None:
        rng_de = np.random.default_rng(777)
        for _ in range(5):
            perturbed = best_points + rng_de.normal(0, 0.1, size=(N, 2))
            inits.append(perturbed)
    
    # Shuffle inits
    rng2 = np.random.default_rng(999)
    indices = rng2.permutation(len(inits))
    inits = [inits[i] for i in indices]
    
    # === Phase 3: Local optimization from each start ===
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        x_opt, s_opt = _local_polish(x0, deadline)
        _update_best(x_opt)
        
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Init {i}/{len(inits)}: best_score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # === Phase 4: Simulated Annealing ===
    if best_points is not None and time.time() < deadline - 10:
        try:
            sa_best, sa_score = _simulated_annealing(best_points, deadline)
            _update_best(sa_best)
        except Exception:
            pass
    
    # === Phase 5: Final high-precision polish ===
    if best_points is not None and time.time() < deadline - 3:
        try:
            res_final = minimize(
                _objective_full,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            _update_best(res_final.x.reshape(N, 2))
        except Exception:
            pass
    
    # === Phase 6: Final coordinate descent ===
    if best_points is not None and time.time() < deadline - 1:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_points, deadline)
            _update_best(cd_pts)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
