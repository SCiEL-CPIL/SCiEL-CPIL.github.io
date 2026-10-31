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

def _convex_polygon_from_lines(angles, offsets):
    """
    Construct a convex polygon from N support lines.
    Each line i is defined by normal n_i = (cos(a_i), sin(a_i)) and offset d_i:
    n_i . x = d_i.
    The vertex v_i is the intersection of line i and line i+1 (mod N).
    For the polygon to be convex and contain the origin (or be valid),
    we assume angles are sorted and offsets are positive.
    """
    n = len(angles)
    if n < 3:
        return np.zeros((0, 2))
    
    # Sort by angle to ensure convexity
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    offsets_sorted = offsets[sort_idx]
    
    # Ensure angles are in [0, 2*pi)
    angles_sorted = angles_sorted % (2 * np.pi)
    
    # Calculate intersections
    vertices = np.zeros((n, 2))
    for i in range(n):
        j = (i + 1) % n
        
        n1 = np.array([np.cos(angles_sorted[i]), np.sin(angles_sorted[i])])
        d1 = offsets_sorted[i]
        
        n2 = np.array([np.cos(angles_sorted[j]), np.sin(angles_sorted[j])])
        d2 = offsets_sorted[j]
        
        # Solve:
        # n1_x * x + n1_y * y = d1
        # n2_x * x + n2_y * y = d2
        # Matrix form: A * [x, y]^T = [d1, d2]^T
        A = np.array([
            [n1[0], n1[1]],
            [n2[0], n2[1]]
        ])
        b = np.array([d1, d2])
        
        det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        if abs(det) < 1e-10:
            # Parallel lines, skip or handle
            vertices[i] = vertices[(i-1)%n] # Fallback
        else:
            inv_det = 1.0 / det
            vertices[i, 0] = (b[0]*A[1,1] - b[1]*A[0,1]) * inv_det
            vertices[i, 1] = (A[0,0]*b[1] - A[1,0]*b[0]) * inv_det
            
    return vertices

def _line_parameter_objective(params):
    """
    Objective for optimizing support line parameters.
    params: [angles (N), offsets (N)]
    """
    angles = params[:N]
    offsets = params[N:]
    
    # Ensure offsets are positive enough to form a valid polygon around origin
    # If an offset is <= 0, the line passes through or behind origin,
    # which might make the polygon non-convex or unbounded in this context.
    # We map offsets to [0.1, 5.0] via tanh or similar, but here we just take raw
    # and penalize if invalid.
    
    # Simple validity check:
    # The polygon must contain the origin? Not necessarily, but for max min area,
    # it should be "fat".
    # Let's just compute and score.
    
    points = _convex_polygon_from_lines(angles, offsets)
    
    if len(points) < 3:
        return 1e10
        
    # Check if points are distinct and form a proper convex hull
    # If the "polygon" is self-intersecting or degenerate, score will be low.
    
    # Center the points for better numerical behavior in scoring?
    # The score is invariant to translation, so centering is good.
    points = points - np.mean(points, axis=0)
    
    # Check for duplicate points
    if np.any(np.diff(np.sort(points[:,0])) < 1e-8) or np.any(np.diff(np.sort(points[:,1])) < 1e-8):
         # This check is too strict, let's rely on _score
         pass

    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _polar_to_cartesian(polar):
    """Convert (angles, radii) to Cartesian points.
    polar shape: (2N,) -> angles[0:N], radii[N:2N]
    """
    angles = polar[:N]
    radii = polar[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _polar_objective(polar):
    """Objective for polar-parameterized DE: minimize negative score."""
    points = _polar_to_cartesian(polar)
    # Center the points
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

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

def _cmaes_polish(points, deadline, max_iter=300):
    """CMA-ES style local optimization for fine-tuning."""
    import cma
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    x0 = best_pts.flatten()
    sigma0 = 0.02
    opts = {
        'maxiter': max_iter,
        'tolopt': 1e-12,
        'tolfun': 1e-12,
        'verbose': -9,
        'bounds': [-10, 10],
        'seed': 42
    }
    
    try:
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop() and time.time() < deadline - 1:
            solutions = es.ask()
            fitnesses = []
            for sol in solutions:
                pts = sol.reshape(N, 2)
                s = _score(pts)
                if s > 0:
                    fitnesses.append(-s)
                else:
                    fitnesses.append(1e10)
                if s > best_s:
                    best_s = s
                    best_pts = pts.copy()
            es.tell(solutions, fitnesses)
    except Exception:
        pass
    
    return best_pts, best_s

def _affine_grid_objective(params):
    """Optimize affine transform of a base grid configuration."""
    # params: [a, b, c, d, tx, ty] for affine transform
    a, b, c, d, tx, ty = params
    
    # Base grid: 13 points in a structured arrangement
    # Using a 4x4 grid with center removed (16-1=15, take 13)
    # Or use a hexagonal lattice pattern
    # Let's use: center + ring of 6 + ring of 6 (but that's 13)
    base = _make_ring_config(6, 6, 0.5, 0)  # 6 outer, 6 inner
    # Add center
    base = np.vstack([np.array([[0.0, 0.0]]), base])
    
    # Apply affine transform
    x = a * base[:, 0] + b * base[:, 1] + tx
    y = c * base[:, 0] + d * base[:, 1] + ty
    points = np.column_stack([x, y])
    
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

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

def _cartesian_to_polar(points):
    """Convert Cartesian points to polar representation (sorted by angle)."""
    points = points - np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1], points[:, 0])
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    # Sort by angle
    order = np.argsort(angles)
    angles_sorted = angles[order]
    radii_sorted = radii[order]
    # Normalize angles to [0, 2*pi)
    angles_sorted = angles_sorted % (2 * np.pi)
    return np.concatenate([angles_sorted, radii_sorted])

def _coordinate_descent_polish(points, deadline, max_iter=5000):
    """Coordinate descent on the hard objective with adaptive step sizes."""
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    # Adaptive step sizes that shrink as we converge
    base_steps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    step_idx = 0  # Start with larger steps
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try perturbations in current step range
                for delta in [base_steps[step_idx], -base_steps[step_idx], 
                              base_steps[step_idx] * 0.5, -base_steps[step_idx] * 0.5]:
                    trial = best_pts.copy()
                    trial[i, coord] += delta
                    trial_s = _score(trial)
                    if trial_s > best_s + 1e-14:
                        best_pts = trial
                        best_s = trial_s
                        improved = True
                        break
        
        if not improved:
            # Shrink step size if no improvement
            if step_idx < len(base_steps) - 1:
                step_idx += 1
            else:
                break
    
    return best_pts, best_s

def _local_polish(points, deadline):
    """Multi-phase local optimization from a starting point."""
    best_x = points.flatten().copy()
    best_s = _score(points)
    
    # Phase 1: Powell on smooth objective (log-sum-exp based) with higher p
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _smooth_objective(x, p=150),
                best_x,
                method='Powell',
                options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-11}
            )
            s = _score(res.x.reshape(N, 2))
            if s > best_s:
                best_s = s
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on full objective
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_full,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True}
            )
            s_nm = _score(res_nm.x.reshape(N, 2))
            if s_nm > best_s:
                best_s = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Phase 3: CMA-ES polish if available
    if time.time() < deadline - 3:
        try:
            cma_pts, cma_s = _cmaes_polish(best_x.reshape(N, 2), deadline)
            if cma_s > best_s:
                best_s = cma_s
                best_x = cma_pts.flatten().copy()
        except Exception:
            pass
    
    # Phase 4: Coordinate descent polish on hard objective
    if time.time() < deadline - 1:
        try:
            cd_pts, cd_s = _coordinate_descent_polish(best_x.reshape(N, 2), deadline)
            if cd_s > best_s:
                best_s = cd_s
                best_x = cd_pts.flatten().copy()
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
    est_iters = int(total_time * 800)  # More iterations with faster eval
    est_iters = max(500, min(est_iters, 60000))
    
    T0 = 0.03
    T_end = 0.00005
    alpha = (T_end / T0) ** (1.0 / max(1, est_iters - 1))
    
    rng = np.random.default_rng(42)
    
    # Track acceptance rate to adapt step size
    accepted = 0
    total = 0
    step_scale = 0.5  # Start smaller for more focused search
    
    for it in range(est_iters):
        if time.time() > time_budget:
            break
        
        T = T0 * (alpha ** it)
        noise = rng.normal(0, T * step_scale, size=(N, 2))
        candidate = current + noise
        candidate = np.clip(candidate, -5, 5)
        
        cand_score = _score(candidate)
        total += 1
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            accepted += 1
            if current_score > best_s:
                best_s = current_score
                best = current.copy()
        else:
            delta = current_score - cand_score
            if delta < 1e-12:
                current = candidate
                current_score = cand_score
                accepted += 1
            else:
                prob = np.exp(-delta / max(T, 1e-10))
                if rng.random() < prob:
                    current = candidate
                    current_score = cand_score
                    accepted += 1
        
        # Adapt step size every 500 iterations for faster adaptation
        if total % 500 == 0 and total > 0:
            acc_rate = accepted / total
            if acc_rate < 0.15:
                step_scale *= 0.7
            elif acc_rate > 0.4:
                step_scale *= 1.2
            step_scale = np.clip(step_scale, 0.05, 3.0)
            accepted = 0
            total = 0
    
    return best, best_s

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

    # === Phase 0: Structural Search in Support Line Space ===
    # This is the new structural change.
    # We represent the convex hull by its 13 edges (support lines).
    # Params: 13 angles, 13 offsets.
    # Angles in [0, 2*pi), Offsets in [0.1, 5.0]
    if time.time() < deadline - 30:
        bounds = [(0, 2 * np.pi)] * N + [(0.1, 5.0)] * N
        try:
            de_line_result = differential_evolution(
                _line_parameter_objective,
                bounds=bounds,
                seed=123,
                maxiter=50, # Fewer iterations, higher popsize
                popsize=40,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                polish=False,
                updating='deferred',
                workers=1
            )
            if de_line_result.fun < 1e9:
                # Reconstruct points from best line params
                best_params = de_line_result.x
                angles = best_params[:N]
                offsets = best_params[N:]
                line_points = _convex_polygon_from_lines(angles, offsets)
                line_points = line_points - np.mean(line_points, axis=0)
                _update_best(line_points)
                
                # Polish this result
                if best_points is not None and time.time() < deadline - 20:
                    polished, _ = _local_polish(best_points, deadline)
                    _update_best(polished)
        except Exception:
            pass

    # === Phase 1: Differential Evolution in polar space ===
    # Polar representation: 13 angles in [0, 2*pi), 13 radii in [0.1, 3.0]
    if time.time() < deadline - 40:
        bounds = [(0, 2 * np.pi)] * N + [(0.05, 3.0)] * N
        try:
            de_result = differential_evolution(
                _polar_objective,
                bounds=bounds,
                seed=42,
                maxiter=100,
                popsize=30,
                tol=1e-7,
                mutation=(0.4, 1.2),
                recombination=0.8,
                polish=False,
                updating='deferred',
                workers=1
            )
            if de_result.success or de_result.fun < 1e9:
                de_points = _polar_to_cartesian(de_result.x)
                de_points = de_points - np.mean(de_points, axis=0)
                _update_best(de_points)
        except Exception:
            pass
    
    # === Phase 1b: Affine grid optimization ===
    if time.time() < deadline - 30:
        try:
            # Optimize affine transform parameters
            aff_result = minimize(
                _affine_grid_objective,
                x0=[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            # Reconstruct points from best affine params
            a, b, c, d, tx, ty = aff_result.x
            base = _make_ring_config(6, 6, 0.5, 0)
            base = np.vstack([np.array([[0.0, 0.0]]), base])
            x = a * base[:, 0] + b * base[:, 1] + tx
            y = c * base[:, 0] + d * base[:, 1] + ty
            aff_points = np.column_stack([x, y])
            _update_best(aff_points)
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
    for _ in range(20):
        r = np.sqrt(rng.random(N)) * 1.5
        theta = rng.random(N) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)
    
    # 5. Hexagonal lattice-based starts
    # 1+6+6 hexagonal pattern
    center = np.array([[0.0, 0.0]])
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring2 = np.column_stack([2.0 * np.cos(angles2), 2.0 * np.sin(angles2)])
    inits.append(np.vstack([center, ring1, ring2]))
    
    # Rotated hexagonal
    rot = np.pi / 6
    angles1r = angles1 + rot
    ring1r = np.column_stack([np.cos(angles1r), np.sin(angles1r)])
    angles2r = angles2 + rot
    ring2r = np.column_stack([2.0 * np.cos(angles2r), 2.0 * np.sin(angles2r)])
    inits.append(np.vstack([center, ring1r, ring2r]))
    
    # 6. If DE found something good, add its neighborhood
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
        if time.time() > deadline - 20:
            break
        
        x_opt, s_opt = _local_polish(x0, deadline)
        _update_best(x_opt)
        
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Init {i}/{len(inits)}: best_score={best_score:.6f}, elapsed={elapsed:.1f}s", flush=True)
    
    # === Phase 4: Simulated Annealing ===
    if best_points is not None and time.time() < deadline - 5:
        try:
            sa_best, sa_score = _simulated_annealing(best_points, deadline)
            _update_best(sa_best)
        except Exception:
            pass
    
    # === Phase 5: Final high-precision polish ===
    if best_points is not None and time.time() < deadline - 2:
        try:
            # Multiple Nelder-Mead restarts with slightly perturbed starts
            rng_final = np.random.default_rng(888)
            for attempt in range(5):
                if time.time() > deadline - 1:
                    break
                if attempt == 0:
                    start_x = best_points.flatten()
                else:
                    start_x = best_points.flatten() + rng_final.normal(0, 0.003, size=N*2)
                
                res_final = minimize(
                    _objective_full,
                    start_x,
                    method='Nelder-Mead',
                    options={'maxiter': 2500, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True}
                )
                _update_best(res_final.x.reshape(N, 2))
                
                # Add CMA-ES polish after each NM run
                if time.time() < deadline - 1.5:
                    try:
                        cma_pts, cma_s = _cmaes_polish(best_points, deadline)
                        _update_best(cma_pts)
                    except Exception:
                        pass
                
                # Add coordinate descent after each NM run
                if time.time() < deadline - 0.5:
                    cd_pts, cd_s = _coordinate_descent_polish(best_points, deadline)
                    _update_best(cd_pts)
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
