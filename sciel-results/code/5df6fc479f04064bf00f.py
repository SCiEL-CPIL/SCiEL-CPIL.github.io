import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _compute_covering_radius(points):
    """
    Compute the covering radius as the maximum facet circumradius.
    """
    points = _normalize(points)
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return np.pi
    
    n_facets = len(hull.simplices)
    max_radius = 0.0
    
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        ab = np.dot(a, b)
        ac = np.dot(a, c)
        bc = np.dot(b, c)
        
        Gram = np.array([
            [1.0, ab, ac],
            [ab, 1.0, bc],
            [ac, bc, 1.0]
        ])
        
        try:
            m = np.linalg.solve(Gram, np.ones(3))
            c_dir = m[0] * a + m[1] * b + m[2] * c
            norm_c_dir = np.linalg.norm(c_dir)
            if norm_c_dir < 1e-10:
                continue
            c_center = c_dir / norm_c_dir
            cos_r = np.clip(np.dot(c_center, a), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > max_radius:
                max_radius = r
        except np.linalg.LinAlgError:
            continue
    
    return max_radius

def _soft_max_radius(points, beta=50.0):
    """
    Compute soft-max of facet radii for smooth optimization.
    """
    points = _normalize(points)
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        return np.pi, np.zeros_like(points)
    
    n_facets = len(hull.simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for i, simplex in enumerate(hull.simplices):
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        ab = np.dot(a, b)
        ac = np.dot(a, c)
        bc = np.dot(b, c)
        
        Gram = np.array([
            [1.0, ab, ac],
            [ab, 1.0, bc],
            [ac, bc, 1.0]
        ])
        
        try:
            m = np.linalg.solve(Gram, np.ones(3))
            c_dir = m[0] * a + m[1] * b + m[2] * c
            norm_c_dir = np.linalg.norm(c_dir)
            if norm_c_dir < 1e-10:
                continue
            centers[i] = c_dir / norm_c_dir
            cos_r = np.clip(np.dot(centers[i], a), -1.0, 1.0)
            radii[i] = np.arccos(cos_r)
        except np.linalg.LinAlgError:
            continue
    
    if np.all(radii == 0):
        return np.pi, np.zeros_like(points)
    
    max_r = np.max(radii)
    exp_terms = np.exp(beta * (radii - max_r))
    log_sum = np.log(np.sum(exp_terms))
    soft_radius = max_r + log_sum / beta
    
    # Compute gradient
    weights = exp_terms / np.sum(exp_terms)
    grads = np.zeros_like(points)
    
    for i in range(n_facets):
        if radii[i] < 1e-12 or weights[i] < 1e-15:
            continue
        w = weights[i]
        C = centers[i]
        r = radii[i]
        sin_r = np.sin(r)
        if sin_r < 1e-12:
            continue
        
        grad_contrib = -C / sin_r
        simplex = hull.simplices[i]
        for k in range(3):
            p_idx = simplex[k]
            p = points[p_idx]
            radial = np.dot(grad_contrib, p) * p
            tangential = grad_contrib - radial
            grads[p_idx] += w * tangential
    
    return soft_radius, grads

def _optimize_with_lbfgs(points, time_budget, beta=50.0):
    """
    Optimize using L-BFGS-B on the soft-max covering radius.
    """
    start_time = time.time()
    current_points = _normalize(points)
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    # Flatten for scipy optimization
    def objective(x):
        pts = _normalize(x.reshape(-1, 3))
        soft_r, _ = _soft_max_radius(pts, beta=beta)
        return soft_r
    
    x0 = current_points.flatten()
    
    # Use L-BFGS-B with time-aware termination
    class TimeAwareOptimizer:
        def __init__(self, deadline):
            self.deadline = deadline
            self.should_stop = False
        
        def should_terminate(self):
            if time.time() > self.deadline:
                self.should_stop = True
            return self.should_stop
    
    optimizer = TimeAwareOptimizer(start_time + time_budget - 2.0)
    
    result = scipy.optimize.minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={
            'maxiter': 2000,
            'maxfun': 5000,
            'gtol': 1e-10,
            'callback': lambda x: optimizer.should_terminate()
        }
    )
    
    if result.x is not None:
        new_points = _normalize(result.x.reshape(-1, 3))
        new_radius = _compute_covering_radius(new_points)
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points
    
    return best_points, best_radius

def _gradient_descent(points, time_budget, beta=50.0):
    """
    Gradient descent with adaptive step size on soft-max radius.
    """
    start_time = time.time()
    current_points = _normalize(points)
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    step_size = 0.05
    min_step = 1e-8
    decay = 0.92
    growth = 1.05
    
    while time.time() - start_time < time_budget - 1.0:
        soft_radius, grads = _soft_max_radius(current_points, beta=beta)
        
        true_radius = _compute_covering_radius(current_points)
        if true_radius < best_radius - 1e-12:
            best_radius = true_radius
            best_points = current_points.copy()
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm < 1e-14:
            break
        
        direction = grads / grad_norm
        new_points = _normalize(current_points + step_size * direction)
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.2)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _repulsive_optimize(points, time_budget, repulsive_strength=1.0, pressure_strength=2.0):
    """
    Optimize using repulsive forces and pressure from worst facets.
    """
    start_time = time.time()
    current_points = _normalize(points)
    best_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    
    step_size = 0.03
    min_step = 1e-8
    decay = 0.9
    growth = 1.08
    
    while time.time() - start_time < time_budget - 1.0:
        points = _normalize(current_points)
        try:
            hull = scipy.spatial.ConvexHull(points)
        except Exception:
            break
        
        n_facets = len(hull.simplices)
        radii = np.zeros(n_facets)
        centers = np.zeros((n_facets, 3))
        
        for i, simplex in enumerate(hull.simplices):
            a = points[simplex[0]]
            b = points[simplex[1]]
            c = points[simplex[2]]
            
            ab = np.dot(a, b)
            ac = np.dot(a, c)
            bc = np.dot(b, c)
            
            Gram = np.array([
                [1.0, ab, ac],
                [ab, 1.0, bc],
                [ac, bc, 1.0]
            ])
            
            try:
                m = np.linalg.solve(Gram, np.ones(3))
                c_dir = m[0] * a + m[1] * b + m[2] * c
                norm_c_dir = np.linalg.norm(c_dir)
                if norm_c_dir < 1e-10:
                    continue
                centers[i] = c_dir / norm_c_dir
                cos_r = np.clip(np.dot(centers[i], a), -1.0, 1.0)
                radii[i] = np.arccos(cos_r)
            except np.linalg.LinAlgError:
                continue
        
        if np.all(radii == 0):
            break
        
        true_radius = np.max(radii)
        if true_radius < best_radius - 1e-12:
            best_radius = true_radius
            best_points = current_points.copy()
        
        n = len(current_points)
        forces = np.zeros_like(current_points)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = current_points[i] - current_points[j]
                d = np.linalg.norm(d_vec)
                if d < 0.01:
                    d = 0.01
                d_hat = d_vec / d
                radial_i = np.dot(d_hat, current_points[i]) * current_points[i]
                tangential_i = d_hat - radial_i
                mag = 1.0 / (d ** 2)
                forces[i] += repulsive_strength * mag * tangential_i
                forces[j] -= repulsive_strength * mag * tangential_i
        
        # Pressure from worst facets
        sorted_idx = np.argsort(radii)[::-1]
        n_worst = min(5, len(sorted_idx))
        
        for idx in sorted_idx[:n_worst]:
            if radii[idx] < 1e-10:
                continue
            C = centers[idx]
            r = radii[idx]
            sin_r = np.sin(r)
            if sin_r < 1e-10:
                continue
            
            weight = (radii[idx] / np.max(radii)) ** 2
            
            for i in range(n):
                dot_ic = np.dot(current_points[i], C)
                ang_dist = np.arccos(np.clip(dot_ic, -1.0, 1.0))
                if ang_dist > r + 0.15:
                    continue
                
                radial_C = dot_ic * current_points[i]
                tangential_C = C - radial_C
                tangential_C_norm = np.linalg.norm(tangential_C)
                if tangential_C_norm < 1e-10:
                    continue
                tangential_C = tangential_C / tangential_C_norm
                
                strength = pressure_strength * weight * max(0, 1 - ang_dist / (r + 0.15))
                forces[i] += strength * tangential_C
        
        force_norm = np.linalg.norm(forces)
        if force_norm < 1e-14:
            break
        
        direction = forces / force_norm
        new_points = _normalize(current_points + step_size * direction)
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < true_radius:
            current_points = new_points
            step_size = min(step_size * growth, 0.15)
        else:
            step_size *= decay
            if step_size < min_step:
                break
    
    return best_points, best_radius

def _simulated_annealing(points, time_budget, T0=0.1, T_min=0.001, cooling=0.995):
    """
    Simulated annealing to escape local minima.
    """
    start_time = time.time()
    current_points = _normalize(points)
    current_radius = _compute_covering_radius(current_points)
    best_points = current_points.copy()
    best_radius = current_radius
    T = T0
    
    while time.time() - start_time < time_budget - 1.0:
        # Random perturbation
        rng = np.random.default_rng()
        perturbation = rng.normal(size=(N, 3)) * T
        new_points = _normalize(current_points + perturbation)
        new_radius = _compute_covering_radius(new_points)
        
        # Accept or reject
        delta = new_radius - current_radius
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_points = new_points
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
        
        T *= cooling
        if T < T_min:
            break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_radius = np.inf
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Fibonacci sphere
    j = np.arange(N) + 0.5
    z = 1.0 - 2.0 * j / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * j
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    seeds.append(fib)
    
    # 2. Random starts with different seeds
    for seed_val in [42, 1, 2, 3, 7, 13, 99, 123, 555, 777]:
        rng = np.random.default_rng(seed_val)
        pts = rng.normal(size=(N, 3))
        pts = _normalize(pts)
        seeds.append(pts)
    
    # 3. Icosahedron-based
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts = _normalize(ico_verts)
    rng_ico = np.random.default_rng(12345)
    extra = rng_ico.normal(size=(50 - len(ico_verts), 3))
    extra = _normalize(extra)
    ico_seed = _normalize(np.vstack([ico_verts, extra]))
    seeds.append(ico_seed)
    
    n_seeds = len(seeds)
    
    # Phase 1: Quick screening with gradient descent
    for i, seed_pts in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining < 15:
            break
        
        time_budget = min(8.0, remaining / (n_seeds - i))
        if time_budget < 2.0:
            break
        
        pts, rad = _gradient_descent(seed_pts, time_budget=time_budget, beta=50.0)
        if rad < best_radius:
            best_radius = rad
            best_points = pts.copy()
    
    # Phase 2: L-BFGS-B optimization on best candidates
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 30:
            budget = min(40.0, remaining - 10)
            if budget > 5:
                pts, rad = _optimize_with_lbfgs(best_points, time_budget=budget, beta=50.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    # Phase 3: Simulated annealing to explore
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 20:
            budget = min(25.0, remaining - 8)
            if budget > 5:
                pts, rad = _simulated_annealing(best_points, time_budget=budget, T0=0.05, T_min=0.0005, cooling=0.998)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    # Phase 4: Final polish with L-BFGS-B
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 15:
            budget = min(20.0, remaining - 5)
            if budget > 3:
                pts, rad = _optimize_with_lbfgs(best_points, time_budget=budget, beta=100.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    # Phase 5: Repulsive force polish
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 10:
            budget = min(12.0, remaining - 3)
            if budget > 2:
                pts, rad = _repulsive_optimize(best_points, time_budget=budget, repulsive_strength=0.5, pressure_strength=3.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    # Phase 6: Final fine gradient polish
    if best_points is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        
        if remaining > 5:
            budget = min(8.0, remaining - 2)
            if budget > 1:
                pts, rad = _gradient_descent(best_points, time_budget=budget, beta=200.0)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts.copy()
    
    if best_points is None:
        best_points = seeds[0]
    
    # Ensure output is normalized
    best_points = _normalize(best_points)
    
    return best_points
