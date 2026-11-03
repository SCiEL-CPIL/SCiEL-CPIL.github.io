import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def _pairwise_distances(pts):
    """Compute all pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

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

def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp for numerical stability."""
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    # Clip to avoid overflow
    neg_clipped = np.clip(neg - neg_max, -500, 500)
    exp_sum = np.sum(np.exp(neg_clipped))
    if exp_sum <= 0:
        return np.min(values)
    return -(neg_max + np.log(exp_sum)) / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth objective: minimize -softmin of all constraint margins.
    
    Margins:
    - Boundary: f_i = 1 - |c_i|
    - Pairwise: g_ij = |c_i - c_j| / 2
    
    r = min(all margins)
    We minimize -r using soft-min approximation.
    """
    pts = x.reshape(N, 2)
    
    # Boundary margins
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Pairwise margins
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    pair_half = d / 2.0
    
    # Combine all margins
    all_vals = np.concatenate([boundary.ravel(), pair_half.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    if len(all_vals) == 0:
        return 0.0
    
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min

def _smooth_objective_and_grad(x, beta=50.0):
    """
    Smooth objective with analytical gradient for L-BFGS-B.
    """
    pts = x.reshape(N, 2)
    
    # Boundary margins
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Pairwise margins
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    pair_half = d / 2.0
    
    # Combine all margins
    all_vals = np.concatenate([boundary.ravel(), pair_half.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    if len(all_vals) == 0:
        return 0.0, np.zeros_like(x)
    
    # Soft-min computation
    neg = -all_vals * beta
    neg_max = np.max(neg)
    neg_clipped = np.clip(neg - neg_max, -500, 500)
    exp_vals = np.exp(neg_clipped)
    exp_sum = np.sum(exp_vals)
    
    if exp_sum <= 0:
        soft_min = np.min(all_vals)
        # Find argmin for gradient
        min_idx = np.argmin(all_vals)
        weights = np.zeros_like(all_vals)
        weights[min_idx] = 1.0
    else:
        soft_min = -(neg_max + np.log(exp_sum)) / beta
        weights = exp_vals / exp_sum
    
    # Gradient: d(-soft_min)/dx = -d(soft_min)/dx
    # soft_min ≈ weighted average of margins
    # d(soft_min)/dx = sum_k weights[k] * d(margin_k)/dx
    # So d(-soft_min)/dx = -sum_k weights[k] * d(margin_k)/dx
    
    grad = np.zeros_like(x)
    
    # Boundary gradients: f_i = 1 - |c_i|
    # d f_i / d c_i = -c_i / |c_i|
    for i in range(N):
        w = weights[i]
        if w < 1e-15:
            continue
        if norms[i] > 1e-10:
            grad[2*i:2*i+2] += -w * (-pts[i] / norms[i])
        else:
            grad[2*i:2*i+2] += 0.0
    
    # Pairwise gradients: g_ij = |c_i - c_j| / 2
    # d g_ij / d c_i = (c_i - c_j) / (2 * |c_i - c_j|)
    # d g_ij / d c_j = (c_j - c_i) / (2 * |c_i - c_j|)
    
    # Index mapping: boundary is 0..N-1, pair_half is N..N+N*N-1
    # pair_half[i*N + j] corresponds to distance between i and j
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            idx = N + i * N + j
            w = weights[idx]
            if w < 1e-15:
                continue
            dist = d[i, j]
            if dist > 1e-10:
                vec = pts[i] - pts[j]
                unit_vec = vec / dist
                # d g_ij / d c_i = unit_vec / 2
                # d g_ij / d c_j = -unit_vec / 2
                grad[2*i:2*i+2] += -w * (unit_vec / 2.0)
                grad[2*j:2*j+2] += -w * (-unit_vec / 2.0)
    
    obj = -soft_min
    return obj, grad

def initial_rings():
    """Generate initial points using concentric rings matching known optimal topology."""
    points = np.zeros((N, 2))
    
    # Center point
    points[0] = [0, 0]
    count = 1
    
    # Ring structure: 1 center, 6, 12, 18, 13 = 50 total
    ring_radii = [0.0, 0.25, 0.5, 0.75, 0.92]
    ring_counts = [1, 6, 12, 18, 13]
    
    for r, n in zip(ring_radii, ring_counts):
        for k in range(n):
            if count >= N:
                break
            angle = 2 * np.pi * k / n
            # Alternate rings get offset
            ring_idx = ring_counts[:ring_counts.index(n)+1].count(n)
            if ring_idx % 2 == 0:
                angle += np.pi / n
            points[count, 0] = r * np.cos(angle)
            points[count, 1] = r * np.sin(angle)
            count += 1
    
    # Add small random perturbation to break symmetry
    points += np.random.normal(0, 0.01, size=(N, 2))
    
    # Ensure inside unit circle
    norms = np.linalg.norm(points, axis=1)
    for i in range(N):
        if norms[i] > 0.98:
            points[i] *= 0.98 / norms[i]
    
    return points

def initial_hexagonal():
    """Generate initial points using a hexagonal lattice pattern."""
    spacing = 0.24
    points = []
    
    max_k = 8
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x**2 + y**2 < 0.92:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) >= N:
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        selected = points[indices[:N]]
    else:
        needed = N - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, size=(needed, 2))
        selected = np.vstack([points, random_pts])
    
    return selected

def initial_random():
    """Generate random initial points inside the unit circle."""
    points = []
    while len(points) < N:
        x, y = np.random.uniform(-1, 1, 2)
        if x**2 + y**2 < 0.85:
            points.append([x, y])
    return np.array(points)

def _lbfbsb_optimize(x0, time_limit=25.0, beta=50.0, maxiter=200):
    """Run L-BFGS-B optimization on the smooth objective with gradient."""
    start_time = time.time()
    
    def fun_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _smooth_objective_and_grad(x, beta=beta)
    
    try:
        result = minimize(fun_and_grad, x0, method='L-BFGS-B', 
                         options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10,
                                  'maxfun': 5000})
        if result.x is not None and np.all(np.isfinite(result.x)):
            return result.x
    except Exception:
        pass
    
    return x0

def _multi_stage_anneal(pts, deadline):
    """
    Multi-stage annealing: start with low beta (smooth landscape) and increase
    to high beta (sharp minimum) for precise convergence.
    """
    current_pts = pts.copy()
    current_score = compute_radius(current_pts)
    
    # Stages: low beta for exploration, high beta for exploitation
    # Added higher betas for sharper convergence
    betas = [10.0, 30.0, 80.0, 200.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        time_left = deadline - time.time() - 8
        if time_left < 3:
            break
        
        x0 = current_pts.flatten()
        optimized = _lbfbsb_optimize(x0, time_limit=min(time_left, 20.0), beta=beta, maxiter=200)
        
        opt_pts = optimized.reshape(N, 2)
        opt_score = compute_radius(opt_pts)
        
        if opt_score > current_score:
            current_pts = opt_pts
            current_score = opt_score
    
    return current_pts, current_score

def _push_apart(points, target_r, iterations=200):
    """
    Iterative constraint projection: push overlapping circles apart and
    boundary-violating circles inward until all constraints are satisfied.
    """
    points = points.copy()
    
    for _ in range(iterations):
        moved = False
        
        # Pairwise constraints
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                min_dist = 2 * target_r
                
                if dist < min_dist:
                    if dist > 1e-10:
                        push = (min_dist - dist) / 2.0
                        direction = diff / dist
                        points[i] += direction * push
                        points[j] -= direction * push
                    else:
                        # Coincident points: separate randomly
                        rand_dir = np.random.normal(0, 1, 2)
                        rand_dir /= np.linalg.norm(rand_dir)
                        points[i] += rand_dir * target_r
                        points[j] -= rand_dir * target_r
                    moved = True
        
        # Boundary constraints
        for i in range(N):
            norm = np.linalg.norm(points[i])
            max_norm = 1.0 - target_r
            if norm > max_norm:
                if norm > 1e-10:
                    points[i] = points[i] * max_norm / norm
                else:
                    points[i] = np.array([max_norm, 0.0])
                moved = True
        
        if not moved:
            break
    
    # Final boundary enforcement
    for i in range(N):
        norm = np.linalg.norm(points[i])
        max_norm = 1.0 - target_r
        if norm > max_norm and norm > 1e-10:
            points[i] = points[i] * max_norm / norm
    
    return points

def _binary_search_polish(pts, deadline):
    """
    Binary search on radius with constraint projection to find the maximum
    feasible radius.
    """
    current_pts = pts.copy()
    current_r = compute_radius(current_pts)
    
    # Binary search bounds
    # Start with a wider range to ensure we don't miss improvements
    r_low = current_r * 0.95
    r_high = current_r * 1.05 + 0.02
    
    # First, try to expand
    # Increase iterations for better convergence
    for _ in range(25):
        if time.time() > deadline - 5:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to achieve radius r_mid
        # Increase push_apart iterations for better feasibility check
        trial_pts = _push_apart(current_pts, r_mid, iterations=500)
        trial_r = compute_radius(trial_pts)
        
        if trial_r >= r_mid - 1e-10:
            # Success: r_mid is feasible
            current_pts = trial_pts
            current_r = trial_r
            r_low = r_mid
        else:
            # Fail: r_mid is not feasible
            r_high = r_mid
        
        if r_high - r_low < 1e-12:
            break
    
    # Final polish with very high beta
    if time.time() < deadline - 5:
        time_left = deadline - time.time() - 3
        if time_left > 3:
            x0 = current_pts.flatten()
            # Use even higher beta for final precision
            optimized = _lbfbsb_optimize(x0, time_limit=min(time_left, 15.0), beta=5000.0, maxiter=300)
            opt_pts = optimized.reshape(N, 2)
            opt_r = compute_radius(opt_pts)
            if opt_r > current_r:
                current_pts = opt_pts
                current_r = opt_r
    
    return current_pts, current_r

def _de_ring_optimize(deadline):
    """
    Differential evolution over ring parameters to find good ring configurations.
    Parameters: ring radii and angular offsets.
    """
    ring_counts = [1, 6, 12, 18, 13]
    
    def ring_to_pts(params):
        radii = params[:4]
        offsets = params[4:]
        
        # Ensure radii are positive and ordered
        radii = np.maximum(radii, 0.05)
        radii = np.sort(radii)
        
        # Ensure outermost ring fits
        max_allowed = 1.0 - 0.12
        radii = np.minimum(radii, max_allowed)
        
        points = []
        # Center
        points.append(np.array([0.0, 0.0]))
        
        for idx, (count, radius) in enumerate(zip(ring_counts[1:], radii)):
            offset = offsets[idx] if idx < len(offsets) else 0.0
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            for k in range(count):
                x = radius * np.cos(angles[k])
                y = radius * np.sin(angles[k])
                points.append(np.array([x, y]))
        
        pts = np.array(points[:N])
        
        # Project inside unit circle
        norms = np.linalg.norm(pts, axis=1)
        for i in range(len(pts)):
            if norms[i] > 0.999:
                pts[i] *= 0.999 / norms[i]
        
        return pts
    
    def neg_score(params):
        pts = ring_to_pts(params)
        return -compute_radius(pts)
    
    # DE bounds
    bounds = [
        (0.15, 0.35),   # r1
        (0.30, 0.55),   # r2
        (0.45, 0.70),   # r3
        (0.60, 0.90),   # r4
        (0, 2*np.pi),   # offset1
        (0, 2*np.pi),   # offset2
        (0, 2*np.pi),   # offset3
        (0, 2*np.pi),   # offset4
    ]
    
    rng = np.random.RandomState(42)
    
    # Run DE with limited budget
    pop_size = 25
    max_gen = 30
    F = 0.7
    CR = 0.8
    
    # Initial population
    pop = np.array([rng.uniform(b[0], b[1]) for b in bounds])
    pop = np.array([np.array([rng.uniform(b[0], b[1]) for b in bounds]) for _ in range(pop_size)])
    
    # Evaluate initial population
    scores = np.array([neg_score(p) for p in pop])
    best_idx = np.argmin(scores)
    best_params = pop[best_idx].copy()
    best_score = scores[best_idx]
    
    for gen in range(max_gen):
        if time.time() > deadline - 15:
            break
        
        for i in range(pop_size):
            # Select three distinct random individuals
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            
            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            
            # Clamp to bounds
            for k, (lo, hi) in enumerate(bounds):
                mutant[k] = np.clip(mutant[k], lo, hi)
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(0, len(bounds))
            for k in range(len(bounds)):
                if rng.random() < CR or k == j_rand:
                    trial[k] = mutant[k]
            
            # Selection
            trial_score = neg_score(trial)
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_params = trial.copy()
    
    best_pts = ring_to_pts(best_params)
    return best_pts

def construct():
    deadline = time.time() + DEADLINE
    
    np.random.seed(42)
    
    best_r = 0.0
    best_points = None
    
    def update_best(pts, r):
        nonlocal best_r, best_points
        if r > best_r:
            best_r = r
            best_points = pts.copy()
    
    # Phase 1: Generate diverse initializations
    initializations = [
        initial_rings(),
        initial_hexagonal(),
        initial_random(),
    ]
    
    # Phase 2: For each initialization, run multi-stage annealing
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 25:
            break
        
        current_pts = init_pts.copy()
        current_r = compute_radius(current_pts)
        update_best(current_pts, current_r)
        
        # Multi-stage annealing
        annealed_pts, annealed_r = _multi_stage_anneal(current_pts, deadline)
        update_best(annealed_pts, annealed_r)
        
        # Push-apart refinement
        if time.time() < deadline - 15:
            target_r = max(annealed_r, 0.12)
            pushed = _push_apart(annealed_pts, target_r, iterations=500)
            pushed_r = compute_radius(pushed)
            update_best(pushed, pushed_r)
            
            # Binary search polish on the best so far
            if pushed_r > annealed_r:
                annealed_pts = pushed
                annealed_r = pushed_r
    
    # Phase 3: Differential evolution over ring parameters
    if time.time() < deadline - 30:
        de_pts = _de_ring_optimize(deadline)
        de_r = compute_radius(de_pts)
        update_best(de_pts, de_r)
        
        if de_r > best_r * 0.95:  # If DE found something competitive
            # Polish the DE result
            if time.time() < deadline - 20:
                annealed_pts, annealed_r = _multi_stage_anneal(de_pts, deadline)
                update_best(annealed_pts, annealed_r)
                
                if time.time() < deadline - 15:
                    target_r = max(annealed_r, 0.12)
                    pushed = _push_apart(annealed_pts, target_r, iterations=500)
                    pushed_r = compute_radius(pushed)
                    update_best(pushed, pushed_r)
    
    # Phase 4: Final binary search polish on the best solution
    if best_points is not None and time.time() < deadline - 10:
        polished_pts, polished_r = _binary_search_polish(best_points, deadline)
        update_best(polished_pts, polished_r)
    
    # Phase 5: Final high-precision L-BFGS-B polish
    if best_points is not None and time.time() < deadline - 5:
        time_left = deadline - time.time() - 3
        if time_left > 3:
            x0 = best_points.flatten()
            # Use extremely high beta for final precision
            optimized = _lbfbsb_optimize(x0, time_limit=min(time_left, 12.0), beta=10000.0, maxiter=500)
            opt_pts = optimized.reshape(N, 2)
            opt_r = compute_radius(opt_pts)
            update_best(opt_pts, opt_r)
    
    # Ensure validity
    if best_points is None:
        best_points = initial_rings()
    
    # Final validation and boundary enforcement
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] > 0.9999:
            best_points[i] *= 0.9999 / norms[i]
    
    # Verify no NaN or Inf
    if not np.all(np.isfinite(best_points)):
        best_points = initial_rings()
        norms = np.linalg.norm(best_points, axis=1)
        for i in range(N):
            if norms[i] > 0.9999:
                best_points[i] *= 0.9999 / norms[i]
    
    return best_points


if __name__ == "__main__":
    start = time.time()
    pts = construct()
    elapsed = time.time() - start
    r = compute_radius(pts)
    print(f"Radius: {r:.10f}")
    print(f"Shape: {pts.shape}")
    print(f"Time: {elapsed:.1f}s")
