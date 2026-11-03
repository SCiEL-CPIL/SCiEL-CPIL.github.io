import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_MARGIN = 15.0  # seconds before hard limit


def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    """Compute the packing score."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return max(min(min_pair, min_boundary), 0.0)


def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp."""
    neg = -values * beta
    neg_max = np.max(neg)
    exp_sum = np.sum(np.exp(neg - neg_max))
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta=50.0):
    """Smooth objective: minimize -softmin of all constraint margins."""
    pts = x.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 0.0
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _ring_initialization(ring_radii, ring_counts, angles_offset=None):
    """Generate points from ring structure."""
    centers = []
    for idx, (count, radius) in enumerate(zip(ring_counts, ring_radii)):
        if count == 1 and radius < 1e-10:
            centers.append(np.array([0.0, 0.0]))
        else:
            if angles_offset is not None and idx < len(angles_offset):
                offset = angles_offset[idx]
            else:
                offset = 0.0
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            for i in range(count):
                centers.append(np.array([xs[i], ys[i]]))
    pts = np.array(centers[:N])
    # Project inside unit circle
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 0.999:
            pts[i] *= 0.999 / norms[i]
    return pts


def _optimize_from_init(pts, deadline, max_beta_stages=5):
    """Multi-stage annealing optimization from initial points."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    betas = [5.0, 15.0, 40.0, 100.0, 250.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11, 'maxfun': 5000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # High-precision polish
    for beta in [500.0, 1000.0, 2000.0, 5000.0]:
        if time.time() > deadline - 5:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 20000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    return current_pts, current_score


def _binary_search_polish(pts, r_target, deadline, max_iters=500):
    """Polish by iteratively pushing points to achieve target radius r_target."""
    pts = pts.copy()
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        # Compute current margins
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        # Find the most violated constraint
        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)
        
        if min_pair_margin < min_boundary_margin - 1e-14:
            # Pair constraint is binding
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            # Boundary constraint is binding
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                # Push outward from center
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        
        # Check convergence
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _de_ring_optimize(deadline):
    """Differential evolution over ring parameters."""
    ring_counts = [1, 6, 12, 18, 13]
    # Parameters: [r1, r2, r3, r4, offset1, offset2, offset3, offset4]
    # r1..r4 are ring radii, offsets are angular offsets
    
    def ring_to_pts(params):
        radii = params[:4]
        offsets = params[4:]
        # Ensure radii are positive and ordered
        radii = np.maximum(radii, 0.05)
        radii = np.sort(radii)
        # Ensure outermost ring fits
        max_allowed = 1.0 - 0.12  # rough bound
        radii = np.minimum(radii, max_allowed)
        return _ring_initialization(radii, ring_counts, offsets)
    
    def neg_score(params):
        pts = ring_to_pts(params)
        return -_score(pts)
    
    # DE bounds
    bounds = [
        (0.15, 0.35),   # r1
        (0.30, 0.55),   # r2
        (0.45, 0.70),   # r3
        (0.60, 0.90),   # r4
        (0, 2*np.pi),   # offset1
        (0, 2*np.pi),   # offset2
        (0, 2*np.pi),   # offset3
        (0, 2*np.pi),   # offset4,
    ]
    
    best_pts = None
    best_score = 0.0
    rng = np.random.RandomState(42)
    
    # Run DE with limited generations
    pop_size = 30
    max_gen = 40
    F = 0.8
    CR = 0.9
    
    # Initial population
    pop = []
    for _ in range(pop_size):
        params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        pop.append(params)
    pop = np.array(pop)
    
    # Evaluate initial population
    scores = np.array([neg_score(p) for p in pop])
    
    for gen in range(max_gen):
        if time.time() > deadline - 10:
            break
        
        for i in range(pop_size):
            if time.time() > deadline - 10:
                break
            
            # Select three distinct random individuals
            a, b, c = rng.choice(pop_size, 3, replace=False)
            
            # Create trial vector
            v = pop[a] + F * (pop[b] - pop[c])
            for j in range(len(bounds)):
                if rng.random() < CR:
                    v[j] = pop[i][j]
                # Bound check
                v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])
            
            # Evaluate trial vector
            trial_score = neg_score(v)
            
            # Selection
            if trial_score < scores[i]:
                pop[i] = v
                scores[i] = trial_score
        
        # Track best
        best_idx = np.argmin(scores)
        if scores[best_idx] < -best_score:
            best_score = -scores[best_idx]
            best_pts = ring_to_pts(pop[best_idx])
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = 150.0 - DEADLINE_MARGIN  # 135s effective deadline
    
    # Start with ring structure initialization
    ring_counts = [1, 6, 12, 18, 13]
    # Initial radii based on known optimal structure
    ring_radii = [0.0, 0.25, 0.50, 0.75, 0.90]
    initial_pts = _ring_initialization(ring_radii, ring_counts)
    
    # Optimize from initial ring structure
    best_pts, best_score = _optimize_from_init(initial_pts, deadline)
    
    # Try DE over ring parameters
    if time.time() < deadline - 30:
        de_pts, de_score = _de_ring_optimize(deadline)
        if de_pts is not None and de_score > best_score:
            best_pts = de_pts
            best_score = de_score
            # Polish the DE result
            best_pts, best_score = _optimize_from_init(best_pts, deadline)
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        best_pts, best_score = _optimize_from_init(best_pts, deadline)
    
    # Binary search polish for final precision
    if time.time() < deadline - 5:
        # Try to push slightly above current score
        target_r = best_score * 1.001
        polished_pts = _binary_search_polish(best_pts, target_r, deadline)
        polished_score = _score(polished_pts)
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score
    
    # Final validation
    final_score = _score(best_pts)
    if final_score <= 0:
        # Fallback to simple hex lattice
        spacing = 0.25
        while True:
            points = []
            for i in range(-5, 6):
                for j in range(-5, 6):
                    x = i * spacing + (j % 2) * spacing * 0.5
                    y = j * spacing * np.sqrt(3) / 2
                    if x*x + y*y <= (1 - spacing/2)**2:
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= N:
                dists = np.linalg.norm(points, axis=1)
                idx = np.argsort(dists)[:N]
                return points[idx]
            spacing *= 0.95
    
    return best_pts
