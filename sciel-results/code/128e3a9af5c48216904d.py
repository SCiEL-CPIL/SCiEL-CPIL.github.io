"""Packing 50 equal circles in a unit circle via ring-based parametric optimization + smooth polish + constraint push."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
SEED = 42
DEADLINE_MARGIN = 15.0


def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
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
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum == 0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta=50.0):
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _ring_to_points(params, ring_counts):
    """
    Convert ring parameters to points.
    params: [r1, r2, ..., rk, theta1, theta2, ..., thetak]
    Radii are sorted to avoid permutation issues.
    """
    n_rings = len(ring_counts)
    radii = params[:n_rings]
    offsets = params[n_rings:]
    radii = np.maximum(radii, 0.01)
    radii_sorted = np.sort(radii)
    
    centers = []
    for idx, (count, radius) in enumerate(zip(ring_counts, radii_sorted)):
        if count == 1 and radius < 1e-10:
            centers.append(np.array([0.0, 0.0]))
        else:
            offset = offsets[idx] if idx < len(offsets) else 0.0
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            for i in range(count):
                centers.append(np.array([xs[i], ys[i]]))
    
    pts = np.array(centers[:N])
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(len(pts)):
        if norms[i] > 0.999:
            pts[i] *= 0.999 / norms[i]
    return pts


def _ring_objective(params, ring_counts):
    pts = _ring_to_points(params, ring_counts)
    return -_score(pts)


def _de_ring_search(ring_counts, deadline):
    """Differential evolution over ring parameters with manual implementation for time control."""
    n_rings = len(ring_counts)
    
    def neg_score(params):
        pts = _ring_to_points(params, ring_counts)
        return -_score(pts)
    
    # Bounds: radii in [0.05, 0.95], offsets in [0, 2*pi]
    bounds = []
    for i in range(n_rings):
        bounds.append((0.05, 0.95))
    for i in range(n_rings):
        bounds.append((0, 2 * np.pi))
    
    rng = np.random.RandomState(SEED)
    pop_size = 35
    max_gen = 80
    F = 0.8
    CR = 0.9
    
    pop = []
    for _ in range(pop_size):
        params = np.array([rng.uniform(b[0], b[1]) for b in bounds])
        pop.append(params)
    pop = np.array(pop)
    
    scores = np.array([neg_score(p) for p in pop])
    best_idx = np.argmin(scores)
    best_params = pop[best_idx].copy()
    best_score_val = scores[best_idx]
    
    for gen in range(max_gen):
        if time.time() > deadline - 30:
            break
        
        for i in range(pop_size):
            a_idx, b_idx, c_idx = rng.choice(pop_size, 3, replace=False)
            v = pop[a_idx] + F * (pop[b_idx] - pop[c_idx])
            for j in range(len(v)):
                v[j] = np.clip(v[j], bounds[j][0], bounds[j][1])
            
            trial = pop[i].copy()
            j_rand = rng.randint(len(v))
            for j in range(len(v)):
                if rng.rand() < CR or j == j_rand:
                    trial[j] = v[j]
            
            trial_score = neg_score(trial)
            if trial_score < scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score < best_score_val:
                    best_score_val = trial_score
                    best_params = trial.copy()
    
    return best_params, best_score_val


def _optimize_ring_params(params0, ring_counts, deadline):
    """Optimize ring parameters using L-BFGS-B."""
    try:
        result = minimize(
            _ring_objective,
            params0,
            args=(ring_counts,),
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result.fun < _ring_objective(params0, ring_counts):
            return result.x
    except Exception:
        pass
    return params0


def _anneal_points(pts, deadline, max_beta_stages=6):
    """Multi-stage annealing optimization from initial points."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 600.0]
    
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
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11, 'maxfun': 6000}
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
    for beta in [1000.0, 2000.0, 5000.0]:
        if time.time() > deadline - 5:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 25000}
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


def _constraint_push(pts, r_target, deadline, max_iters=500):
    """Push points to achieve target radius via constraint satisfaction."""
    pts = pts.copy()
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)
        
        if min_pair_margin < min_boundary_margin - 1e-14:
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
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
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


def _ring_initialization():
    """Generate ring-based initial configuration."""
    points = np.zeros((N, 2))
    points[0] = [0.0, 0.0]
    r1 = 0.25
    for i in range(6):
        angle = i * 2 * np.pi / 6
        points[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    r2 = 0.5
    for i in range(12):
        angle = (i + 0.5) * 2 * np.pi / 12
        points[7 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    r3 = 0.72
    for i in range(12):
        angle = i * 2 * np.pi / 12
        points[19 + i] = [r3 * np.cos(angle), r3 * np.sin(angle)]
    r4 = 0.88
    for i in range(19):
        angle = (i + 0.5) * 2 * np.pi / 19
        points[31 + i] = [r4 * np.cos(angle), r4 * np.sin(angle)]
    return points


def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    np.random.seed(SEED)
    
    best_points = None
    best_score = -1.0
    
    ring_counts = [1, 6, 12, 12, 19]
    
    # Strategy 1: Differential Evolution over ring parameters
    if time.time() < deadline - 50.0:
        try:
            de_params, de_score_val = _de_ring_search(ring_counts, deadline)
            # Polish the DE result
            best_params = _optimize_ring_params(de_params, ring_counts, deadline)
            de_points = _ring_to_points(best_params, ring_counts)
            de_score = _score(de_points)
            
            if de_score > best_score:
                best_score = de_score
                best_points = de_points.copy()
        except Exception:
            pass
    
    # Strategy 2: Ring-based initialization + multi-stage smooth optimization
    if time.time() < deadline - 40.0:
        initial = _ring_initialization()
        result_points, result_score = _anneal_points(initial, deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
    
    # Strategy 3: Hexagonal initialization + multi-stage smooth optimization
    if time.time() < deadline - 30.0:
        initial = _hexagonal_initial()
        result_points, result_score = _anneal_points(initial, deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
    
    # Strategy 4: Random restarts with smooth optimization
    for restart in range(2):
        if time.time() > deadline - 25.0:
            break
        
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0.05, 0.85, N)
        initial = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        result_points, result_score = _anneal_points(initial, deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()
    
    # Final constraint push to improve the best solution
    if best_points is not None and time.time() < deadline - 5.0:
        r_target = best_score
        pushed = _constraint_push(best_points, r_target, deadline, max_iters=200)
        pushed_score = _score(pushed)
        if pushed_score > best_score:
            best_score = pushed_score
            best_points = pushed.copy()
        
        # One more round of smooth polish after push
        if time.time() < deadline - 3.0:
            final_points, final_score = _anneal_points(best_points, deadline)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
    
    # Ensure valid output
    if best_points is None:
        best_points = _hexagonal_initial()
    
    best_points = np.array(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return best_points
