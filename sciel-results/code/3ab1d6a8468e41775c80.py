import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0

def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

def _score(pts):
    if len(pts) != N or pts.shape != (N, 2):
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

def _smooth_grad(x, beta=50.0):
    pts = x.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0

    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = np.maximum(all_vals, -1.0)

    m = np.min(all_vals)
    if m < -0.5:
        return np.zeros_like(x)

    exp_terms = np.exp(-beta * (all_vals - m))
    weights = exp_terms / np.sum(exp_terms)

    grad = np.zeros_like(pts)

    # Boundary part
    w_boundary = weights[N:]
    dist_origin = np.sqrt(np.sum(pts ** 2, axis=1))
    for i in range(N):
        if dist_origin[i] > 1e-10:
            grad[i] -= w_boundary[i] * beta * pts[i] / dist_origin[i]

    # Pair part
    w_pair = weights[:N*N].reshape(N, N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            w = w_pair[i, j]
            if d[i, j] > 1e-10:
                direction = (pts[i] - pts[j]) / d[i, j]
                grad[i] += w * beta * direction

    return grad.flatten()

def _anneal_points(pts, deadline, max_beta_stages=6):
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

def _simulated_annealing(initial_pts, deadline, rng):
    """
    Run Simulated Annealing starting from initial_pts.
    """
    current_pts = initial_pts.copy()
    current_score = _score(current_pts)
    
    T_start = 0.05
    T_end = 0.0001
    alpha = 0.9995 # Cooling rate
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Determine number of steps based on time
    # Estimate cost per step. _score is O(N^2). N=50 -> 2500 ops. 
    # 1000 steps should be fast (<0.1s). 
    # We will run until deadline is approached.
    
    T = T_start
    step_size = 0.05
    
    while time.time() < deadline - 5.0:
        if T < T_end:
            break
            
        # Generate a random move
        idx = rng.randint(N)
        angle = rng.uniform(0, 2 * np.pi)
        # Step size decreases slightly with T? Or fixed? 
        # Fixed step size with random direction is standard.
        # Let's scale step size with T to allow larger jumps early.
        current_step = step_size * (T / T_start)
        
        new_pts = current_pts.copy()
        new_pts[idx, 0] += current_step * np.cos(angle)
        new_pts[idx, 1] += current_step * np.sin(angle)
        
        # Check if still inside roughly (don't strictly enforce, let score handle it)
        # But if it goes way outside, score becomes very negative.
        
        new_score = _score(new_pts)
        
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_pts = new_pts
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        
        T *= alpha
        
    return best_pts, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Initial points: Hexagonal packing
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            P_best = points[indices[:N]]
            break
        spacing *= 0.98
        
    score_best = _score(P_best)
    
    # Run multiple SA restarts
    # Each restart takes some time. We can run 5-10 restarts.
    num_restarts = 8
    time_per_restart = (deadline - 20.0) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        rng = np.random.RandomState(i * 12345 + 67890)
        
        # Perturb initial points slightly to avoid identical trajectories
        if i == 0:
            start_pts = P_best.copy()
        else:
            start_pts = P_best + rng.uniform(-0.05, 0.05, size=(N, 2))
            # Ensure start_pts is roughly valid (inside circle)
            norms = np.linalg.norm(start_pts, axis=1)
            for j in range(N):
                if norms[j] > 0.95:
                    start_pts[j] *= 0.95 / norms[j]
                    
        # Run SA
        P_sa, score_sa = _simulated_annealing(start_pts, deadline - 10.0 - i*time_per_restart, rng)
        
        if score_sa > score_best:
            P_best = P_sa
            score_best = score_sa
            
        # Polish the result of this restart
        if time.time() < deadline - 5.0:
            P_polished, score_polished = _anneal_points(P_sa, deadline - 2.0)
            if score_polished > score_best:
                P_best = P_polished
                score_best = score_polished

    # Final global polish on the best found so far
    if time.time() < deadline - 2.0:
        P_final, score_final = _anneal_points(P_best, deadline)
        if score_final > score_best:
            P_best = P_final
            score_best = score_final

    # Safety check
    radii = np.sqrt(np.sum(P_best ** 2, axis=1))
    if np.any(~np.isfinite(P_best)) or np.any(radii >= 1.0):
        # Fallback to hex
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            margin = spacing / 2.0
            points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
            if len(points) >= N:
                indices = np.argsort(radius, kind="stable")
                P_best = points[indices[:N]]
                break
            spacing *= 0.98

    return P_best
