"""Pack 50 equal circles in the unit disk via ring-structured DE + multi-stage smooth annealing + constraint push + binary search."""
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


def _ring_to_points(params, ring_counts):
    n_rings = len(ring_counts)
    radii = params[:n_rings]
    offsets = params[n_rings:]
    radii = np.maximum(radii, 0.01)
    radii = np.sort(radii)
    
    centers = []
    for idx, (count, radius) in enumerate(zip(ring_counts, radii)):
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


def _constraint_push(pts, r_target, deadline, max_iters=500):
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
        
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _de_ring_search(ring_counts, deadline):
    n_rings = len(ring_counts)
    
    def ring_to_pts(params):
        return _ring_to_points(params, ring_counts)
    
    def neg_score(params):
        pts = ring_to_pts(params)
        return -_score(pts)
    
    bounds = []
    for i in range(n_rings):
        bounds.append((0.05, 0.95))
    for i in range(n_rings):
        bounds.append((0, 2 * np.pi))
    
    rng = np.random.RandomState(42)
    pop_size = 40
    max_gen = 60
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
    best_score = scores[best_idx]
    
    for gen in range(max_gen):
        if time.time() > deadline - 25:
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
                if trial_score < best_score:
                    best_params = trial.copy()
                    best_score = trial_score
    
    best_pts = ring_to_pts(best_params)
    return best_pts


def _get_hex_initial_points():
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
            return points[indices[:N]]
        spacing *= 0.98


def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 12, 19],
    ]
    
    P_best = _get_hex_initial_points()
    score_best = _score(P_best)
    
    for ring_counts in ring_structures:
        if time.time() > deadline - 50.0:
            break
        
        P_candidate = _de_ring_search(ring_counts, deadline)
        score_candidate = _score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate
            score_best = score_candidate
        
        if time.time() < deadline - 30.0:
            P_polished, score_polished = _anneal_points(P_candidate, deadline)
            if score_polished > score_best:
                P_best = P_polished
                score_best = score_polished
        
        if time.time() < deadline - 20.0:
            r_low = score_best * 0.99
            r_high = 0.145
            
            for _ in range(6):
                if time.time() > deadline - 15.0:
                    break
                if r_high - r_low < 1e-8:
                    break
                    
                r_mid = (r_low + r_high) / 2.0
                P_init = P_best.copy()
                
                P_opt = _constraint_push(P_init, r_mid, deadline)
                score_opt = _score(P_opt)
                
                if score_opt >= r_mid * 0.999:
                    if score_opt > score_best:
                        P_best = P_opt
                        score_best = score_opt
                    r_low = r_mid
                else:
                    r_high = r_mid
    
    if time.time() < deadline - 10.0:
        remaining_iters = min(2500, int((deadline - 10.0 - time.time()) * 400))
        remaining_iters = max(300, remaining_iters)
        P_best, score_best = _anneal_points(P_best, deadline - 10.0)
    
    if time.time() < deadline - 5.0:
        P_final, score_final = _anneal_points(P_best, deadline)
        if score_final > score_best:
            P_best = P_final
            score_best = score_final
        
        if time.time() < deadline - 2.0:
            r_target = score_best * 1.00001
            P_opt2 = _constraint_push(P_best, r_target, deadline)
            score_opt2 = _score(P_opt2)
            if score_opt2 > score_best:
                P_best = P_opt2
                score_best = score_opt2
    
    radii = np.sqrt(np.sum(P_best ** 2, axis=1))
    if np.any(~np.isfinite(P_best)) or np.any(radii >= 1.0):
        P_best = _get_hex_initial_points()
    
    return P_best
