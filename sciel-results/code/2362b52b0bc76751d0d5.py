"""Pack 50 equal circles in the unit disk via log-sum-exp soft-min optimization + annealing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 115.0  # seconds, safety margin before 120s


def _pairwise_distances(P):
    """Compute pairwise distances for N points in 2D. Returns (N,N) matrix."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2)


def _score(P):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    container_margin = 1.0 - np.sqrt(np.sum(P ** 2, axis=1))
    min_container = np.min(container_margin)
    d = _pairwise_distances(P)
    min_pair = np.min(d) / 2.0
    return min(min_container, min_pair)


def _softmin_logsumexp(values, beta):
    """Numerically stable soft-min using log-sum-exp."""
    m = np.min(values)
    if not np.isfinite(m):
        return m
    shifted = values - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - np.log(np.sum(np.exp(-beta * shifted))) / beta


def _smooth_objective(P, beta, r_target):
    """
    Smooth objective: penalize violations of the constraint that all distances >= 2*r_target
    and all container margins >= r_target.
    
    We want to maximize min(min_pair/2, min_container).
    Equivalent to: for a target r, check if all pair dists >= 2r and all container margins >= r.
    
    Here we use a soft penalty: sum of soft-min violations.
    """
    d = _pairwise_distances(P)
    pair_violations = 2.0 * r_target - d  # positive when violated
    
    container_margin = 1.0 - np.sqrt(np.sum(P ** 2, axis=1))
    container_violations = r_target - container_margin  # positive when violated
    
    # Soft-min of the negative violations (we want them all <= 0)
    # Use log-sum-exp to get a smooth approximation of the max violation
    all_violations = np.concatenate([pair_violations.ravel(), container_violations.ravel()])
    
    # We want to minimize the maximum violation; use soft-max via log-sum-exp
    # soft_max(violations, beta) ≈ max(violations) for large beta
    m = np.max(all_violations)
    if not np.isfinite(m):
        return m
    shifted = all_violations - m
    log_sum = np.log(np.sum(np.exp(beta * shifted)))
    soft_max_val = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    # If soft_max_val <= 0, all constraints satisfied (approximately)
    # Return a penalty that is 0 when satisfied, positive when violated
    return max(0.0, soft_max_val)


def _gradient_smooth(P, beta, r_target):
    """
    Gradient of the smooth objective with respect to P.
    """
    d = _pairwise_distances(P)
    pair_violations = 2.0 * r_target - d
    
    container_margin = 1.0 - np.sqrt(np.sum(P ** 2, axis=1))
    container_violations = r_target - container_margin
    
    all_violations = np.concatenate([pair_violations.ravel(), container_violations.ravel()])
    
    m = np.max(all_violations)
    if not np.isfinite(m) or m <= 0:
        # No violations, gradient is zero
        return np.zeros_like(P)
    
    shifted = all_violations - m
    exp_shifted = np.exp(beta * shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    
    grad = np.zeros_like(P)
    
    # Pairwise part
    pair_weights = weights[:N * (N - 1)]
    pair_weights_matrix = pair_weights.reshape(N, N - 1)
    
    # Reconstruct pairwise weights properly
    # all_violations[:N*(N-1)] corresponds to pair_violations.ravel()
    # pair_violations[i,j] for i != j
    pw = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                pw[i, j] = pair_weights[idx]
                idx += 1
    
    # Gradient of (2r - d_ij) w.r.t. P_i = (P_i - P_j) / d_ij
    # Sum over j of pw[i,j] * (P_i - P_j) / d_ij
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    inv_d = np.zeros_like(d)
    np.divide(1.0, d, out=inv_d, where=d > 1e-12)
    
    # grad_i = sum_j pw[i,j] * (P_i - P_j) / d_ij
    grad_pair = np.sum(pw * inv_d, axis=1)[:, np.newaxis] * P - (P.T @ (pw * inv_d)).T
    
    # Wait, let me redo this more carefully
    # d(P_i - P_j)/dP_i = (P_i - P_j)/|P_i - P_j|
    # So grad_i = sum_{j != i} pw[i,j] * (P_i - P_j) / d_ij
    
    # Let's compute it directly
    grad_pair = np.zeros_like(P)
    for i in range(N):
        for j in range(N):
            if i != j:
                if d[i, j] > 1e-12:
                    grad_pair[i] += pw[i, j] * (P[i] - P[j]) / d[i, j]
    
    # Container part
    container_weights = weights[N * (N - 1):]
    dist_origin = np.sqrt(np.sum(P ** 2, axis=1))
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 1e-12)
    
    # Gradient of (r - (1 - |P_i|)) = (r + |P_i| - 1) w.r.t. P_i = P_i / |P_i|
    # So grad_i += container_weights[i] * P_i / |P_i|
    grad_container = (container_weights * inv_dist_origin)[:, np.newaxis] * P
    
    return grad_pair + grad_container


def _anneal_points(P_init, r_target, t_start=0.01, t_end=0.001, n_iters=5000, seed=42):
    """
    Simulated annealing to find a configuration where all circles of radius r_target fit.
    """
    rng = np.random.default_rng(seed)
    P = P_init.copy()
    best_P = P.copy()
    best_score = _score(P)
    
    for it in range(n_iters):
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (it / n_iters)
        
        # Pick a random point and perturb it
        idx = rng.integers(0, N)
        delta = t * rng.normal(size=2)
        new_pos = P[idx] + delta
        
        # Keep inside container
        new_r = np.sqrt(np.sum(new_pos ** 2))
        if new_r > 0.999:
            new_pos *= 0.999 / new_r
        
        P[idx] = new_pos
        new_score = _score(P)
        
        delta_score = new_score - best_score
        if delta_score > 0 or rng.random() < np.exp(delta_score / t):
            if new_score > best_score:
                best_score = new_score
                best_P = P.copy()
        else:
            P[idx] -= delta  # Revert
    
    return best_P, best_score


def _hexagonal_init():
    """Generate hexagonal lattice points inside unit circle."""
    spacing = 0.25
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x ** 2 + y ** 2 <= (1.0 - spacing / 2) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        # Take the N closest to origin
        dists = np.sqrt(np.sum(points ** 2, axis=1))
        idx = np.argsort(dists, kind='stable')
        return points[idx[:N]]
    return np.random.default_rng(0).normal(size=(N, 2)) * 0.5


def _ring_init(ring_counts, radii=None, offsets=None):
    """Generate ring-based initialization."""
    P = np.zeros((N, 2))
    k = len(ring_counts)
    if radii is None:
        radii = np.linspace(0.1, 0.9, k)
    if offsets is None:
        offsets = np.zeros(k)
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count + n_pts, 0] = R * np.cos(angles)
            P[count:count + n_pts, 1] = R * np.sin(angles)
            count += n_pts
    return P


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT
    
    # Start with multiple initializations
    initializations = []
    
    # Hexagonal
    P_hex = _hexagonal_init()
    initializations.append(P_hex)
    
    # Ring structures
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 10, 14, 19],
    ]
    
    for ring_counts in ring_structures:
        radii = np.linspace(0.15, 0.85, len(ring_counts))
        offsets = np.random.default_rng(hash(tuple(ring_counts)) % 2**32).uniform(0, 2 * np.pi, len(ring_counts))
        P_ring = _ring_init(ring_counts, radii, offsets)
        initializations.append(P_ring)
    
    # Evaluate all initializations
    best_P = initializations[0]
    best_score = _score(best_P)
    for P in initializations[1:]:
        s = _score(P)
        if s > best_score:
            best_score = s
            best_P = P
    
    # Binary search on radius with annealing
    r_low = best_score * 0.95
    r_high = 0.15
    
    # First, try to improve from current best using annealing at current radius
    if time.time() < deadline - 30:
        P_annealed, score_annealed = _anneal_points(best_P, best_score, t_start=0.02, t_end=0.002, n_iters=3000, seed=12345)
        if score_annealed > best_score:
            best_P = P_annealed
            best_score = score_annealed
            r_low = best_score
    
    # Binary search on radius
    for _ in range(8):
        if time.time() > deadline - 20:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try annealing at this radius
        P_candidate = best_P.copy()
        P_annealed, score_annealed = _anneal_points(P_candidate, r_mid, t_start=0.015, t_end=0.001, n_iters=2000, seed=42)
        
        if score_annealed >= r_mid - 1e-10:
            # Feasible!
            best_P = P_annealed
            best_score = max(best_score, score_annealed)
            r_low = r_mid
        else:
            r_high = r_mid
    
    # Final refinement: gradient-based optimization at the best radius
    if time.time() < deadline - 10:
        r_target = best_score * 1.00005
        
        def obj(P_flat):
            P = P_flat.reshape(N, 2)
            return _smooth_objective(P, beta=50.0, r_target=r_target)
        
        def grad(P_flat):
            P = P_flat.reshape(N, 2)
            return _gradient_smooth(P, beta=50.0, r_target=r_target).flatten()
        
        try:
            res = minimize(obj, best_P.flatten(), jac=grad, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            P_opt = res.x.reshape(N, 2)
            score_opt = _score(P_opt)
            if score_opt > best_score:
                best_P = P_opt
                best_score = score_opt
        except Exception:
            pass
    
    # Final local search polish
    if time.time() < deadline - 5:
        P_polished, score_polished = _anneal_points(best_P, best_score, t_start=0.005, t_end=0.0005, n_iters=1500, seed=99999)
        if score_polished > best_score:
            best_P = P_polished
            best_score = score_polished
    
    # Validate
    radii = np.sqrt(np.sum(best_P ** 2, axis=1))
    if np.any(~np.isfinite(best_P)) or np.any(radii >= 1.0):
        best_P = _hexagonal_init()
    
    return best_P
