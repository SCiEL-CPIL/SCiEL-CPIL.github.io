"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _ring_initial(n=50):
    """Create ring-based initial points matching known 50-circle packing structure.
    Known structure: center + rings of 6, 12, 18, 13 (or similar)."""
    best_points = None
    best_score = -1
    
    configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 10, 16, 17],
        [0, 6, 12, 18, 14],
        [1, 6, 11, 17, 15],
        [1, 6, 13, 14, 16],
        [0, 8, 14, 16, 12],
    ]
    
    for config in configs:
        points_list = []
        
        if config[0] > 0:
            points_list.append(np.array([0.0, 0.0]))
        
        num_rings = len(config) - 1
        for ring_idx in range(num_rings):
            count = config[ring_idx + 1]
            if count == 0:
                continue
            
            base_radius = (ring_idx + 1) * 0.25
            max_allowed = 1.0 - 0.13
            radius = min(base_radius, max_allowed)
            
            angle_offset = ring_idx * np.pi / count
            
            for k in range(count):
                angle = 2 * np.pi * k / count + angle_offset
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                points_list.append(np.array([x, y]))
        
        if len(points_list) != n:
            continue
        
        pts = np.array(points_list)
        
        if np.any(np.linalg.norm(pts, axis=1) > 1.0):
            continue
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    if best_points is None or best_score < 0.1:
        hex_pts = _hexagonal_initial(n)
        hex_score = _compute_score(hex_pts)
        if hex_score > best_score:
            best_points = hex_pts
            best_score = hex_score
    
    return best_points


def _hexagonal_initial(n=50):
    """Create initial hexagonal lattice points inside unit disk."""
    best_points = None
    best_score = -1
    
    for base_spacing in [0.22, 0.23, 0.24, 0.25, 0.26]:
        spacing = base_spacing
        for _ in range(15):
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists)[:n]
                pts = points[idx]
                score = _compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
            spacing *= 0.97
    
    if best_points is None:
        rng = np.random.default_rng(42)
        best_points = np.zeros((n, 2))
        for i in range(n):
            while True:
                angle = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                pt = np.array([r*np.cos(angle), r*np.sin(angle)])
                if np.all(np.linalg.norm(pt - best_points[:i], axis=1) > 0.15):
                    best_points[i] = pt
                    break
    return best_points


def _compute_score(points):
    """Compute the packing score: min(min_pair_dist/2, min(1-|c_i|))."""
    n = len(points)
    if n == 0:
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    if min_boundary <= 0:
        return 0.0
    
    D = cdist(points, points)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    return min(min_pair / 2.0, min_boundary)


def _fast_score_delta(points, idx, old_pos):
    """Fast incremental score update when only point idx changes from old_pos to points[idx].
    Returns (new_score, min_pair_half, min_boundary) without full cdist."""
    n = len(points)
    
    # New boundary distance for changed point
    new_norm = np.linalg.norm(points[idx])
    
    # Min boundary over all points
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    if min_boundary <= 0:
        return 0.0, 0.0, 0.0
    
    # Min pair distance: need min over all pairs
    # Only pairs involving idx changed, so we need:
    #   min(old_min_pair_excluding_idx, min over j != idx of |points[idx] - points[j]|)
    # But we don't have old_min_pair_excluding_idx cached here, so do full computation
    # This is still faster than cdist for small changes if we use vectorized ops
    diffs = points - points[idx]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    dists[idx] = np.inf
    min_pair_involving_idx = np.min(dists)
    
    # For a true incremental approach we'd need to cache, but let's at least
    # avoid the full NxN cdist by computing only what's needed
    # Actually for N=50, let's just do the full thing but with numpy instead of cdist
    D = np.sqrt(((points[:, np.newaxis, :] - points[np.newaxis, :, :])**2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    return min(min_pair / 2.0, min_boundary), min_pair / 2.0, min_boundary


def _smooth_objective_with_t(x, beta=100.0):
    """Smooth surrogate: minimize -t + (1/beta)*logsumexp(beta*(t - d_ij/2)) + (1/beta)*logsumexp(beta*(t - (1-|ci|)))
    where d_ij are pairwise distances. This is an epigraph formulation."""
    t = x[0]
    points = x[1:].reshape(-1, 2)
    n = len(points)
    
    # Pairwise distances
    D = cdist(points, points)
    np.fill_diagonal(D, np.inf)
    pair_dists = D[D > 0] / 2.0
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    
    # Violations: positive means constraint is violated
    pair_violations = t - pair_dists
    boundary_violations = t - boundary_dists
    
    # Log-sum-exp of beta * violations (soft-max scaled by beta)
    def _logsumexp(v):
        m = np.max(v)
        if m == -np.inf:
            return -np.inf
        return m + np.log(np.sum(np.exp(v - m)))
    
    lse_pair = _logsumexp(beta * pair_violations)
    lse_bound = _logsumexp(beta * boundary_violations)
    
    # Objective: -t + (1/beta) * (lse_pair + lse_bound)
    # When all violations <= 0, lse terms are small (negative), so -t dominates
    obj = -t + (lse_pair + lse_bound) / beta
    
    return obj


def _polish(points, time_budget=20.0):
    """High-precision polish using scipy minimize with smooth objective."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    t0 = best_score
    x0 = np.concatenate([[t0], points.flatten()])
    
    betas = [30.0, 60.0, 120.0, 250.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 3:
            break
        
        try:
            result = minimize(
                _smooth_objective_with_t,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 80}
            )
            
            new_points = result.x[1:].reshape(-1, 2)
            
            if np.any(np.linalg.norm(new_points, axis=1) > 1.0):
                continue
            
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
                x0 = np.concatenate([[new_score], new_points.flatten()])
        except Exception:
            continue
    
    return best_points, best_score


def _simulated_annealing(points, time_budget=100.0):
    """Run simulated annealing with fast incremental scoring."""
    n = len(points)
    points = points.copy()
    best_points = points.copy()
    best_score = _compute_score(points)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    T = 0.003
    T_min = 1e-13
    cooling_rate = 0.9997
    step_scale = 0.025
    
    rng = np.random.default_rng(12345)
    
    iters = 0
    while time.time() < deadline and T > T_min:
        idx = rng.integers(0, n)
        move_type = rng.random()
        
        if move_type < 0.55:
            delta = (rng.random(2) - 0.5) * 2 * step_scale * (0.5 + T * 50)
            new_points = points.copy()
            new_points[idx] = points[idx] + delta
        elif move_type < 0.75:
            jdx = rng.integers(0, n)
            if jdx == idx:
                continue
            new_points = points.copy()
            new_points[idx], new_points[jdx] = points[jdx].copy(), points[idx].copy()
        elif move_type < 0.90:
            r = np.linalg.norm(points[idx])
            if r > 0.01:
                dir_vec = points[idx] / r
                delta_r = (rng.random() - 0.5) * 0.04
                new_points = points.copy()
                new_points[idx] = points[idx] + dir_vec * delta_r
            else:
                continue
        else:
            # Move toward/away from nearest neighbor
            diffs = points - points[idx]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists[idx] = np.inf
            nearest = np.argmin(dists)
            direction = (points[nearest] - points[idx]) / dists[nearest]
            step = (rng.random() - 0.5) * 0.03
            new_points = points.copy()
            new_points[idx] = points[idx] + direction * step
        
        if np.any(np.linalg.norm(new_points, axis=1) > 1.0):
            continue
        
        new_score = _compute_score(new_points)
        
        if new_score >= best_score or rng.random() < np.exp((new_score - best_score) / T):
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        
        T *= cooling_rate
        iters += 1
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    points = _ring_initial(N)
    current_score = _compute_score(points)
    
    # Main SA phase
    points, current_score = _simulated_annealing(points, time_budget=55.0)
    
    # Restart phase with perturbations
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        for seed in [7, 13, 42, 99, 123, 256, 512]:
            if time.time() > deadline - 35:
                break
            remaining_sa = min(10, deadline - 35 - (time.time() - start_time))
            if remaining_sa < 3:
                break
            rng = np.random.default_rng(seed)
            perturbed = points + (rng.random((N, 2)) - 0.5) * 0.012
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                perturbed[mask] /= norms[mask][:, None] * 1.001
            p, s = _simulated_annealing(perturbed, time_budget=remaining_sa)
            if s > current_score:
                points = p
                current_score = s
    
    # Final polish
    remaining = deadline - 5 - (time.time() - start_time)
    if remaining > 3:
        points, current_score = _polish(points, time_budget=remaining)
    
    return points
# EVOLVE-BLOCK-END
