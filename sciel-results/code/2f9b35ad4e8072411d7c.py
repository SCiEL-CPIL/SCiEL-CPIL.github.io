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
    # Try multiple ring configurations
    best_points = None
    best_score = -1
    
    # Ring configurations: (center, ring1_count, ring2_count, ring3_count, ...)
    configs = [
        [1, 6, 12, 18, 13],   # 50 total
        [1, 6, 12, 14, 17],   # 50 total
        [1, 6, 10, 16, 17],   # 50 total
        [0, 6, 12, 18, 14],   # 50 total
        [1, 6, 11, 17, 15],   # 50 total
    ]
    
    for config in configs:
        points_list = []
        
        # Center point
        if config[0] > 0:
            points_list.append(np.array([0.0, 0.0]))
        
        # Rings
        num_rings = len(config) - 1
        for ring_idx in range(num_rings):
            count = config[ring_idx + 1]
            if count == 0:
                continue
            
            # Radial distance for this ring
            # Approximate: ring radius ~ (ring_idx + 1) * 2*r where r ~ 0.126
            # So ring radii roughly: 0.252, 0.504, 0.756, 1.0
            # But we need to fit inside unit circle with margin
            base_radius = (ring_idx + 1) * 0.25
            # Adjust to keep inside unit circle
            max_allowed = 1.0 - 0.13  # margin for circle radius
            radius = min(base_radius, max_allowed)
            
            # Angular offset for this ring
            angle_offset = ring_idx * np.pi / count
            
            for k in range(count):
                angle = 2 * np.pi * k / count + angle_offset
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                points_list.append(np.array([x, y]))
        
        if len(points_list) != n:
            continue
        
        pts = np.array(points_list)
        
        # Check validity
        if np.any(np.linalg.norm(pts, axis=1) > 1.0):
            continue
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Also try hexagonal as fallback
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
    
    # Min distance from any center to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    if min_boundary <= 0:
        return 0.0
    
    # Min pairwise distance using cdist for speed
    D = cdist(points, points)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    return min(min_pair / 2.0, min_boundary)


def _smooth_objective_with_t(x, beta=100.0):
    """Smooth surrogate with explicit t variable.
    x = [t, x1, y1, x2, y2, ..., xN, yN]
    Minimize: -t + (1/beta) * log(sum(exp(-beta * (d_ij/2 - t)))) + (1/beta) * log(sum(exp(-beta * (1-|c_i| - t))))
    This is the log-sum-exp of the constraint violations, which is 0 when all constraints satisfied with margin."""
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
    
    # Constraint violations: we want d_ij/2 >= t and 1-|c_i| >= t
    # So violation = t - d_ij/2 and t - (1-|c_i|)
    pair_violations = t - pair_dists
    boundary_violations = t - boundary_dists
    
    # Log-sum-exp of violations (should be <= 0 for feasible)
    all_violations = np.concatenate([pair_violations, boundary_violations])
    
    # Soft max of violations
    max_v = np.max(all_violations)
    if max_v > 0:
        log_sum_exp = max_v + np.log(np.sum(np.exp(all_violations - max_v))) / beta
    else:
        log_sum_exp = max_v + np.log(np.sum(np.exp(all_violations - max_v))) / beta
    
    # Objective: -t + penalty for positive violations
    # If all violations <= 0, penalty is small
    return -t + log_sum_exp


def _polish(points, time_budget=20.0):
    """High-precision polish using scipy minimize with smooth objective."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Use t-variable formulation for better convergence
    t0 = best_score
    x0 = np.concatenate([[t0], points.flatten()])
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0]
    
    for beta in betas:
        if time.time() > deadline - 3:
            break
        
        try:
            result = minimize(
                _smooth_objective_with_t,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if result.success or result.fun < 0:
                new_t = result.x[0]
                new_points = result.x[1:].reshape(-1, 2)
                
                # Check validity
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
    """Run simulated annealing to improve the packing."""
    n = len(points)
    points = points.copy()
    best_points = points.copy()
    best_score = _compute_score(points)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial temperature
    T = 0.005
    T_min = 1e-12
    cooling_rate = 0.9998
    step_scale = 0.03
    
    rng = np.random.default_rng(12345)
    
    while time.time() < deadline and T > T_min:
        # Propose a random move
        idx = rng.integers(0, n)
        move_type = rng.random()
        
        if move_type < 0.6:
            # Random displacement
            delta = (rng.random(2) - 0.5) * 2 * step_scale * T * 10
            new_points = points.copy()
            new_points[idx] = points[idx] + delta
        elif move_type < 0.8:
            # Swap two points
            jdx = rng.integers(0, n)
            if jdx == idx:
                continue
            new_points = points.copy()
            new_points[idx], new_points[jdx] = points[jdx].copy(), points[idx].copy()
        elif move_type < 0.9:
            # Move along radial direction
            r = np.linalg.norm(points[idx])
            if r > 0.01:
                dir_vec = points[idx] / r
                delta_r = (rng.random() - 0.5) * 0.05
                new_points = points.copy()
                new_points[idx] = points[idx] + dir_vec * delta_r
            else:
                continue
        else:
            # Reflect a point through origin (with noise)
            new_points = points.copy()
            new_points[idx] = -points[idx] + (rng.random(2) - 0.5) * 0.01
        
        # Check validity
        if np.any(np.linalg.norm(new_points, axis=1) > 1.0):
            continue
        
        new_score = _compute_score(new_points)
        
        # Accept or reject
        if new_score >= best_score or rng.random() < np.exp((new_score - best_score) / T):
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        
        T *= cooling_rate
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Generate initial guess using ring-based structure
    points = _ring_initial(N)
    current_score = _compute_score(points)
    
    # Run simulated annealing
    points, current_score = _simulated_annealing(points, time_budget=60.0)
    
    # If time remains, try a few more restarts with different seeds
    elapsed = time.time() - start_time
    if elapsed < deadline - 25:
        for seed in [7, 13, 42, 99, 123]:
            if time.time() > deadline - 30:
                break
            rng = np.random.default_rng(seed)
            # Perturb the best solution
            perturbed = points + (rng.random((N, 2)) - 0.5) * 0.015
            # Ensure inside unit circle
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                perturbed[mask] /= norms[mask][:, None] * 1.005
            p, s = _simulated_annealing(perturbed, time_budget=min(12, deadline - 35 - (time.time() - start_time)))
            if s > current_score:
                points = p
                current_score = s
    
    # Final high-precision polish pass
    remaining = deadline - 5 - (time.time() - start_time)
    if remaining > 3:
        points, current_score = _polish(points, time_budget=remaining)
    
    return points
# EVOLVE-BLOCK-END
