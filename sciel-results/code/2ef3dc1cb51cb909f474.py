import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the packing radius for a set of points."""
    if points is None or len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    norms = np.linalg.norm(points, axis=1)
    boundary_r = 1.0 - np.max(norms)
    if boundary_r < 0:
        return 0.0
    
    d = pdist(points)
    if len(d) == 0:
        pair_r = float('inf')
    else:
        pair_r = np.min(d) / 2.0
    
    r = min(pair_r, boundary_r)
    return max(0.0, r)

def _soft_min(values, alpha=20.0):
    """Smooth approximation of min(values) using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    max_val = np.max(values)
    if max_val < -50.0:
        return max_val
    shifted = values - max_val
    log_sum = np.log(np.sum(np.exp(alpha * shifted))) / alpha
    return max_val + log_sum / alpha

def _smooth_objective(x, alpha=20.0):
    """
    Smooth objective: minimize -r where r approximates the packing radius.
    We use a smooth approximation of the minimum distance.
    """
    points = x.reshape(-1, 2)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    
    # Upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Smooth min
    r_approx = _soft_min(all_dists, alpha)
    
    return -r_approx

def _smooth_gradient(x, alpha=20.0):
    """
    Gradient of the smooth objective.
    """
    points = x.reshape(-1, 2)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    
    # Upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min weights
    max_val = np.max(all_dists)
    shifted = all_dists - max_val
    weights = np.exp(alpha * shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(x)
    
    # Gradient of pairwise distances
    # d(dist_ij)/d(points[i]) = (points[i] - points[j]) / dist_ij
    for k in range(len(pair_dists)):
        i, j = iu[0][k], iu[1][k]
        w = weights[k] / alpha
        d = pair_dists[k]
        if d > 1e-10:
            diff_ij = points[i] - points[j]
            grad[i*2:i*2+2] += w * diff_ij / d
            grad[j*2:j*2+2] -= w * diff_ij / d
    
    # Gradient of boundary distances
    # d(1 - ||p_i||)/d(p_i) = -p_i / ||p_i||
    for i in range(N):
        w = weights[len(pair_dists) + i] / alpha
        n = norms[i]
        if n > 1e-10:
            grad[i*2:i*2+2] -= w * points[i] / n
    
    return -grad  # We're minimizing -r, so gradient is -grad_r

def _generate_hex_rings():
    """
    Generate a hexagonal ring configuration.
    Center circle + rings of circles in hexagonal pattern.
    """
    # For N=300, use rings: 1 + 6 + 12 + 18 + ... 
    # Ring k has 6*k circles
    # Sum_{k=1}^{m} 6k = 3*m*(m+1)
    # 1 + 3*m*(m+1) = 300 => 3m^2 + 3m = 299 => m^2 + m ~ 99.67
    # m=9: 1+3*9*10=271, m=10: 1+3*10*11=331
    # So use m=9 full rings (271 circles) + 29 extra
    
    pts = [[0.0, 0.0]]
    remaining = N - 1
    
    # Determine ring radii
    # Optimal ring radius for hexagonal packing with circle radius r:
    # Ring k radius = k * r * sqrt(3)
    # But we need to fit in unit circle, so scale
    
    # Start with estimated r ~ 0.053
    r_est = 0.053
    ring_radii = []
    for k in range(1, 11):
        ring_radii.append(k * r_est * np.sqrt(3))
    
    # Distribute circles among rings
    # Ring k can hold 6*k circles in hexagonal pattern
    counts = []
    for k in range(1, 11):
        counts.append(6 * k)
    
    total_in_rings = sum(counts)
    if total_in_rings < remaining:
        # Add extra to last ring
        counts[-1] += remaining - total_in_rings
    elif total_in_rings > remaining:
        # Remove from last rings
        excess = total_in_rings - remaining
        for k in range(len(counts)-1, -1, -1):
            remove = min(excess, counts[k])
            counts[k] -= remove
            excess -= remove
            if excess == 0:
                break
    
    for ring_idx, count in enumerate(counts):
        if count <= 0:
            continue
        r_ring = ring_radii[ring_idx]
        angles = 2 * np.pi * np.arange(count) / count
        # Offset alternating rings
        offset = 0.0 if ring_idx % 2 == 0 else np.pi / count
        x = r_ring * np.cos(angles + offset)
        y = r_ring * np.sin(angles + offset)
        for j in range(count):
            pts.append([x[j], y[j]])
    
    if len(pts) != N:
        # Fallback: adjust
        while len(pts) < N:
            pts.append([0.0, 0.0])
        pts = pts[:N]
    
    return np.array(pts)

def _generate_hex_lattice(spacing=0.055, rotation=0.0):
    """Generate hexagonal lattice points inside unit disk."""
    k = int(1.0 / spacing) + 5
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            x_rot = x * np.cos(rotation) - y * np.sin(rotation)
            y_rot = x * np.sin(rotation) + y * np.cos(rotation)
            if np.hypot(x_rot, y_rot) <= 1.0:
                pts.append((x_rot, y_rot))
    
    if len(pts) < N:
        return None
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _optimize_local(points, time_budget=20.0):
    """
    Optimize a local configuration using L-BFGS-B with smooth objective.
    """
    start_time = time.time()
    x0 = points.flatten().copy()
    
    alpha = 20.0
    
    def obj(x):
        return _smooth_objective(x, alpha)
    
    def grad(x):
        return _smooth_gradient(x, alpha)
    
    best_x = x0.copy()
    best_score = _compute_score(points)
    
    options = {
        'maxiter': 200,
        'ftol': 1e-10,
        'gtol': 1e-7,
        'maxls': 20
    }
    
    try:
        result = minimize(obj, x0, jac=grad, method='L-BFGS-B', options=options)
        
        if result.x is not None and np.all(np.isfinite(result.x)):
            candidate = result.x.reshape(-1, 2)
            score = _compute_score(candidate)
            if score > best_score:
                best_score = score
                best_x = candidate.flatten().copy()
    except Exception:
        pass
    
    # If we have time, try with higher alpha (sharper min)
    if time.time() - start_time < time_budget * 0.5:
        alpha2 = 50.0
        
        def obj2(x):
            return _smooth_objective(x, alpha2)
        
        def grad2(x):
            return _smooth_gradient(x, alpha2)
        
        try:
            result2 = minimize(obj2, best_x, jac=grad2, method='L-BFGS-B', options=options)
            if result2.x is not None and np.all(np.isfinite(result2.x)):
                candidate = result2.x.reshape(-1, 2)
                score = _compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_x = candidate.flatten().copy()
        except Exception:
            pass
    
    return best_x.reshape(-1, 2), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Generate multiple initial candidates
    candidates = []
    
    # 1. Hexagonal rings
    pts_rings = _generate_hex_rings()
    if pts_rings is not None:
        r = _compute_score(pts_rings)
        candidates.append((pts_rings, r))
    
    # 2. Hexagonal lattices with different spacings and rotations
    for spacing in [0.052, 0.054, 0.056, 0.058]:
        for rot in [0.0, np.pi/6, np.pi/12, np.pi/3]:
            pts = _generate_hex_lattice(spacing, rotation=rot)
            if pts is not None:
                r = _compute_score(pts)
                if r > 0:
                    candidates.append((pts, r))
    
    if not candidates:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
    
    # Sort by initial score
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Optimize top candidates
    max_to_optimize = 6
    for i, (init_pts, init_r) in enumerate(candidates[:max_to_optimize]):
        if time.time() > deadline - 5.0:
            break
        
        # Time budget for this optimization
        remaining_time = deadline - time.time()
        time_budget = min(remaining_time * 0.3, 25.0)
        
        if time_budget < 2.0:
            break
        
        opt_pts, opt_score = _optimize_local(init_pts, time_budget=time_budget)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
    
    # If we still have time, try more candidates or refine best
    if best_points is not None and time.time() < deadline - 10.0:
        # Refine the best solution further
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            refined_pts, refined_score = _optimize_local(best_points, time_budget=remaining_time * 0.8)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()
    
    if best_points is None:
        # Return best initial candidate
        best_points = candidates[0][0]
    
    return best_points

# EVOLVE-BLOCK-END
