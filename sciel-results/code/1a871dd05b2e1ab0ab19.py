"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0

    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))

    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))

    return min(min_pair / 2.0, min_boundary)

def _hex_lattice_init(spacing, angle_deg=0):
    """Generate hexagonal lattice points within the unit disk."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    k = int(1.0 / spacing) + 5
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rows.append((xr, yr))

    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0
    points = points[mask]
    
    if len(points) < N:
        return None

    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]

def _radial_ring_init(n_rings=15):
    """Initialize points in concentric rings."""
    points = []
    
    # Ring 0: center
    points.append([0.0, 0.0])
    
    # Estimate spacing
    base_spacing = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    for ring in range(1, n_rings):
        r = ring * base_spacing * 0.95
        if r >= 1.0:
            break
        
        # Number of points that fit in this ring
        # Circumference = 2*pi*r, each point needs ~2*r_spacing of arc
        # But we need to account for the chord distance
        n_in_ring = int(2 * np.pi * r / (base_spacing * 1.05))
        if n_in_ring < 1:
            n_in_ring = 1
        
        # Don't exceed remaining points
        remaining = N - len(points)
        n_in_ring = min(n_in_ring, remaining)
        
        # Add offset for alternating rings
        offset = (ring % 2) * np.pi / n_in_ring
        
        for i in range(n_in_ring):
            theta = 2 * np.pi * i / n_in_ring + offset
            points.append([r * np.cos(theta), r * np.sin(theta)])
    
    points = np.array(points[:N])
    
    # If we don't have enough, fill with random
    if len(points) < N:
        deficit = N - len(points)
        angles = np.random.uniform(0, 2 * np.pi, deficit)
        radii = np.random.uniform(0.1, 0.9, deficit)
        extra = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        points = np.vstack([points, extra])
    
    return points

def _soft_min_gradient(points, beta=50.0):
    """
    Compute gradient of soft-min of all constraints using log-sum-exp.
    Returns (gradient, soft_min_value).
    """
    n = len(points)
    
    # Pair distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    
    # Get all constraint values
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]
    
    # Combine all constraints
    all_dists = np.concatenate([pair_dists, boundary_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d_k)))
    # Gradient: sum(exp(-beta * d_k) * grad_d_k) / sum(exp(-beta * d_k))
    
    # For numerical stability, subtract max
    max_d = np.max(all_dists)
    exp_vals = np.exp(-beta * (all_dists - max_d))
    total_weight = np.sum(exp_vals)
    
    if total_weight < 1e-10:
        return np.zeros_like(points), 0.0
    
    weights = exp_vals / total_weight
    
    grad = np.zeros_like(points)
    
    # Pair constraints
    i_vals, j_vals = iu
    for k in range(len(pair_dists)):
        i, j = i_vals[k], j_vals[k]
        d = pair_dists[k]
        if d > 1e-12:
            vec = (points[i] - points[j]) / d
            grad[i] += weights[k] * vec
            grad[j] += weights[k] * (-vec)
    
    # Boundary constraints
    for k in range(len(boundary_dists)):
        i = k
        norm = np.linalg.norm(points[i])
        if norm > 1e-12:
            grad[i] += weights[len(pair_dists) + k] * (-points[i] / norm)
    
    # Soft min value
    soft_min_val = max_d - np.log(total_weight) / beta
    
    return grad, soft_min_val

def _local_optimize(points, max_iter=2000, deadline=None, lr_init=0.005):
    """
    Local optimization using soft-min gradient ascent.
    """
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    lr = lr_init
    patience = 0
    max_patience = 150
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        score = _compute_score(best_points)
        if score > best_score + 1e-10:
            best_score = score
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        grad, _ = _soft_min_gradient(best_points, beta=50.0)
        
        if np.linalg.norm(grad) < 1e-8:
            break
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        new_points = best_points + lr * grad
        
        # Project into unit disk
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / norms[mask, np.newaxis] * 0.9999
        
        new_score = _compute_score(new_points)
        
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            patience = 0
            lr = min(lr * 1.02, 0.02)
        else:
            lr *= 0.9
            if lr < 1e-6:
                break
    
    return best_points

def _final_polish(points, max_iter=500, deadline=None):
    """Final high-precision polish with very small steps."""
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    lr = 0.001
    patience = 0
    max_patience = 100
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        score = _compute_score(best_points)
        if score > best_score + 1e-12:
            best_score = score
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        grad, _ = _soft_min_gradient(best_points, beta=100.0)
        
        if np.linalg.norm(grad) < 1e-10:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        new_points = best_points + lr * grad
        
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / norms[mask, np.newaxis] * 0.99999
        
        new_score = _compute_score(new_points)
        
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            patience = 0
        else:
            lr *= 0.85
            if lr < 1e-8:
                break
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = -np.inf
    
    # Generate initializations
    inits = []
    
    # Hex lattice inits
    base_spacing = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    for factor in [0.95, 0.98, 1.0, 1.02, 1.05]:
        spacing = base_spacing * factor
        for angle_deg in [0, 15, 30, 45, 60, 75]:
            pts = _hex_lattice_init(spacing, angle_deg)
            if pts is not None:
                score = _compute_score(pts)
                inits.append((score, pts))
    
    # Radial ring init
    ring_pts = _radial_ring_init()
    ring_score = _compute_score(ring_pts)
    inits.append((ring_score, ring_pts))
    
    # Sort and pick top initializations
    inits.sort(key=lambda x: -x[0])
    top_inits = inits[:4]
    
    # Optimize each
    for score, pts in top_inits:
        if time.time() > deadline - 15:
            break
        
        optimized = _local_optimize(pts, max_iter=1500, deadline=deadline - 5)
        opt_score = _compute_score(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()
    
    # Final polish
    if best_points is not None and time.time() < deadline - 10:
        polished = _final_polish(best_points, max_iter=300, deadline=deadline - 2)
        pol_score = _compute_score(polished)
        if pol_score > best_score:
            best_score = pol_score
            best_points = polished.copy()
    
    # Fallback
    if best_points is None:
        pts = _hex_lattice_init(base_spacing, 0)
        if pts is not None:
            best_points = pts
        else:
            angles = np.linspace(0, 2 * np.pi * N, N)
            radii = np.linspace(0, 0.95, N)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Final validation
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.9999
    
    return best_points
# EVOLVE-BLOCK-END
