"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0
    
    # Min distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))
    
    # Min pair distance via vectorized computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))
    
    return min(min_pair / 2.0, min_boundary)


def _soft_min_value_and_grad(pos, t=80.0):
    """
    Compute soft-min of all pairwise distances and boundary distances, plus gradient.
    Returns (soft_min_value, gradient, actual_min).
    """
    n = len(pos)
    pos = pos.reshape(n, 2)

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)

    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(pos, axis=1)

    # Flatten pairwise distances (upper triangle)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]

    # All distances
    all_dists = np.concatenate([pair_dists, boundary_dists])

    # Compute weights using log-sum-exp for numerical stability
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    exp_terms = np.exp(-t * shifted)
    S = np.sum(exp_terms)

    # Gradient of soft-min: f = (1/t)*log(S) + d_min
    # df/dx = -sum_k w_k * d(d_k)/dx  where w_k = exp(-t*(d_k-d_min))/S

    grad = np.zeros_like(pos)

    # Pairwise gradient
    pair_weights = exp_terms[:len(pair_dists)] / S
    diff_pairs = diff[iu]
    norm_pairs = pair_dists[:, np.newaxis]
    norm_pairs[norm_pairs < 1e-12] = 1e-12
    unit_vecs = diff_pairs / norm_pairs

    i_idx, j_idx = iu
    np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * unit_vecs)

    # Boundary gradient
    boundary_weights = exp_terms[len(pair_dists):] / S
    norms = np.linalg.norm(pos, axis=1)[:, np.newaxis]
    norms[norms < 1e-12] = 1e-12
    unit_pos = pos / norms
    # d(1-|x|)/dx = -x/|x|, so contribution is -w * (-unit_pos) = w * unit_pos
    grad += boundary_weights[:, np.newaxis] * unit_pos

    soft_min_val = (1.0 / t) * np.log(S) + d_min
    return soft_min_val, grad, d_min


def _hex_lattice_init(spacing, angle_deg=0):
    """Generate hexagonal lattice points within the unit disk."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            # Rotate
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rows.append((xr, yr))

    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]

    if len(points) < N:
        return None

    # Sort by distance from center and take N closest
    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]


def _radial_ring_init():
    """
    Place circles in concentric rings. Each ring has a radial distance and
    an angular offset.
    """
    best_points = None
    best_score = -np.inf

    # Try different ring structures
    for n_rings in range(5, 16):
        # Distribute N circles among rings
        # Outer rings can hold more circles
        weights = []
        for i in range(n_rings):
            r_i = (i + 1) / n_rings * 0.95
            weights.append(2 * np.pi * r_i)

        total_w = sum(weights)
        counts = []
        remaining = N
        for i in range(n_rings - 1):
            c_i = max(1, int(round(N * weights[i] / total_w)))
            c_i = min(c_i, remaining - (n_rings - 1 - i))
            counts.append(c_i)
            remaining -= c_i
        counts.append(remaining)

        # Now compute positions
        points = []
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            r_i = (i + 1) / n_rings * 0.95
            # Angular offset to stagger rings
            offset = (i * np.pi) / max(1, count)
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            for a in angles:
                points.append((r_i * np.cos(a), r_i * np.sin(a)))

        if len(points) != N:
            continue

        points = np.array(points)
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    return best_points


def _local_optimize_lbfgs(points, deadline=None):
    """
    Local optimization using L-BFGS-B on the soft-min surrogate.
    Start with low temperature (smooth) and increase to sharp for fine-tuning.
    """
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)

    # Temperature schedule: start smooth, get sharper
    t_schedule = [10.0, 20.0, 40.0, 80.0, 120.0, 160.0]
    
    for t in t_schedule:
        if deadline is not None and time.time() > deadline:
            break

        def objective(x):
            if deadline is not None and time.time() > deadline:
                return np.inf
            x = x.reshape(n, 2)
            val, grad, _ = _soft_min_value_and_grad(x, t)
            return val, -grad  # We want to minimize negative soft-min (i.e., maximize soft-min)
        
        # Actually, we want to maximize the minimum distance, so we minimize -soft_min
        # But scipy minimize wants to minimize, so we negate
        def objective_neg(x):
            if deadline is not None and time.time() > deadline:
                return np.inf
            x = x.reshape(n, 2)
            val, grad, _ = _soft_min_value_and_grad(x, t)
            # We want to maximize val, so minimize -val
            return -val, -grad

        x0 = best_points.reshape(-1)
        result = minimize(objective_neg, x0, jac=True, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.fun < 0:  # Improved
            new_points = result.x.reshape(n, 2)
            # Ensure points stay inside unit circle
            norms = np.linalg.norm(new_points, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                new_points[mask] = new_points[mask] / norms[mask, np.newaxis]
            
            score = _compute_score(new_points)
            if score > best_score:
                best_score = score
                best_points = new_points.copy()

    return best_points


def _local_optimize_gradient(points, max_iter=500, lr=0.005, deadline=None):
    """
    Local optimization using gradient ascent on the soft-min of distances.
    Maximizes the minimum pairwise and boundary distance.
    """
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)

    t = 80.0  # Temperature for soft-min
    current_lr = lr
    patience = 0
    max_patience = 30

    start_time = time.time()

    for iteration in range(max_iter):
        # Deadline check
        if deadline is not None and time.time() > deadline:
            break

        # Compute gradient
        _, grad, actual_min = _soft_min_value_and_grad(best_points, t)

        # Gradient ascent (we want to maximize min distance)
        new_points = best_points + current_lr * grad

        # Keep points inside unit circle
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / norms[mask, np.newaxis]

        # Compute actual score
        score = _compute_score(new_points)

        if score > best_score:
            best_score = score
            best_points = new_points.copy()
            patience = 0
            # Slightly increase learning rate if making progress
            current_lr = min(current_lr * 1.01, 0.015)
        else:
            patience += 1
            # Decay learning rate
            current_lr *= 0.95

        if patience >= max_patience:
            break

    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin before 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Hexagonal lattice initialization with various angles
    if time.time() < deadline - 60:
        for angle_deg in [0, 30, 60, 90]:
            if time.time() > deadline - 30:
                break
            
            # Try different spacing factors
            for spacing_factor in [0.95, 0.98, 1.0, 1.02]:
                spacing = 2.0 * spacing_factor / (np.sqrt(N) + 1.0)
                points = _hex_lattice_init(spacing, angle_deg)
                
                if points is not None:
                    score = _compute_score(points)
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
    
    # Strategy 2: Radial ring initialization
    if time.time() < deadline - 60:
        points = _radial_ring_init()
        if points is not None:
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # Local optimization on best result using L-BFGS-B with soft-min
    if best_points is not None and time.time() < deadline - 20:
        optimized = _local_optimize_lbfgs(best_points, deadline=deadline - 10)
        opt_score = _compute_score(optimized)
        if opt_score > best_score:
            best_points = optimized
            best_score = opt_score
    
    # Additional gradient-based refinement if time permits
    if best_points is not None and time.time() < deadline - 5:
        refined = _local_optimize_gradient(best_points, max_iter=100, lr=0.003, 
                                           deadline=deadline - 2)
        ref_score = _compute_score(refined)
        if ref_score > best_score:
            best_points = refined
            best_score = ref_score
    
    # Final validation
    if best_points is None or len(best_points) != N:
        # Emergency fallback: simple hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        best_points = _hex_lattice_init(spacing, 0)
        
        if best_points is None or len(best_points) < N:
            # Last resort: uniform grid
            k = int(np.ceil(np.sqrt(N)))
            spacing = 2.0 / (k + 1)
            points = []
            for i in range(k):
                for j in range(k):
                    x = (i + 0.5) * spacing - 1.0
                    y = (j + 0.5) * spacing - 1.0
                    if x**2 + y**2 <= 1.0 - spacing/2:
                        points.append((x, y))
                    if len(points) >= N:
                        break
                if len(points) >= N:
                    break
            best_points = np.array(points[:N])
    
    return best_points[:N]
# EVOLVE-BLOCK-END
