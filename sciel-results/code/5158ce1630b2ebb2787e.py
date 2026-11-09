"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """Feasible common radius: min of boundary clearance and half min pairwise distance."""
    if points is None or len(points) < 2:
        return 0.0
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dist_boundary))
    d = pdist(points)
    min_pair = float(np.min(d)) / 2.0
    return min(min_boundary, min_pair)


def _hexagonal_lattice(spacing):
    """Hexagonal lattice points inside the unit disk (with margin), closest to centre."""
    k = int(1.0 / spacing) + 3
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _concentric_rings(n_rings):
    """Concentric rings packing; returns points or None."""
    if n_rings <= 0:
        return None
    if n_rings == 1:
        ring_counts = [N]
    else:
        remaining = N - 1
        base = remaining // (n_rings - 1)
        extra = remaining % (n_rings - 1)
        ring_counts = [1] + [base + (1 if i < extra else 0) for i in range(n_rings - 1)]
    pts = []
    if ring_counts[0] > 0:
        pts.append([0.0, 0.0])
    for ring_idx in range(1, n_rings):
        count = ring_counts[ring_idx]
        if count <= 0:
            continue
        ring_radius = ring_idx / n_rings * 0.95
        for i in range(count):
            angle = 2 * np.pi * i / count
            pts.append([ring_radius * np.cos(angle), ring_radius * np.sin(angle)])
    if len(pts) != N:
        return None
    return np.array(pts)


def _square_grid(spacing):
    """Square grid points inside the unit disk, closest to centre."""
    k = int(1.0 / spacing) + 3
    pts = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing
            y = j * spacing
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                pts.append((x, y))
    if len(pts) < N:
        return None
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]


def _soft_min_obj(points, alpha=50.0):
    """
    Soft-min objective: maximize t such that all constraints hold.
    We minimize -t + penalty for violations.
    Constraints:
      - |c_i| + r <= 1  for all i
      - |c_i - c_j| >= 2r  for all i<j
    Using soft-min: r_soft = min over constraints of (constraint_value)
    We want to maximize r_soft.
    
    Alternative: define f(points) = min_i (1 - |c_i|, min_{i<j} |c_i-c_j|/2)
    Use log-sum-exp to get a smooth approximation of the min.
    """
    n = len(points)
    # Boundary distances
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    # Pairwise distances / 2
    d = pdist(points)
    pair_dists = d / 2.0
    
    # Combine all constraints
    all_vals = np.concatenate([dists_boundary, pair_dists])
    
    # Soft min via log-sum-exp: min(x) ≈ -1/alpha * log(sum(exp(-alpha*x)))
    # To avoid overflow, shift by min
    m = np.min(all_vals)
    # If m is very negative, the soft min is close to m
    if alpha > 0:
        # exp(-alpha * (all_vals - m)) is safe since all_vals - m >= 0
        shifted = all_vals - m
        log_sum = np.log(np.sum(np.exp(-alpha * shifted)))
        soft_min_val = m - log_sum / alpha
    else:
        soft_min_val = m
    
    return -soft_min_val  # We want to minimize this (i.e., maximize soft_min_val)


def _soft_min_grad(points, alpha=50.0):
    """
    Gradient of the soft-min objective w.r.t. points.
    """
    n = len(points)
    pts = points.reshape(-1, 2)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    dists_boundary = 1.0 - norms
    
    # Pairwise distances / 2
    d = pdist(pts)
    pair_dists = d / 2.0
    
    all_vals = np.concatenate([dists_boundary, pair_dists])
    m = np.min(all_vals)
    
    if alpha <= 0:
        # Just return gradient of the min (subgradient)
        # Find which constraint is active
        min_idx = np.argmin(all_vals)
        if min_idx < n:
            # Boundary constraint for point min_idx
            grad = np.zeros_like(pts)
            if norms[min_idx] > 1e-12:
                grad[min_idx] = -pts[min_idx] / norms[min_idx]
            else:
                grad[min_idx] = np.array([0.0, 0.0])
            return grad
        else:
            # Pairwise constraint
            pair_idx = min_idx - n
            # Convert pair index to (i, j)
            # pdist uses condensed form: pair (i,j) with i<j
            # Index k corresponds to (i, j) where k = i*(2n-i-1)//2 + j - i - 1
            # This is complex; for simplicity, use a different approach
            # For now, return zero gradient (subgradient is complex)
            return np.zeros_like(pts)
    
    # Soft min gradient
    shifted = all_vals - m
    exp_vals = np.exp(-alpha * shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft_min_val = m - log(sum(exp(-alpha*shifted))) / alpha
    # d(soft_min_val)/d(all_vals[k]) = -weights[k] / alpha * (-alpha) = weights[k] ... wait
    # Let's recompute:
    # soft_min_val = m - (1/alpha) * log(sum_k exp(-alpha*(all_vals[k] - m)))
    # d(soft_min_val)/d(all_vals[k]) = -delta_{argmin(k)} + (1/alpha) * d/d(all_vals[k]) log(sum_j exp(-alpha*(all_vals[j]-m)))
    # The second term: (1/alpha) * sum_j [d/d(all_vals[k]) exp(-alpha*(all_vals[j]-m))] / sum_j exp(-alpha*(all_vals[j]-m))
    # = (1/alpha) * sum_j [-alpha * delta_{jk} * exp(-alpha*(all_vals[j]-m))] / sum_j exp(-alpha*(all_vals[j]-m))
    # = -sum_j delta_{jk} * exp(-alpha*(all_vals[j]-m)) / sum_j exp(-alpha*(all_vals[j]-m))
    # = -weights[k]
    # Also d(m)/d(all_vals[k]) = delta_{argmin(k)}
    # So d(soft_min_val)/d(all_vals[k]) = -delta_{argmin(k)} - weights[k]
    
    # This is getting complex. Let's use a simpler approach:
    # Just compute the gradient numerically or use a different formulation.
    
    # For now, return a simple gradient based on the active constraint
    min_idx = np.argmin(all_vals)
    grad = np.zeros_like(pts)
    
    if min_idx < n:
        # Boundary constraint
        if norms[min_idx] > 1e-12:
            grad[min_idx] = -pts[min_idx] / norms[min_idx]
    else:
        # Pairwise constraint - approximate
        pair_idx = min_idx - n
        # For simplicity, just return zero (this is a subgradient)
        pass
    
    return grad


def _lbfgs_refine(points, deadline, alpha=50.0, max_iter=1000):
    """
    Use L-BFGS to minimize the soft-min objective.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    
    def obj(x):
        if time.time() > deadline:
            return np.inf
        pts = x.reshape(-1, 2)
        return _soft_min_obj(pts, alpha)
    
    # Use L-BFGS-B for bounded optimization
    # But we don't have explicit bounds; the soft-min handles the constraints
    # Use L-BFGS instead
    try:
        result = minimize(
            obj,
            points.reshape(-1),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-6, 'maxls': 20}
        )
        if result.success or result.fun < _soft_min_obj(points, alpha):
            refined_pts = result.x.reshape(-1, 2)
            # Ensure points are within the unit disk
            norms = np.linalg.norm(refined_pts, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                refined_pts[mask] = refined_pts[mask] / norms[mask, None] * 0.999
            return refined_pts
        else:
            return points
    except Exception:
        return points


def _refine_local(points, deadline, max_iters=2000, step_size=0.02):
    """Hill-climbing local refinement; respects the deadline."""
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _compute_radius(points)
    rng = np.random.default_rng(12345)
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        idx = rng.integers(0, len(points))
        original = points[idx].copy()
        # Try several perturbations of this point; keep the best improving one.
        best_local = original
        best_local_r = best_r
        for _ in range(6):
            if time.time() > deadline:
                break
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            points[idx] = original + np.array([dx, dy])
            r = _compute_radius(points)
            if r > best_local_r:
                best_local_r = r
                best_local = points[idx].copy()
        points[idx] = best_local
        if best_local_r > best_r:
            best_r = best_local_r
            best_points = points.copy()
    return best_points, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s wall-clock limit

    candidates = []

    # Hexagonal lattice: try a range of spacings and keep the best feasible.
    for trial in range(40):
        spacing = 0.05 + trial * 0.005
        pts = _hexagonal_lattice(spacing)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    # Concentric rings: a few ring counts.
    for n_rings in range(3, 20):
        pts = _concentric_rings(n_rings)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    # Square grid: a range of spacings.
    for trial in range(30):
        spacing = 0.05 + trial * 0.005
        pts = _square_grid(spacing)
        if pts is not None:
            r = _compute_radius(pts)
            if r > 0:
                candidates.append((pts, r))

    if not candidates:
        # Fallback: single ring of N points.
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_points, best_r = candidates[0]

    # Refine the best candidates while time permits.
    for cand_idx in range(min(3, len(candidates))):
        if time.time() > deadline - 15.0:
            break
        cand_pts, cand_r = candidates[cand_idx]
        refined_pts, refined_r = _refine_local(cand_pts, deadline,
                                               max_iters=1500, step_size=0.02)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Additional refinement with L-BFGS on the best candidate
    if time.time() < deadline - 30.0:
        lbfgs_pts = _lbfgs_refine(best_points, deadline, alpha=50.0, max_iter=500)
        if lbfgs_pts is not None:
            lbfgs_r = _compute_radius(lbfgs_pts)
            if lbfgs_r > best_r:
                best_points, best_r = lbfgs_pts, lbfgs_r

    # Final polish with local refinement
    if time.time() < deadline - 10.0:
        refined_pts, refined_r = _refine_local(best_points, deadline,
                                               max_iters=1000, step_size=0.01)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    return best_points

# EVOLVE-BLOCK-END
