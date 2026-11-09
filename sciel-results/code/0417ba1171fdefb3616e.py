"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

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


def _soft_min(values, beta=50.0):
    """Smooth approximation of min via log-sum-exp."""
    m = np.min(values)
    return m - np.log(np.sum(np.exp(beta * (values - m)))) / beta


def _smooth_objective(pts_flat, beta=50.0):
    """
    Returns a smooth surrogate for the negative radius using KDTree for speed.
    """
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    # Boundary clearances
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Use KDTree to find nearest neighbors (the binding constraint)
    tree = cKDTree(pts)
    # Get 5 nearest neighbors for each point
    k = min(5, n - 1)
    dists, _ = tree.query(pts, k=k + 1)  # k+1 includes self
    # dists[:, 0] is 0 (self), dists[:, 1:] are nearest neighbors
    nn_dists = dists[:, 1:]  # shape (n, k)
    # The minimum pairwise distance is min over all nn_dists
    # Use soft-min over the nearest-neighbor distances and boundary
    all_vals = np.concatenate([boundary, nn_dists.flatten()])
    return -_soft_min(all_vals, beta)


def _smooth_gradient(pts_flat, beta=50.0):
    """
    Gradient of the smooth objective w.r.t. flattened points using KDTree.
    """
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Use KDTree to find nearest neighbors
    tree = cKDTree(pts)
    k = min(5, n - 1)
    dists, indices = tree.query(pts, k=k + 1)
    # indices[:, 0] is self, indices[:, 1:] are nearest neighbors
    nn_indices = indices[:, 1:]  # shape (n, k)
    nn_dists = dists[:, 1:]  # shape (n, k)
    
    # Combine boundary and nn distances for soft-min
    all_vals = np.concatenate([boundary, nn_dists.flatten()])
    
    # Soft-min weights
    m = np.min(all_vals)
    exps = np.exp(beta * (all_vals - m))
    weights = exps / np.sum(exps)
    
    grad = np.zeros_like(pts_flat)
    
    # Boundary part
    for i in range(n):
        if norms[i] > 1e-10:
            grad[2*i:2*i+2] -= weights[i] * pts[i] / norms[i]
    
    # Pairwise part: for each point i and its k nearest neighbors j
    for i in range(n):
        for kk in range(k):
            j = nn_indices[i, kk]
            w = weights[n + i * k + kk]
            diff = pts[i] - pts[j]
            dist = nn_dists[i, kk]
            if dist > 1e-10:
                grad[2*i:2*i+2] += w * diff / (2 * dist)
                grad[2*j:2*j+2] -= w * diff / (2 * dist)
    
    return grad


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


def _refine_simulated_annealing(points, deadline, max_iters=5000, T_start=0.01, T_end=0.0001):
    """Simulated annealing local refinement; respects the deadline."""
    if points is None:
        return None, 0.0
    points = points.copy()
    best_points = points.copy()
    best_r = _compute_radius(points)
    current_r = best_r
    rng = np.random.default_rng(777)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (i / max_iters)
        
        idx = rng.integers(0, len(points))
        original = points[idx].copy()
        
        # Random perturbation
        dx = rng.uniform(-0.05, 0.05)
        dy = rng.uniform(-0.05, 0.05)
        points[idx] = original + np.array([dx, dy])
        
        new_r = _compute_radius(points)
        delta_r = new_r - current_r
        
        # Accept if better or with probability exp(delta_r / T)
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
        else:
            points[idx] = original
            
    return best_points, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s wall-clock limit

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
    # First pass: hill climbing on top candidates
    for cand_idx in range(min(3, len(candidates))):
        if time.time() > deadline - 60.0:
            break
        cand_pts, cand_r = candidates[cand_idx]
        refined_pts, refined_r = _refine_local(cand_pts, deadline - 55.0,
                                               max_iters=2000, step_size=0.015)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Second pass: smooth optimization via L-BFGS-B (faster with KDTree)
    if time.time() < deadline - 25.0:
        try:
            from scipy.optimize import minimize
            # Try multiple beta values; start with sharper for better resolution
            for beta in [80.0, 50.0, 30.0, 120.0]:
                if time.time() > deadline - 10.0:
                    break
                pts_flat = best_points.flatten()
                res = minimize(
                    _smooth_objective,
                    pts_flat.copy(),
                    jac=_smooth_gradient,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'gtol': 1e-7, 'ftol': 1e-14}
                )
                if res.x is not None:
                    new_pts = res.x.reshape(-1, 2)
                    new_r = _compute_radius(new_pts)
                    if new_r > best_r:
                        best_points, best_r = new_pts, new_r
        except Exception:
            pass

    # Third pass: simulated annealing for exploration
    if time.time() < deadline - 15.0:
        refined_pts, refined_r = _refine_simulated_annealing(best_points, deadline - 10.0,
                                                             max_iters=2000, T_start=0.008, T_end=0.0001)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Final polish: very tight hill climbing with small steps
    if time.time() < deadline - 3.0:
        refined_pts, refined_r = _refine_local(best_points, deadline,
                                               max_iters=2000, step_size=0.003)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    # Final micro-polish: even tighter steps on the best configuration
    if time.time() < deadline - 0.5:
        refined_pts, refined_r = _refine_local(best_points, deadline,
                                               max_iters=500, step_size=0.001)
        if refined_pts is not None and refined_r > best_r:
            best_points, best_r = refined_pts, refined_r

    return best_points

# EVOLVE-BLOCK-END
