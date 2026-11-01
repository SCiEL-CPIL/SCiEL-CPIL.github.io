"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the packing score: min of pairwise distances/2 and boundary distances."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points are inside the unit circle
    dists_to_origin = np.linalg.norm(points, axis=1)
    if np.any(dists_to_origin >= 1.0):
        return 0.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    # Minimum distance to boundary
    min_boundary = 1.0 - np.max(dists_to_origin)
    return min(min_pairwise, min_boundary)


def _hex_lattice_init(n_points, seed=0):
    """Generate initial positions using hexagonal lattice within unit disk."""
    rng = np.random.RandomState(seed)
    best_spacing = None
    best_points = None
    
    # Try different spacings
    for spacing in np.linspace(0.12, 0.25, 15):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) >= n_points:
            # Sort by distance from center, take closest n_points
            order = np.argsort(radius[mask], kind="stable")
            selected = points[order[:n_points]]
            score = _compute_score(selected)
            if best_points is None or score > _compute_score(best_points):
                best_points = selected.copy()
                best_spacing = spacing
    
    if best_points is not None:
        return best_points
    
    # Fallback: random positions
    theta = rng.uniform(0, 2 * np.pi, n_points)
    r = np.sqrt(rng.uniform(0, 1, n_points))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _force_optimize(points, max_iter=5000, deadline=None):
    """Optimize using force-based method (repulsion + boundary constraint)."""
    points = points.copy()
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Parameters
    step_size = 0.05
    decay = 0.999
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        n = len(points)
        forces = np.zeros_like(points)
        
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.array([0.001, 0.001])
                    dist = np.linalg.norm(diff)
                # Strong repulsion when overlapping
                if dist < 0.3:
                    force_mag = 1.0 / (dist + 1e-6)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Boundary force: push inward if too close to edge
        for i in range(n):
            dist_to_origin = np.linalg.norm(points[i])
            if dist_to_origin > 0.85:
                # Push toward center
                forces[i] -= 0.5 * points[i] / dist_to_origin
        
        # Update positions
        for i in range(n):
            points[i] += step_size * forces[i]
            # Keep within unit circle
            dist = np.linalg.norm(points[i])
            if dist > 1.0:
                points[i] = points[i] / dist
        
        # Check score
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        step_size *= decay
    
    return best_points


def _nelder_mead_optimize(points, max_iter=2000, deadline=None):
    """Optimize using Nelder-Mead on the negative score."""
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Penalty for going outside unit circle
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return 10.0 + np.sum(dists[dists > 1.0] - 1.0)
        return -_compute_score(pts)
    
    try:
        result = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        new_points = result.x.reshape(-1, 2)
        new_score = _compute_score(new_points)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
    except Exception:
        pass
    
    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Try multiple initializations and optimization strategies
    for seed in range(10):
        if time.time() > deadline:
            break
        
        # Hexagonal lattice initialization
        points = _hex_lattice_init(N, seed=seed)
        
        # Force-based optimization
        points = _force_optimize(points, max_iter=2000, deadline=deadline)
        score = _compute_score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Nelder-Mead polish
        if time.time() < deadline - 5.0:
            points = _nelder_mead_optimize(points, max_iter=500, deadline=deadline)
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # Final check
    if best_points is None:
        # Fallback to simple hexagonal
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        order = np.argsort(radius[mask], kind="stable")
        best_points = points[order[:N]]
    
    return best_points
# EVOLVE-BLOCK-END
