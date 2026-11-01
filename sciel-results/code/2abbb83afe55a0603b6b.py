"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Efficient pairwise distance computation
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    for spacing in np.linspace(0.15, 0.35, 80):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    if best_pts is not None:
        return best_pts
    # Fallback: random points in unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _compute_forces(pts, softness=10.0):
    """Compute repulsion forces using vectorized operations with smooth surrogate."""
    # Pairwise distance matrix (N x N)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 2)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq, 1e10)  # Avoid division by zero on diagonal
    
    # Smooth repulsion: use soft-min surrogate for the min distance
    # Force magnitude: strong when close, decays smoothly
    dist = np.sqrt(dist_sq)
    
    # Repulsion force: F = 1/(d^2 + eps) * direction, but only for close pairs
    # Use a smooth cutoff to avoid discontinuities
    cutoff = 0.5
    weight = np.where(dist < cutoff, 1.0 / (dist_sq + 1e-8), 0.0)
    
    # Force direction: diff / dist (normalized)
    with np.errstate(divide='ignore', invalid='ignore'):
        dir_vec = diff / dist[:, :, np.newaxis]
    dir_vec[~np.isfinite(dir_vec)] = 0.0
    
    # Accumulate forces: sum over all j for each i
    forces = np.sum(weight[:, :, np.newaxis] * dir_vec, axis=1)
    
    # Add boundary attraction: push points outward if too close to center
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_force = np.zeros_like(pts)
    mask = radii < 0.9
    boundary_force[mask] = 0.1 * pts[mask] / (radii[mask, np.newaxis] + 1e-10)
    
    forces += boundary_force
    
    return forces


def _repulsion_optimize(pts0, max_time=120.0):
    """Optimize using repulsion forces between circles (vectorized)."""
    pts = pts0.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Adaptive step size
    step_size = 0.005
    n_iter = 0
    stuck_count = 0
    patience = 100
    
    while time.time() < _DEADLINE:
        n_iter += 1
        
        # Compute forces using vectorized method
        forces = _compute_forces(pts)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Update positions
        new_pts = pts + step_size * forces
        
        # Keep points inside the container
        new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            new_pts[outside] = new_pts[outside] / new_radii[outside, np.newaxis]
        
        # Check if valid
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            stuck_count = 0
            # Slightly increase step size if making good progress
            if n_iter % 50 == 0:
                step_size = min(step_size * 1.05, 0.02)
        else:
            stuck_count += 1
            # Reduce step size if stuck
            if stuck_count > patience:
                step_size *= 0.85
                stuck_count = 0
        
        # Occasionally add random perturbation to escape local minima
        if n_iter % 300 == 0:
            perturb = np.random.uniform(-0.01, 0.01, (N, 2))
            perturbed_pts = pts + perturb
            perturbed_radii = np.hypot(perturbed_pts[:, 0], perturbed_pts[:, 1])
            outside_p = perturbed_radii > 1.0
            if np.any(outside_p):
                perturbed_pts[outside_p] = perturbed_pts[outside_p] / perturbed_radii[outside_p, np.newaxis]
            perturbed_score = _score(perturbed_pts)
            if perturbed_score > best_score:
                best_score = perturbed_score
                best_pts = perturbed_pts.copy()
                pts = perturbed_pts.copy()
                step_size = 0.005
        
        pts = new_pts
    
    return best_pts


def _polish(pts, max_time=10.0):
    """Final high-precision polish pass with tighter tolerances."""
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    step_size = 0.001
    n_iter = 0
    stuck_count = 0
    patience = 50
    
    while time.time() < max_time:
        n_iter += 1
        
        # Compute forces
        forces = _compute_forces(pts)
        
        # Normalize
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Small update
        new_pts = pts + step_size * forces
        
        # Keep inside container
        new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        outside = new_radii > 1.0
        if np.any(outside):
            new_pts[outside] = new_pts[outside] / new_radii[outside, np.newaxis]
        
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > patience:
                step_size *= 0.7
                if step_size < 1e-6:
                    break
                stuck_count = 0
        
        pts = new_pts
    
    return best_pts


def construct():
    start_time = time.time()
    
    # Get initial hexagonal lattice configuration
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # Run repulsion-based optimization
    try:
        if time.time() < _DEADLINE - 15:  # Leave 15s for polish
            opt_pts = _repulsion_optimize(pts0)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
    except Exception:
        pass
    
    # Final polish pass if we have time
    try:
        remaining = _DEADLINE - time.time()
        if remaining > 5.0:
            polished = _polish(best_pts, max_time=min(remaining - 2.0, 10.0))
            polished_score = _score(polished)
            if polished_score > best_score:
                best_pts = polished
                best_score = polished_score
    except Exception:
        pass
    
    # Final safety check: ensure validity
    final_score = _score(best_pts)
    if final_score <= 0:
        # Return the initial hex lattice as fallback
        return pts0
    
    return best_pts

# EVOLVE-BLOCK-END
