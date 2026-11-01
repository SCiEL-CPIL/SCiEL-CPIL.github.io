"""Packing 50 equal circles in a unit circle - force-based optimization with hexagonal lattice init."""
import time
import numpy as np

N = 50
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # 40s safety margin before 150s kill


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
    # Pairwise distances using broadcasting
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(n, seed=42):
    """
    Generate initial positions on a hexagonal lattice, then select n closest to center.
    This provides a good structured starting point for optimization.
    """
    rng = np.random.RandomState(seed)
    points = []
    # Generate hexagonal grid
    spacing = 0.18
    k = 12
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r < 0.95:
                points.append((x, y))
    
    points = np.array(points)
    # Add small random perturbation to break symmetry
    points += rng.randn(len(points), 2) * 0.005
    
    # Sort by distance from center, take n closest
    dists = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists, kind='stable')[:n]
    return points[idx].copy()


def _random_init(n, seed=42):
    """Random initial positions inside unit circle."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(rng.uniform(0, 1, n))
    pts = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
    return pts


def _force_optimize(positions, max_time=80.0):
    """
    Optimize positions using force-based relaxation with time limit.
    Uses repulsive forces between overlapping circles and boundary constraints.
    """
    start_time = time.time()
    positions = positions.copy()
    n = len(positions)
    
    # Initial radius estimate
    current_r = _score(positions)
    best_r = current_r
    best_positions = positions.copy()
    
    # Adaptive step size
    step = 0.03
    lr = 0.997  # learning rate decay
    patience = 0
    patience_limit = 500
    
    while time.time() - start_time < max_time:
        # Compute forces: repulsion between overlapping circles, attraction to boundary
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion using vectorized operations
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                dist = np.sqrt(np.sum(diff ** 2))
                if dist < 1e-10:
                    # Random small perturbation
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                    continue
                # Repulsive force when too close
                min_dist = 2.0 * current_r
                if dist < min_dist:
                    force_mag = (min_dist - dist) / max(dist, 1e-10)
                    # Cap the force magnitude to avoid instability
                    force_mag = min(force_mag, 2.0)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Boundary constraint: push inward if outside
        for i in range(n):
            r = np.hypot(positions[i, 0], positions[i, 1])
            max_r = 1.0 - current_r
            if r > max_r:
                force_mag = (r - max_r) / max(r, 1e-10)
                force_mag = min(force_mag, 2.0)
                forces[i] -= force_mag * positions[i] / r
        
        # Apply forces with step size
        positions += step * forces
        
        # Keep inside container (hard constraint)
        radii = np.hypot(positions[:, 0], positions[:, 1])
        max_allowed = 1.0 - current_r * 0.999
        mask = radii > max_allowed
        if np.any(mask):
            positions[mask] *= max_allowed / radii[mask]
        
        # Update radius estimate
        new_r = _score(positions)
        if new_r > best_r:
            best_r = new_r
            best_positions = positions.copy()
            patience = 0
        else:
            patience += 1
        
        current_r = max(new_r, best_r * 0.995)
        
        # Decay step size
        step *= lr
        
        # If step gets too small or no improvement for a while, try a restart with larger step
        if step < 0.002 or patience > patience_limit:
            positions = best_positions + np.random.randn(n, 2) * 0.005
            step = 0.02
            current_r = best_r
            patience = 0
            patience_limit = max(patience_limit // 2, 100)
    
    return best_positions, best_r


def _nelder_mead_refine(positions, max_time=20.0):
    """
    Refine positions using Nelder-Mead optimization to maximize the minimum distance.
    This is a local refinement step that can fine-tune the packing.
    """
    start_time = time.time()
    positions = positions.copy()
    n = len(positions)
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Compute negative of the minimum distance (we want to maximize it)
        # Use a smooth approximation for better convergence
        dists = np.hypot(pts[:, 0], pts[:, 1])
        r_cont = 1.0 - np.max(dists)
        
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dmat = np.sqrt(dx ** 2 + dy ** 2)
        np.fill_diagonal(dmat, np.inf)
        min_pair = np.min(dmat)
        r_pair = min_pair / 2.0
        
        r = min(r_cont, r_pair)
        return -r
    
    from scipy.optimize import minimize
    
    x0 = positions.flatten()
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        refined_pts = result.x.reshape(-1, 2)
        # Project inside
        r = np.hypot(refined_pts[:, 0], refined_pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            refined_pts[mask] = refined_pts[mask] / r[mask, None] * 0.9999
        return refined_pts
    except Exception:
        return positions


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0
    
    # Try multiple initializations and optimization strategies
    inits = []
    inits.append(_hex_lattice_init(N, seed=42))
    inits.append(_random_init(N, seed=123))
    inits.append(_hex_lattice_init(N, seed=999))
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 30.0:
            break
        
        # Run force-based optimization
        sa_pts, sa_score = _force_optimize(init_pts, max_time=50.0)
        
        # Nelder-Mead refinement
        refined_pts = _nelder_mead_refine(sa_pts, max_time=15.0)
        score = _score(refined_pts)
        
        if score > best_score:
            best_score = score
            best_pts = refined_pts.copy()
    
    # Final local refinement on the best found
    if best_pts is not None and time.time() < _DEADLINE - 10.0:
        final_pts = _nelder_mead_refine(best_pts, max_time=8.0)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    
    if best_pts is None:
        # Fallback
        best_pts = _hex_lattice_init(N, seed=42)
    
    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r[mask, None] * 0.9999
    
    return best_pts
