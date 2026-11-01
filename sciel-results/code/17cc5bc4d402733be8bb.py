"""Circle packing: 50 equal circles in unit disk.
Strategy: Force-directed simulation with 1/r^4 repulsion and hard constraint projection."""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

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

def _force_energy(points, target_r):
    """
    Compute force-based energy and forces.
    Uses 1/r^4 repulsion for pairs and a spring-like force to maintain target distance.
    Also includes boundary penalty.
    """
    pts = points.copy()
    n = len(pts)
    forces = np.zeros_like(pts)
    energy = 0.0
    
    # Pairwise repulsion (1/r^4)
    # We want to push points apart if they are closer than 2*target_r
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-12:
                # Random kick if coincident
                forces[i] += np.random.randn(2) * 0.1
                forces[j] -= np.random.randn(2) * 0.1
                energy += 1.0
                continue
            
            # Force magnitude: k / dist^4
            # We want this to be strong when close
            # Let's scale it so that at dist = 2*target_r, force is moderate
            # Actually, simple repulsion: F = C / dist^4
            # Direction: away from each other
            unit_vec = diff / dist
            f_mag = 1.0 / (dist ** 4)
            forces[i] += f_mag * unit_vec
            forces[j] -= f_mag * unit_vec
            energy += f_mag
    
    # Boundary penalty: push inward if outside
    for i in range(n):
        r = np.linalg.norm(pts[i])
        if r > 1.0:
            # Push inward
            unit_vec = pts[i] / r
            f_mag = (r - 1.0) ** 2 * 100.0
            forces[i] -= f_mag * unit_vec
            energy += f_mag
        elif r > 1.0 - 2.0 * target_r:
            # Gentle push inward to keep margin
            unit_vec = pts[i] / max(r, 1e-12)
            f_mag = (1.0 - r - 2.0 * target_r) ** 2 * 10.0
            forces[i] -= f_mag * unit_vec
            energy += f_mag
            
    return energy, forces

def _force_optimize(points, target_r, max_iter=1000, step_size=0.01, deadline=None):
    """
    Optimize positions using force-directed relaxation.
    """
    pts = points.copy()
    current_energy = np.inf
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5.0:
            break
            
        energy, forces = _force_energy(pts, target_r)
        
        # Simple gradient descent step
        pts_new = pts - step_size * forces
        
        # Hard constraint: project to unit disk
        r = np.linalg.norm(pts_new, axis=1)
        mask = r > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / r[mask][:, np.newaxis]
            
        # Check if energy improved
        new_energy, _ = _force_energy(pts_new, target_r)
        
        if new_energy < current_energy:
            pts = pts_new
            current_energy = new_energy
        else:
            # Try smaller step
            step_size *= 0.9
            if step_size < 1e-8:
                break
                
    return pts

def _nelder_mead_polish(points, max_iter=500, deadline=None):
    """
    Polish using Nelder-Mead on the raw min-score objective.
    """
    def objective(x):
        pts = x.reshape(-1, 2)
        # Penalty for going outside
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return 10.0 + np.sum(dists[dists > 1.0] - 1.0)
        return -_compute_score(pts)
    
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        new_points = res.x.reshape(-1, 2)
        # Ensure validity
        if np.all(np.linalg.norm(new_points, axis=1) < 1.0):
            return new_points
    except Exception:
        pass
    return points

def _hex_init(n, seed=0):
    """Hexagonal lattice init."""
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
        
        if len(points) >= n:
            order = np.argsort(radius[mask], kind="stable")
            selected = points[order[:n]]
            score = _compute_score(selected)
            if best_points is None or score > _compute_score(best_points):
                best_points = selected.copy()
                best_spacing = spacing
    
    if best_points is not None:
        return best_points
    
    return _random_init(n, seed)

def _random_init(n, seed):
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Strategy:
    # 1. Generate diverse initializations
    # 2. For each, run Force-Directed Optimization with target radius
    # 3. Polish with Nelder-Mead
    
    inits = []
    # Hex inits
    inits.append(_hex_init(N, seed=0))
    inits.append(_hex_init(N, seed=1))
    # Random inits
    for s in range(5):
        inits.append(_random_init(N, seed=s))
    
    # Target radius estimate
    target_r = 0.125
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
            
        current_pts = init_pts.copy()
        
        # Force optimization
        current_pts = _force_optimize(current_pts, target_r, max_iter=2000, step_size=0.005, deadline=deadline)
        
        # Polish with Nelder-Mead
        if time.time() < deadline - 5.0:
            polished_pts = _nelder_mead_polish(current_pts, max_iter=500, deadline=deadline)
            score = _compute_score(polished_pts)
            if score > best_score:
                best_score = score
                best_points = polished_pts.copy()
        
        # Also check unpolished
        score_unpolished = _compute_score(current_pts)
        if score_unpolished > best_score:
            best_score = score_unpolished
            best_points = current_pts.copy()
            
    # Fallback
    if best_points is None:
        best_points = _hex_init(N, seed=0)
        
    return best_points

# EVOLVE-BLOCK-END
