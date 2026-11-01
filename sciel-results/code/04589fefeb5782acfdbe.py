"""Packing 50 equal circles in a unit circle - force-based optimization."""
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
    # Pairwise distances using broadcasting
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(spacing):
    """Generate points on a hexagonal lattice inside the unit disk."""
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    return points


def _force_optimize(pts, max_iters=2000, dt=0.01):
    """Force-based optimization: repulsion between circles + attraction to center."""
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        
        # Compute pairwise forces (repulsion)
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random perturbation for coincident points
                    d = np.random.randn(2) * 1e-5
                    dist = np.linalg.norm(d)
                if dist < 0.3:  # Only repel when close
                    force_mag = (0.3 - dist) / dist
                    forces[i] += force_mag * d
                    forces[j] -= force_mag * d
        
        # Attraction to center (weak)
        for i in range(n):
            forces[i] -= 0.01 * pts[i]
        
        # Boundary constraint: push inward if too close to edge
        for i in range(n):
            r = np.linalg.norm(pts[i])
            if r > 0.95:
                forces[i] -= 0.1 * pts[i] / r
        
        # Update positions with adaptive step size
        pts += dt * forces
        
        # Clamp to unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 0.99
        if np.any(mask):
            pts[mask] *= 0.99 / radii[mask]
    
    return pts


def _repulsion_optimize(pts, max_iters=3000):
    """Optimize using inverse-square repulsion and boundary constraints."""
    pts = pts.copy()
    n = len(pts)
    r_target = 0.125  # Target radius
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(2) * 1e-5
                    dist = np.linalg.norm(d)
                # Repulsion when closer than 2*r_target
                min_dist = 2.0 * r_target
                if dist < min_dist * 1.5:
                    # Stronger repulsion when overlapping
                    force_mag = (min_dist * 1.5 - dist) ** 2 / (dist + 1e-10)
                    forces[i] += force_mag * d / dist
                    forces[j] -= force_mag * d / dist
        
        # Weak attraction to center
        for i in range(n):
            forces[i] -= 0.005 * pts[i]
        
        # Boundary repulsion
        for i in range(n):
            r = np.linalg.norm(pts[i])
            if r > 0.9:
                forces[i] -= (r - 0.9) * 2.0 * pts[i] / r
        
        # Adaptive step size
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 1e-10:
            step = 0.05 / max_force
            pts += step * np.tanh(forces)
        
        # Clamp to unit circle
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 0.999
        if np.any(mask):
            pts[mask] *= 0.999 / radii[mask]
        
        # Check convergence
        if it > 100 and max_force < 1e-6:
            break
    
    return pts


def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    best_pts = None
    best_score = 0.0
    
    # Try different initial spacings
    spacings = [0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19]
    
    for spacing in spacings:
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Get initial points from hex lattice
        points = _hex_lattice_init(spacing)
        
        if len(points) < N:
            continue
        
        # Select N points closest to center
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        pts = points[idx].copy()
        
        # Optimize with repulsion
        pts = _repulsion_optimize(pts, max_iters=2000)
        
        # Compute score
        score = _score(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # If we have a good result, try to improve with more iterations
    if best_pts is not None and best_score > 0.12:
        if time.time() < _DEADLINE - 20.0:
            best_pts = _repulsion_optimize(best_pts, max_iters=5000)
            new_score = _score(best_pts)
            if new_score > best_score:
                best_score = new_score
    
    # Final fallback: return hex lattice if optimization failed
    if best_pts is None or best_score <= 0:
        spacing = 0.25
        points = _hex_lattice_init(spacing)
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        best_pts = points[idx]
    
    return best_pts
# EVOLVE-BLOCK-END
