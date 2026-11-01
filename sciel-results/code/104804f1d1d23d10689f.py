"""Packing 50 equal circles in a unit circle - vectorized force optimization."""
import time
import numpy as np

N = 50
_DEADLINE = time.time() + 145.0


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(spacing):
    """Generate N points on a hexagonal lattice inside the unit disk."""
    k = int(1.0 / spacing) + 3
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    if len(points) < N:
        return None
    radii = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radii, kind='stable')[:N]
    return points[idx].copy()


def _optimize(pts, max_iters=8000):
    """Vectorized repulsion-based optimization to maximize minimum distance."""
    pts = pts.copy()
    n = len(pts)
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    dt = 0.015
    r_target = 0.252
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 3.0:
            break
        
        d = pts[i_idx] - pts[j_idx]
        dist = np.hypot(d[:, 0], d[:, 1])
        
        dist_safe = np.maximum(dist, 1e-10)
        force_mag = np.zeros(len(i_idx))
        mask = dist < r_target
        force_mag[mask] = (r_target - dist[mask]) / dist_safe[mask]
        
        d_unit = d / dist_safe[:, np.newaxis]
        
        forces = np.zeros_like(pts)
        np.add.at(forces, i_idx, force_mag[:, np.newaxis] * d_unit)
        np.add.at(forces, j_idx, -force_mag[:, np.newaxis] * d_unit)
        
        forces -= 0.001 * pts
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        bmask = radii > 0.98
        if np.any(bmask):
            forces[bmask] -= 0.05 * pts[bmask] / radii[bmask][:, np.newaxis]
        
        pts += dt * forces
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        cmask = radii > 0.999
        if np.any(cmask):
            pts[cmask] *= 0.999 / radii[cmask]
        
        if it % 200 == 0 and it > 0:
            score = _score(pts)
            if score > 0.1255:
                break
    
    return pts


def construct():
    """Construct optimal packing of 50 circles in unit circle."""
    best_pts = None
    best_score = 0.0
    
    spacings = [0.252, 0.25, 0.248, 0.245, 0.24, 0.235, 0.23, 0.225, 0.22]
    
    for spacing in spacings:
        if time.time() > _DEADLINE - 25.0:
            break
        
        points = _hex_lattice_init(spacing)
        if points is None:
            continue
        
        pts = _optimize(points, max_iters=6000)
        score = _score(pts)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is not None and best_score > 0.1:
        for r in range(3):
            if time.time() > _DEADLINE - 10.0:
                break
            pts = best_pts + np.random.randn(N, 2) * 0.01
            pts = _optimize(pts, max_iters=4000)
            score = _score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
    
    if best_pts is None or best_score <= 0:
        points = _hex_lattice_init(0.25)
        if points is not None:
            best_pts = points
        else:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radii = np.full(N, 0.5)
            best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return best_pts
