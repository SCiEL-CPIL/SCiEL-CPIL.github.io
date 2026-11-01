"""Packing 50 equal circles in a unit circle - multi-start force-based optimization."""
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
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_pair, r_cont)


def _hex_lattice_init():
    """Generate a hexagonal lattice initialization."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    for _ in range(100):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Sort by distance from center, take first N
            idx = np.argsort(radius[mask])[:N]
            return points[idx]
        spacing *= 0.98
    # Fallback: uniform grid
    pts = np.random.randn(N, 2) * 0.5
    return pts


def _random_init():
    """Random initialization inside unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _ring_init():
    """Ring-based initialization with concentric rings."""
    pts = np.zeros((N, 2))
    remaining = N
    ring_idx = 0
    while remaining > 0:
        if ring_idx == 0:
            n_ring = 1
        else:
            # Approximate number of points that fit on a ring
            n_ring = min(remaining, int(6 * ring_idx) + 2)
        if ring_idx == 0:
            pts[:1] = [[0, 0]]
            remaining -= 1
        else:
            angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
            r = 0.85 * (ring_idx / (np.sqrt(N) / 2))
            r = min(r, 0.95)
            pts[1 - ring_idx + ring_idx: 1 - ring_idx + ring_idx + n_ring] = np.column_stack([
                r * np.cos(angles), r * np.sin(angles)
            ])
            remaining -= n_ring
        ring_idx += 1
    # Shuffle
    idx = np.random.permutation(N)
    return pts[idx]


def _force_optimize(pts, max_iter=5000, lr=0.01):
    """Force-based optimization using repulsion and container constraint."""
    pts = pts.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Adaptive learning rate
    step = lr
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        dx = pts[:, 0:1] - pts[0:1, 0]
        dy = pts[:, 1:2] - pts[0:1, 1]
        dmat = np.sqrt(dx ** 2 + dy ** 2)
        np.fill_diagonal(dmat, np.inf)
        
        # Repulsive force: 1/d^2 for close pairs
        for i in range(N):
            for j in range(i + 1, N):
                d = dmat[i, j]
                if d < 0.3:  # Only act on relatively close pairs
                    f = 1.0 / (d * d + 1e-6)
                    fx = (pts[i, 0] - pts[j, 0]) / d * f
                    fy = (pts[i, 1] - pts[j, 1]) / d * f
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Container constraint: push inward if too close to boundary
        dists = np.hypot(pts[:, 0], pts[:, 1])
        for i in range(N):
            if dists[i] > 0.85:
                f_cont = (dists[i] - 0.85) * 5.0
                forces[i, 0] -= pts[i, 0] / dists[i] * f_cont
                forces[i, 1] -= pts[i, 1] / dists[i] * f_cont
        
        # Update positions
        pts += step * forces
        
        # Clip to unit circle
        dists_new = np.hypot(pts[:, 0], pts[:, 1])
        mask = dists_new > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists_new[mask][:, None]
        
        # Score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            no_improve = 0
            step = min(step * 1.05, 0.05)  # Increase step slightly
        else:
            no_improve += 1
            if no_improve > 50:
                step *= 0.9  # Decrease step
                no_improve = 0
        
        # Anneal step
        step = max(step * 0.9995, 0.001)
    
    return best_pts, best_score


def _nelder_mead_optimize(pts, max_iter=2000):
    """Nelder-Mead style optimization for final refinement."""
    pts = pts.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    alpha = 0.005  # Perturbation size
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 3:
            break
        
        # Try random perturbations
        for _ in range(10):
            trial = pts.copy()
            idx = np.random.randint(N)
            trial[idx, 0] += np.random.randn() * alpha
            trial[idx, 1] += np.random.randn() * alpha
            
            # Clip to unit circle
            d = np.hypot(trial[idx, 0], trial[idx, 1])
            if d > 1.0:
                trial[idx] = trial[idx] / d
            
            s = _score(trial)
            if s > best_score:
                best_score = s
                best_pts = trial.copy()
                pts = trial.copy()
                no_improve = 0
                break
        else:
            no_improve += 1
            if no_improve > 100:
                alpha *= 0.8
                no_improve = 0
    
    return best_pts, best_score


def construct():
    """Main entry point: find optimal packing of 50 circles in unit circle."""
    start_time = time.time()
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Hexagonal lattice
    inits.append(_hex_lattice_init())
    
    # 2. Random
    for _ in range(3):
        inits.append(_random_init())
    
    # 3. Ring-based
    inits.append(_ring_init())
    
    # 4. Another hex with different spacing
    spacing = 0.2
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    if len(points) >= N:
        idx = np.argsort(radius) [:N]
        inits.append(points[idx])
    
    best_pts = None
    best_score = 0.0
    
    for init in inits:
        if time.time() > _DEADLINE - 20:
            break
        
        # Force optimization
        pts, score = _force_optimize(init, max_iter=3000, lr=0.01)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
        
        # Nelder-Mead refinement
        if time.time() > _DEADLINE - 10:
            break
        pts2, score2 = _nelder_mead_optimize(pts, max_iter=1000)
        if score2 > best_score:
            best_score = score2
            best_pts = pts2.copy()
    
    # Final verification
    if best_pts is None:
        return _hex_lattice_init()
    
    # Ensure validity
    dists = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = dists > 1.0
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / dists[mask][:, None]
    
    return best_pts
