"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def _score(pts):
    """Compute packing score: min of half-min-distance and min boundary distance."""
    if len(pts) < 2:
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    return min(min_pair / 2.0, boundary.min())


def _hex_init():
    """Structured hexagonal lattice initialization."""
    spacing = 0.22
    candidates = []
    k = 8
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                candidates.append([x, y])
    candidates = np.array(candidates)
    if len(candidates) < N:
        # Fallback: random points
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.random.uniform(0, 0.8, N)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    # Select N closest to center
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:N]
    return candidates[idx]


def _simulated_annealing(pts, deadline):
    """Simulated annealing with vectorized distance computation."""
    pts = pts.copy()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    t0 = time.time()
    T = 0.15
    cooling = 0.9995
    step_size = 0.05
    n_steps = 0
    
    while time.time() < deadline - 5.0 and T > 1e-6:
        n_steps += 1
        # Pick a random point and perturb
        i = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        new_pts = pts.copy()
        new_pts[i] = pts[i] + delta
        
        # Quick validity check: must be inside unit circle
        if np.linalg.norm(new_pts[i]) >= 1.0:
            T *= cooling
            continue
        
        new_score = _score(new_pts)
        delta_e = new_score - current_score
        
        if delta_e > 0 or np.random.random() < np.exp(delta_e / T):
            pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        
        T *= cooling
        
        # Adapt step size
        if n_steps % 1000 == 0:
            step_size *= 0.95
    
    return best_pts, best_score


def _local_polish(pts, deadline):
    """High-precision local polish using gradient-free optimization."""
    pts = pts.copy()
    current_score = _score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    t0 = time.time()
    step_size = 0.01
    
    while time.time() < deadline - 2.0 and step_size > 1e-8:
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    new_pts = pts.copy()
                    new_pts[i] = pts[i] + [dx, dy]
                    if np.linalg.norm(new_pts[i]) >= 1.0:
                        continue
                    new_score = _score(new_pts)
                    if new_score > current_score + 1e-12:
                        pts = new_pts
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_pts = pts.copy()
        if not improved:
            step_size *= 0.5
    
    return best_pts, best_score


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Phase 1: Structured initialization
    pts = _hex_init()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Phase 2: Simulated annealing with multiple restarts
    n_restarts = 0
    while time.time() < deadline - 30.0:
        n_restarts += 1
        if n_restarts == 1:
            start_pts = best_pts.copy()
        else:
            # Perturb best solution for diversity
            start_pts = best_pts + np.random.normal(0, 0.03, (N, 2))
            # Ensure validity
            start_pts = np.clip(start_pts, -0.95, 0.95)
            norms = np.linalg.norm(start_pts, axis=1)
            mask = norms > 0.95
            start_pts[mask] *= 0.95 / norms[mask, None]
        
        sa_pts, sa_score = _simulated_annealing(start_pts, deadline - 30.0)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
    
    # Phase 3: Local polish
    if time.time() < deadline - 5.0:
        polished_pts, polished_score = _local_polish(best_pts, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts
    
    return best_pts
# EVOLVE-BLOCK-END
