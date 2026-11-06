"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    # distance to each of the 4 walls
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_min = wall.min()
    
    # nearest neighbour distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    
    return min(wall_min, 0.5 * nn)

def _neg_score(pts):
    return -_score(pts)

def _make_hex_initial():
    """
    Create a hexagonal lattice configuration for 50 points.
    """
    row_lengths = [7, 6, 7, 6, 7, 6, 7, 4]
    n_rows = len(row_lengths)
    spacing = 0.112
    v_spacing = spacing * np.sqrt(3.0) / 2.0
    
    pts = []
    total_height = (n_rows - 1) * v_spacing
    start_y = -total_height / 2.0
    
    for i, n in enumerate(row_lengths):
        y = start_y + i * v_spacing
        if i % 2 == 1:
            offset = spacing / 2.0
        else:
            offset = 0.0
        
        xs = np.arange(n) * spacing
        center = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
        xs = xs - center + offset
        
        for x in xs:
            pts.append([x, y])
    
    return np.array(pts)

def _make_grid_initial():
    """Create a grid configuration."""
    rows = 5
    cols = 10
    pts = []
    x_coords = np.linspace(-0.4, 0.4, cols)
    y_coords = np.linspace(-0.4, 0.4, rows)
    for y in y_coords:
        for x in x_coords:
            pts.append([x, y])
    return np.array(pts)

def _make_random_initial():
    """Create a random configuration."""
    return np.random.uniform(-0.4, 0.4, (N, 2))

def _physics_relax(pts, deadline, steps=3000, jitter=0.01, damping=0.9):
    """
    Physics-based relaxation using repulsive forces.
    """
    pts = pts.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    for step in range(steps):
        if time.time() > deadline - 5.0:
            break
            
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, 1.0)
        
        inv_d2 = 1.0 / (d * d)
        np.fill_diagonal(inv_d2, 0.0)
        
        force_mag = inv_d2 * 0.01
        forces += np.sum(diff * force_mag[:, :, None], axis=1)
        
        # Wall repulsion
        r_guess = best_score
        margin = 0.1
        
        x = pts[:, 0]
        dist_left = x + 0.5
        mask_left = dist_left < r_guess
        forces[mask_left, 0] += margin * (r_guess - dist_left[mask_left])
        
        dist_right = 0.5 - x
        mask_right = dist_right < r_guess
        forces[mask_right, 0] -= margin * (r_guess - dist_right[mask_right])
        
        y = pts[:, 1]
        dist_bottom = y + 0.5
        mask_bottom = dist_bottom < r_guess
        forces[mask_bottom, 1] += margin * (r_guess - dist_bottom[mask_bottom])
        
        dist_top = 0.5 - y
        mask_top = dist_top < r_guess
        forces[mask_top, 1] -= margin * (r_guess - dist_top[mask_top])
        
        pts += forces * damping
        pts += np.random.normal(0, jitter, pts.shape)
        pts = np.clip(pts, -0.499, 0.499)
        
        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            
        if step % 100 == 0:
            jitter *= 0.95
            
    return best_pts, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    np.random.seed(42)
    
    best_pts = None
    best_score = 0.0
    
    initializers = [
        _make_hex_initial,
        _make_grid_initial,
        _make_random_initial
    ]
    
    time_per_start = (deadline - t0 - 10.0) / len(initializers)
    
    for init_func in initializers:
        if time.time() > deadline - 15.0:
            break
            
        pts = init_func()
        steps = 2000
        pts, score = _physics_relax(pts, deadline, steps=steps)
        
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    if best_pts is not None and time.time() < deadline - 5.0:
        try:
            res = minimize(_neg_score, best_pts.flatten(), method="Nelder-Mead",
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 200, 'adaptive': True})
            
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
            
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
