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
    Uses a structured pattern with 8 rows: [7, 6, 7, 6, 7, 6, 7, 4] = 50
    """
    row_lengths = [7, 6, 7, 6, 7, 6, 7, 4]
    n_rows = len(row_lengths)
    
    # Optimal spacing for hex packing
    spacing = 0.112
    v_spacing = spacing * np.sqrt(3.0) / 2.0
    
    pts = []
    total_height = (n_rows - 1) * v_spacing
    start_y = -total_height / 2.0
    
    for i, n in enumerate(row_lengths):
        y = start_y + i * v_spacing
        # Alternate horizontal offset for hexagonal packing
        if i % 2 == 1:
            offset = spacing / 2.0
        else:
            offset = 0.0
        
        # Generate x positions for this row
        xs = np.arange(n) * spacing
        center = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
        xs = xs - center + offset
        
        for x in xs:
            pts.append([x, y])
    
    return np.array(pts)

def _gradient_descent(pts, deadline, lr=0.001, n_steps=2000):
    """
    Simple gradient descent to maximize minimum distance.
    Uses repulsive forces between all pairs and attractive forces to center.
    """
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Compute distances
        diff = best_pts[:, None, :] - best_pts[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Find closest pair
        min_d = d.min()
        min_idx = np.unravel_index(np.argmin(d), d.shape)
        
        # Compute forces
        forces = np.zeros_like(best_pts)
        
        # Repulsive force from closest pair
        i, j = min_idx
        direction = best_pts[i] - best_pts[j]
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
            forces[i] += direction * 0.5
            forces[j] -= direction * 0.5
        
        # Wall repulsion
        for k in range(N):
            for dim in range(2):
                if best_pts[k, dim] < -0.5 + best_score:
                    forces[k, dim] += 0.5
                elif best_pts[k, dim] > 0.5 - best_score:
                    forces[k, dim] -= 0.5
        
        # Update
        best_pts += lr * forces
        
        # Clip to valid range
        best_pts = np.clip(best_pts, -0.49, 0.49)
        
        # Check score
        new_score = _score(best_pts)
        if new_score > best_score:
            best_score = new_score
        else:
            # Revert and reduce learning rate
            best_pts -= lr * forces
            lr *= 0.9
        
        # Adaptive learning rate based on progress
        if step % 100 == 0:
            lr = max(0.0001, min(0.01, lr * 1.1))
    
    return best_pts, best_score

def _refine_free_points(pts, t0, deadline):
    """
    Refine points with free optimization using Powell method.
    """
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Phase 1: Gradient descent
    if time.time() < deadline - 20.0:
        best_pts, best_score = _gradient_descent(best_pts, deadline)
    
    # Phase 2: Powell optimization
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_neg_score, best_pts.flatten(), method="Powell",
                           options={'maxiter': 200, 'ftol': 1e-9, 'xtol': 1e-9, 'disp': False})
            
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polishing if time remains
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(_neg_score, best_pts.flatten(), method="Nelder-Mead",
                            options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 500, 'adaptive': True})
            
            new_pts = res2.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin for 150s limit
    
    # Phase 0: Generate initial hex lattice
    best_pts = _make_hex_initial()
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        # Fallback: Random points
        np.random.seed(42)
        best_pts = np.random.uniform(-0.4, 0.4, (N, 2))
        
    best_score = _score(best_pts)
    
    # Phase 1: Refine with free-point optimization
    best_pts, best_score = _refine_free_points(best_pts, t0, deadline)
    
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
