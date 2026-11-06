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
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_min = wall.min()
    
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
    configs = [
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 6, 7, 6, 7, 6, 7, 4],
        [8, 7, 6, 7, 6, 7, 6, 3],
        [5, 6, 7, 6, 7, 6, 7, 6],
        [7, 7, 7, 7, 7, 7, 7, 1],
        [6, 6, 6, 6, 6, 6, 6, 6],
        [5, 5, 5, 5, 5, 5, 5, 10],
        [8, 8, 8, 8, 8, 8, 6, 4],
        [9, 8, 8, 8, 8, 8, 5, 4],
        [10, 9, 8, 8, 8, 7, 6, 4],
    ]
    
    best_pts = None
    best_score = -1
    
    for rows in configs:
        if sum(rows) != N:
            continue
            
        pts = []
        max_row = max(rows)
        spacing = 1.0 / max_row
        h = spacing * np.sqrt(3.0) / 2.0
        
        total_height = (len(rows) - 1) * h
        start_y = -total_height / 2.0
        
        for i, n in enumerate(rows):
            y = start_y + i * h
            offset = (0.5 * spacing) if (i % 2 != 0) else 0.0
            xs = np.arange(n) * spacing
            center = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
            xs = xs - center + offset
            for x in xs:
                pts.append([x, y])
        
        pts = np.array(pts)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts

    return best_pts

def _simulated_annealing(pts, t0=10.0, t_end=0.01, iterations=50000, deadline=None):
    """
    Simulated annealing to escape local minima.
    Uses random perturbations with acceptance based on energy difference.
    """
    current = pts.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    rng = np.random.default_rng(42)
    
    for it in range(iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate perturbation: move random subset of points
        n_move = rng.integers(1, N // 2 + 1)
        idx = rng.choice(N, size=n_move, replace=False)
        step_size = 0.01 * (1.0 + 0.5 * np.exp(-it / 1000.0))
        delta = rng.normal(0, step_size, size=(n_move, 2))
        
        new_pts = current.copy()
        new_pts[idx] += delta
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_score = _score(new_pts)
        delta_e = new_score - current_score
        
        # Accept if better, or with probability exp(delta_e / temperature)
        if delta_e > 0 or rng.random() < np.exp(delta_e / max(t0 * np.exp(-it / iterations), 1e-10)):
            current = new_pts
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
    
    return best, best_score

def _local_refine(pts, deadline=None):
    """
    Local refinement using Nelder-Mead with tight tolerances.
    """
    try:
        res = minimize(_neg_score, pts.flatten(), method="Nelder-Mead",
                       options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 2000, 'adaptive': True})
        
        new_pts = res.x.reshape(N, 2)
        new_score = _score(new_pts)
        
        if new_score > _score(pts):
            return new_pts, new_score
        else:
            return pts, _score(pts)
    except Exception:
        return pts, _score(pts)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    best_pts = _make_hex_initial()
    best_score = _score(best_pts)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 1: Coarse Nelder-Mead to get out of initial basin
    if time.time() < deadline - 100.0:
        try:
            res = minimize(_neg_score, best_pts.flatten(), method="Nelder-Mead",
                           options={'xatol': 1e-4, 'fatol': 1e-5, 'maxiter': 2000, 'adaptive': True})
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
    
    # Phase 2: Simulated annealing with multiple restarts from different temperatures
    sa_configs = [
        (20.0, 30000),
        (10.0, 40000),
        (5.0, 50000),
    ]
    
    for sa_t0, sa_iters in sa_configs:
        if time.time() > deadline - 30.0:
            break
        
        sa_pts, sa_score = _simulated_annealing(
            best_pts, t0=sa_t0, t_end=0.01, iterations=sa_iters, 
            deadline=deadline - 25.0
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts
        
        # Local refinement after each SA phase
        if time.time() < deadline - 20.0:
            refined_pts, refined_score = _local_refine(best_pts, deadline=deadline - 15.0)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined_pts
    
    # Phase 3: Final tight local refinement
    if time.time() < deadline - 5.0:
        final_pts, final_score = _local_refine(best_pts, deadline=deadline - 2.0)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
