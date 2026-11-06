"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def _hex_lattice_params_to_pts(params):
    """
    Convert parametric hex lattice parameters to point coordinates.
    params: [spacing, angle, row_offsets...]
    We use a structured pattern with variable row lengths.
    """
    # Fixed row pattern that sums to 50
    # Using 7 rows with pattern [7, 8, 7, 7, 7, 7, 7] = 50
    row_lengths = [7, 8, 7, 7, 7, 7, 7]
    n_rows = len(row_lengths)
    
    spacing = params[0]
    angle = params[1]
    
    # Row-specific offsets (7 parameters)
    row_offsets = params[2:2+n_rows]
    
    # Vertical spacing
    v_spacing = spacing * np.sqrt(3.0) / 2.0
    
    pts = []
    total_height = (n_rows - 1) * v_spacing
    start_y = -total_height / 2.0
    
    for i, n in enumerate(row_lengths):
        y = start_y + i * v_spacing
        offset = row_offsets[i]
        
        # Generate x positions for this row
        xs = np.arange(n) * spacing
        center = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
        xs = xs - center + offset
        
        for x in xs:
            pts.append([x, y])
    
    pts = np.array(pts)
    
    # Apply rotation
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ rot.T
    
    return pts

def _neg_lattice_score(params):
    """Negative score for lattice parameter optimization."""
    pts = _hex_lattice_params_to_pts(params)
    return -_score(pts)

def _make_hex_initial():
    """
    Create a hexagonal lattice configuration for 50 points using differential evolution.
    """
    # Parameter bounds: [spacing, angle, 7 row offsets]
    bounds = [
        (0.08, 0.15),  # spacing
        (-0.3, 0.3),   # angle
    ] + [(-0.2, 0.2) for _ in range(7)]  # row offsets
    
    # Use differential evolution for global search
    try:
        result = differential_evolution(
            _neg_lattice_score,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            polish=True
        )
        if result.success or result.fun < -0.05:
            best_params = result.x
        else:
            best_params = result.x
    except Exception:
        # Fallback to manual hex pattern
        best_params = np.array([0.11, 0.0] + [0.0]*7)
    
    return _hex_lattice_params_to_pts(best_params)

def _refine_free_points(pts, t0, deadline):
    """
    Refine points with free optimization using Powell method.
    """
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    # Phase 1: Powell optimization
    try:
        res = minimize(_neg_score, best_pts.flatten(), method="Powell",
                       options={'maxiter': 500, 'ftol': 1e-9, 'xtol': 1e-9, 'disp': False})
        
        new_pts = res.x.reshape(N, 2)
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead polishing if time remains
    if time.time() < deadline - 5.0:
        try:
            res2 = minimize(_neg_score, best_pts.flatten(), method="Nelder-Mead",
                            options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 1000, 'adaptive': True})
            
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
    
    # Phase 0: Generate initial hex lattice via differential evolution
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
