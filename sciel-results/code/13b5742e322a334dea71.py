import numpy as np
import time
from scipy.optimize import minimize

def _compute_max_projection(theta_out, phis, centers, base_vertices):
    """Vectorized computation of max projection for all hexagons."""
    n_hex = len(centers)
    # Outer hexagon face normals
    angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Pre-compute rotated vertices for each hexagon
    # For each hexagon i, vertices = base_vertices @ R(phis[i]).T + centers[i]
    # We need: max over all i, all vertices, all normals of (vertex . normal)
    
    # Vectorize: for each hexagon, compute all 6x6 dot products
    max_proj = 0.0
    for i in range(n_hex):
        c, s = np.cos(phis[i]), np.sin(phis[i])
        rot = np.array([[c, -s], [s, c]])
        verts = base_vertices @ rot.T + centers[i]  # (6, 2)
        dots = verts @ normals.T  # (6, 6)
        max_proj = max(max_proj, np.max(dots))
    
    return max_proj

def _objective_fast(theta_out, phis, centers, base_vertices):
    """Fast objective using precomputed data."""
    max_proj = _compute_max_projection(theta_out, phis, centers, base_vertices)
    return 2.0 * max_proj / np.sqrt(3)

def _objective_smooth_fast(theta_out, phis, centers, base_vertices, beta=100.0):
    """Smooth surrogate using log-sum-exp, vectorized."""
    n_hex = len(centers)
    angles = np.deg2rad(30 + np.arange(6) * 60) + theta_out
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    all_projs = []
    for i in range(n_hex):
        c, s = np.cos(phis[i]), np.sin(phis[i])
        rot = np.array([[c, -s], [s, c]])
        verts = base_vertices @ rot.T + centers[i]
        dots = verts @ normals.T
        all_projs.append(dots.flatten())
    
    all_projs = np.concatenate(all_projs)
    
    m = np.max(all_projs)
    if m < -1e6:
        m = 0.0
    soft_max = m + (1.0 / beta) * np.log(np.sum(np.exp(beta * (all_projs - m))))
    
    return 2.0 * soft_max / np.sqrt(3)

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses a flower packing (1 center, 6 ring 1, 4 ring 2) with optimized rotations.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Centers of the 11 hexagons
    eps = 1e-10
    centers = np.array([[0.0, 0.0]])
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    def objective(params):
        """
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        return _objective_fast(theta_out, phis, centers, base_vertices)
    
    def objective_smooth(params, beta=100.0):
        """
        Smooth objective using log-sum-exp approximation for the max projection.
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        return _objective_smooth_fast(theta_out, phis, centers, base_vertices, beta)
    
    # Initial guess: 
    # Outer rotation 0.
    # Inner rotations: 
    # Try to align faces radially.
    x0 = np.zeros(n_hex + 1)
    x0[0] = 0.0
    for i in range(n_hex):
        cx, cy = centers[i]
        if cx == 0 and cy == 0:
            x0[i+1] = 0.0
        else:
            alpha = np.arctan2(cy, cx)
            # Align face normal (30 deg in base) to alpha
            x0[i+1] = alpha - np.deg2rad(30)
            
    best_score = float('inf')
    best_params = None
    
    # Phase 1: Multi-start optimization using smooth surrogate first, then exact
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 20:
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            # Try all vertices aligned
            x_start = np.zeros(n_hex + 1)
        else:
            # Random perturbations
            x_start = x0 + np.random.uniform(-0.2, 0.2, n_hex + 1)
            x_start[0] = np.random.uniform(0, np.pi/3)
            
        # First, use smooth objective to get into a good basin
        try:
            res_smooth = minimize(objective_smooth, x_start, method='L-BFGS-B',
                                  options={'maxiter': 500, 'disp': False, 'ftol': 1e-12})
            x_start = res_smooth.x
        except Exception:
            pass
        
        # Then refine with exact objective
        try:
            res = minimize(objective, x_start, method='Powell',
                          options={'maxiter': 1000, 'disp': False, 'ftol': 1e-14, 'xtol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return _fallback_solution()
    
    # Phase 2: Nelder-Mead polish on exact objective
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-14, 'maxiter': 5000, 'disp': False})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 2:
        try:
            res3 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-16, 'maxiter': 10000, 'disp': False})
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
            
    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    side = best_score + 1e-9
    
    return inner, (0.0, 0.0), side, np.rad2deg(theta_out)

def _fallback_solution():
    """
    Analytical solution: all inner hexagons at 30 degrees, outer at 30 degrees.
    """
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    side = 4.0 + 1e-7
    return inner, (0.0, 0.0), side, 30.0
