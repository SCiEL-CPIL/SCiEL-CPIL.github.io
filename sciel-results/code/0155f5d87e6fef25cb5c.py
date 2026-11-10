import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = len(x_flat)
    
    start_time = time.time()
    time_limit = 575.0  # Safety margin before 630s kill
    
    def _make_densified(path):
        """Build the full densified path: start + path + end, with linear interpolation."""
        full = [(0.0, 0.0, 0.0)]
        for p in path:
            full.append((float(p[0]), float(p[1]), float(p[2])))
        full.append((-2.0, 0.0, 90.0))
        dens = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]
            p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m):
                t = (k + 1) / m
                dens.append((
                    p1[0] + (p2[0] - p1[0]) * t,
                    p1[1] + (p2[1] - p1[1]) * t,
                    p1[2] + (p2[2] - p1[2]) * t,
                ))
        dens.append(full[-1])
        return dens
    
    def _placement_mask(cx, cy, ca_deg):
        """Boolean mask of grid points inside corridor placement (cx, cy, ca_deg)."""
        ca = np.deg2rad(ca_deg)
        cos_a = np.cos(ca)
        sin_a = np.sin(ca)
        dx = x_flat - cx
        dy = y_flat - cy
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return in_arm1 | in_arm2
    
    def evaluate_path(path):
        """Full evaluation: returns area score."""
        dens = _make_densified(path)
        mask = np.ones(n_points, dtype=bool)
        for (cx, cy, ca_deg) in dens:
            m = _placement_mask(cx, cy, ca_deg)
            mask &= m
            if not np.any(mask):
                return 0.0
        return float(np.sum(mask)) * 0.0001
    
    # Parametric path representation using Fourier series
    # Path is defined by control points that we optimize
    # We use a smooth parameterization: x(t), y(t), a(t) where t in [0, 1]
    
    def path_from_params(params, n_points_path=8):
        """Convert parameter vector to a path of waypoints."""
        # params: [x0, y0, a0, x1, y1, a1, ..., x7, y7, a7]
        # We use a monotonic angle constraint: angles must be non-decreasing
        n = n_points_path
        xs_path = params[0:n]
        ys_path = params[n:2*n]
        as_path = params[2*n:3*n]
        
        # Enforce monotonic angles: clip to be non-decreasing
        for i in range(1, n):
            as_path[i] = max(as_path[i], as_path[i-1])
        
        # Clip angles to [0, 90]
        as_path = np.clip(as_path, 0.0, 90.0)
        
        path = []
        for i in range(n):
            path.append((float(xs_path[i]), float(ys_path[i]), float(as_path[i])))
        return path
    
    def objective(params):
        """Negative score to minimize."""
        path = path_from_params(params)
        score = evaluate_path(path)
        return -score
    
    # Initial path
    initial_path = [
        (-0.1, 0.1, 15.0),
        (-0.2, 0.2, 20.0),
        (-0.3, 0.3, 34.0),
        (-0.3, 0.3, 40.0),
        (-0.4, 0.3, 50.0),
        (-0.5, 0.2, 55.0),
        (-0.6, 0.2, 75.0),
        (-0.7, 0.3, 86.0),
    ]
    
    n_path = len(initial_path)
    initial_params = np.array([
        initial_path[i][0] for i in range(n_path)
    ] + [
        initial_path[i][1] for i in range(n_path)
    ] + [
        initial_path[i][2] for i in range(n_path)
    ])
    
    best_score = evaluate_path(initial_path)
    best_path = initial_path[:]
    
    # Phase 1: Nelder-Mead optimization on the parameter vector
    if time.time() - start_time < time_limit - 120:
        try:
            result = minimize(
                objective,
                initial_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-4,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            if result.fun < -best_score:
                best_score = -result.fun
                best_path = path_from_params(result.x)
        except Exception:
            pass
    
    # Phase 2: Coordinate descent polish with multiple scales
    polish_deadline = time_limit - 5.0
    
    scales = [
        (0.05, 1.0),
        (0.02, 0.5),
        (0.01, 0.25),
        (0.005, 0.1),
    ]
    
    for polish_xy, polish_a in scales:
        if time.time() > polish_deadline:
            break
        for pass_iter in range(3):
            if time.time() > polish_deadline:
                break
            improved = False
            for idx in range(len(best_path)):
                if time.time() > polish_deadline:
                    break
                p = best_path[idx]
                best_local = best_score
                best_local_p = p
                
                # Try perturbations in x
                for dx in (-polish_xy, -polish_xy * 0.5, polish_xy * 0.5, polish_xy):
                    test_p = (p[0] + dx, p[1], p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p
                
                # Try perturbations in y
                for dy in (-polish_xy, -polish_xy * 0.5, polish_xy * 0.5, polish_xy):
                    test_p = (p[0], p[1] + dy, p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p
                
                # Try perturbations in a
                for da in (-polish_a, -polish_a * 0.5, polish_a * 0.5, polish_a):
                    new_a = np.clip(p[2] + da, 0.0, 90.0)
                    test_p = (p[0], p[1], new_a)
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p
                
                if best_local > best_score:
                    best_path[idx] = best_local_p
                    best_score = best_local
                    improved = True
            
            if not improved:
                break
    
    # Phase 3: Fine-tune with very small steps
    if time.time() < polish_deadline:
        fine_xy = 0.002
        fine_a = 0.05
        for pass_iter in range(5):
            if time.time() > polish_deadline:
                break
            improved = False
            for idx in range(len(best_path)):
                if time.time() > polish_deadline:
                    break
                p = best_path[idx]
                best_local = best_score
                best_local_p = p
                
                for dx in (-fine_xy, fine_xy):
                    test_p = (p[0] + dx, p[1], p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p
                
                for dy in (-fine_xy, fine_xy):
                    test_p = (p[0], p[1] + dy, p[2])
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p
                
                for da in (-fine_a, fine_a):
                    new_a = np.clip(p[2] + da, 0.0, 90.0)
                    test_p = (p[0], p[1], new_a)
                    test_path = best_path[:]
                    test_path[idx] = test_p
                    s = evaluate_path(test_path)
                    if s > best_local:
                        best_local = s
                        best_local_p = test_p
                
                if best_local > best_score:
                    best_path[idx] = best_local_p
                    best_score = best_local
                    improved = True
            
            if not improved:
                break
    
    return np.array(best_path)
