# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    def exact_score_from_pts(pts):
        """Compute R = (d_min/d_max)^2 from a (16,2) array."""
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def exact_objective(x):
        pts = x.reshape(n, d)
        return -exact_score_from_pts(pts)

    def smooth_objective(x, t=0.05):
        """Differentiable approximation for local optimization."""
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        dists_norm = dists / d_max
        d_min_norm = -np.log(np.sum(np.exp(-dists_norm / t)) / len(dists_norm)) * t
        return -d_min_norm

    best_x = None
    best_score = 0.0

    def update_best(x, score):
        nonlocal best_x, best_score
        if score > best_score:
            best_score = score
            best_x = x.copy()

    # Phase 1: Global search with Differential Evolution on free coordinates
    bounds = [(-2.0, 2.0)] * (n * d)
    
    try:
        if time.time() < deadline - 90:
            res_de = differential_evolution(
                exact_objective,
                bounds,
                maxiter=300,
                popsize=40,
                seed=42,
                tol=1e-10,
                init='latinhypercube',
                polish=False,
                updating='immediate'
            )
            
            de_score = exact_score_from_pts(res_de.x.reshape(n, d))
            update_best(res_de.x, de_score)
    except Exception:
        pass

    # Phase 2: Nelder-Mead local refinement on exact objective
    if best_x is not None and time.time() < deadline - 30:
        try:
            res_nm = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            nm_score = exact_score_from_pts(res_nm.x.reshape(n, d))
            update_best(res_nm.x, nm_score)
        except Exception:
            pass

    # Phase 3: Multi-start with diverse initializations + Nelder-Mead
    if time.time() < deadline - 20:
        np.random.seed(123)
        inits = []
        
        # Random uniform
        inits.append(np.random.uniform(-1.5, 1.5, (n, d)))
        
        # Points on a circle
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
        
        # Two concentric rings (8+8)
        angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
        angles2 = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 8, endpoint=False)
        ring1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
        ring2 = 0.5 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        inits.append(np.vstack([ring1, ring2]))
        
        # 4x4 grid
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        inits.append(np.column_stack([xx.ravel(), yy.ravel()]))
        
        # Hexagonal 4x4
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3)/2])
        hex_pts = np.zeros((n, 2))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                hex_pts[idx] = i * a1 + j * a2
        inits.append(hex_pts)
        
        # 3+4+3+3+3 pattern (center + rings)
        center = np.array([[0.0, 0.0]])
        r1_angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        r1 = 0.6 * np.column_stack([np.cos(r1_angles), np.sin(r1_angles)])
        r2_angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
        r2 = 1.2 * np.column_stack([np.cos(r2_angles), np.sin(r2_angles)])
        inits.append(np.vstack([center, r1, r2]))
        
        # Additional random starts
        for _ in range(4):
            inits.append(np.random.uniform(-1.5, 1.5, (n, d)))
        
        for init_pts in inits:
            if time.time() > deadline - 15:
                break
            try:
                init_flat = init_pts.ravel()
                res = minimize(
                    exact_objective,
                    init_flat,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                s = exact_score_from_pts(res.x.reshape(n, d))
                update_best(res.x, s)
            except Exception:
                continue

    # Phase 4: Final polish with L-BFGS-B on smooth objective
    if best_x is not None and time.time() < deadline - 5:
        try:
            res_lbfgs = minimize(
                lambda x: smooth_objective(x, t=0.03),
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            s = exact_score_from_pts(res_lbfgs.x.reshape(n, d))
            update_best(res_lbfgs.x, s)
        except Exception:
            pass

    # Fallback
    if best_x is None:
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        best_x = np.column_stack((xx.ravel(), yy.ravel())).ravel()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
