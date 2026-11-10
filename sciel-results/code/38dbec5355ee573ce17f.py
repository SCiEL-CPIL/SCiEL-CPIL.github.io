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

    def exact_score(x):
        pts = x.reshape(n, d)
        return exact_score_from_pts(pts)

    def exact_objective(x):
        return -exact_score(x)

    # --- Hexagonal lattice parameterization ---
    # 16 points arranged in a 4x4 block on a hexagonal lattice
    # Lattice vectors: a1 = (s, 0), a2 = (s*aspect, s*sqrt(3)*aspect/2)
    # Points at (i*a1 + j*a2) for i,j in 0..3, plus small perturbations
    # Parameters: [s, aspect, dx0,dy0, dx1,dy1, ..., dx15,dy15]
    # That's 2 + 32 = 34 parameters, but we can reduce by noting symmetry
    
    def hex_lattice_points(params):
        """
        params: [s, aspect, dx0, dy0, dx1, dy1, ..., dx15, dy15]
        s: lattice spacing
        aspect: aspect ratio of lattice (1.0 = perfect hexagonal)
        dx_i, dy_i: small perturbations for each point
        Returns (16, 2) array
        """
        s = params[0]
        aspect = params[1]
        
        # Lattice vectors
        a1 = np.array([s, 0.0])
        a2 = np.array([s * aspect, s * np.sqrt(3) * aspect / 2.0])
        
        # Base 4x4 grid positions
        base_pts = np.zeros((n, 2))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                base_pts[idx] = i * a1 + j * a2
        
        # Add perturbations
        pts = base_pts.copy()
        for k in range(n):
            pts[k, 0] += params[2 + 2*k]
            pts[k, 1] += params[2 + 2*k + 1]
        
        return pts

    def hex_objective(params):
        pts = hex_lattice_points(params)
        return -exact_score_from_pts(pts)

    def hex_soft_objective(params):
        """Smooth version for local optimization."""
        pts = hex_lattice_points(params)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        dists_norm = dists / d_max
        # Soft minimum with small temperature
        t = 0.05
        d_min_norm = -np.log(np.sum(np.exp(-dists_norm / t)) / len(dists_norm)) * t
        return -d_min_norm

    best_x = None
    best_score = 0.0

    # Phase 1: Optimize hexagonal lattice parameters
    # Parameters: [s, aspect, 32 perturbations]
    # s in [0.5, 2.0], aspect in [0.5, 1.5], perturbations in [-0.3, 0.3]
    hex_bounds = [(0.5, 2.0), (0.5, 1.5)] + [(-0.3, 0.3)] * (2 * n)
    
    try:
        if time.time() < deadline - 30:
            res_hex = differential_evolution(
                hex_objective,
                hex_bounds,
                maxiter=300,
                popsize=20,
                seed=42,
                tol=1e-8,
                init='latinhypercube',
                polish=False,
                updating='deferred'
            )
            
            if time.time() < deadline - 15:
                hex_params = res_hex.x
                hex_pts = hex_lattice_points(hex_params)
                hex_score = exact_score_from_pts(hex_pts)
                
                if hex_score > best_score:
                    best_score = hex_score
                    best_x = hex_pts.ravel()
    except Exception:
        pass

    # Phase 2: Local refinement on full coordinates
    if best_x is not None and time.time() < deadline - 10:
        try:
            # Use a smooth objective for local refinement
            def smooth_full_objective(x):
                pts = x.reshape(n, d)
                dists = pdist(pts)
                d_max = np.max(dists)
                if d_max < 1e-12:
                    return 0.0
                dists_norm = dists / d_max
                t = 0.05
                d_min_norm = -np.log(np.sum(np.exp(-dists_norm / t)) / len(dists_norm)) * t
                return -d_min_norm
            
            res_local = minimize(
                smooth_full_objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            local_score = exact_score(res_local.x)
            if local_score > best_score:
                best_score = local_score
                best_x = res_local.x
        except Exception:
            pass

    # Phase 3: Try a few random restarts with local optimization if needed
    if best_score < 0.5 and time.time() < deadline - 10:
        np.random.seed(42)
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            x_start = np.random.uniform(-1, 1, (n * d))
            try:
                def smooth_full_objective(x):
                    pts = x.reshape(n, d)
                    dists = pdist(pts)
                    d_max = np.max(dists)
                    if d_max < 1e-12:
                        return 0.0
                    dists_norm = dists / d_max
                    t = 0.05
                    d_min_norm = -np.log(np.sum(np.exp(-dists_norm / t)) / len(dists_norm)) * t
                    return -d_min_norm
                
                res = minimize(smooth_full_objective, x_start, method='L-BFGS-B', options={'maxiter': 100})
                s = exact_score(res.x)
                if s > best_score:
                    best_score = s
                    best_x = res.x
            except Exception:
                continue

    # Fallback: 4x4 grid
    if best_x is None:
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        best_x = np.column_stack((xx.ravel(), yy.ravel())).ravel()
        dmax_init = np.max(pdist(best_x.reshape(n, d)))
        if dmax_init > 1e-12:
            best_x = (best_x / dmax_init).ravel()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
