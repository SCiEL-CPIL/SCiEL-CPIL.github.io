# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses hexagonal lattice parameterization with discrete removal patterns.
    
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

    # Hexagonal lattice: 19 sites in a 4x4 hex grid, remove 3 to get 16
    # Generate all 19 hex lattice points
    def get_hex_19():
        """Generate 19 hex lattice points (4x4 hex grid with corners)."""
        # Hex lattice basis
        a1 = np.array([1.0, 0.0])
        a2 = np.array([0.5, np.sqrt(3)/2])
        
        # Generate points in hex coordinates (i, j) where 0 <= i, j <= 3
        # and |i - j| <= 1 for hex shape, or use a different pattern
        # Use a 4x4 square grid in hex basis
        pts = []
        for i in range(4):
            for j in range(4):
                p = i * a1 + j * a2
                pts.append(p)
        
        # This gives 16 points. Need 19. Let's use a different pattern.
        # Use hexagonal packing: center + 6 + 12 = 19
        # Or use 5x4 = 20, remove 1
        # Better: use a 4x4 hex grid which naturally has 16 points
        # For 19 points, use: row 0: 4, row 1: 5, row 2: 4, row 3: 5, row 4: 1? No.
        
        # Let's use a proper hexagonal cluster of 19:
        # Center point
        # Ring 1: 6 points
        # Ring 2: 12 points
        # Total: 1 + 6 + 12 = 19
        
        pts = [np.array([0.0, 0.0])]
        
        # Ring 1: 6 points at distance 1
        for k in range(6):
            angle = k * np.pi / 3
            pts.append(np.array([np.cos(angle), np.sin(angle)]))
        
        # Ring 2: 12 points at distance 2
        for k in range(12):
            angle = k * np.pi / 6
            # Hexagonal distances: alternate between 2 and sqrt(3)
            if k % 2 == 0:
                r = 2.0
            else:
                r = np.sqrt(3)
            pts.append(r * np.array([np.cos(angle), np.sin(angle)]))
        
        return np.array(pts)

    # Generate all combinations of 3 sites to remove from 19
    def get_removal_patterns():
        """Return list of tuples of 3 indices to remove from 19 points."""
        return list(combinations(range(19), 3))

    # Hex lattice parameterization:
    # params = [aspect, angle, dx0, dy0, ..., dx15, dy15]
    # aspect: aspect ratio of hex lattice (1.0 = perfect)
    # angle: rotation angle
    # dx_i, dy_i: small perturbations for each of the 16 remaining points
    
    def hex_lattice_points(params, removal_idx):
        """
        params: [aspect, angle, dx0, dy0, ..., dx15, dy15] = 2 + 32 = 34 params
        removal_idx: tuple of 3 indices to remove from the 19-point hex lattice
        Returns: (16, 2) array of points
        """
        aspect = params[0]
        angle = params[1]
        perturbations = params[2:]
        
        # Generate 19-point hex lattice
        pts19 = get_hex_19()
        
        # Apply aspect ratio (scale y-axis)
        pts19[:, 1] *= aspect
        
        # Apply rotation
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts19 = pts19 @ rot.T
        
        # Remove the specified indices
        keep_idx = [i for i in range(19) if i not in removal_idx]
        pts16 = pts19[keep_idx]
        
        # Apply small perturbations
        for i in range(16):
            pts16[i, 0] += perturbations[2*i]
            pts16[i, 1] += perturbations[2*i + 1]
        
        return pts16

    def hex_objective(params, removal_idx):
        """Objective for hex lattice parameterization."""
        pts = hex_lattice_points(params, removal_idx)
        return -exact_score_from_pts(pts)

    def hex_soft_objective(params, removal_idx, t=0.05):
        """Soft objective for hex lattice (differentiable approximation)."""
        pts = hex_lattice_points(params, removal_idx)
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

    # Phase 1: Try all removal patterns with initial hex lattice
    removal_patterns = get_removal_patterns()
    
    for rem_idx in removal_patterns:
        if time.time() > deadline - 120:
            break
        
        # Initial params: aspect=1.0, angle=0, no perturbations
        init_params = np.zeros(34)
        init_params[0] = 1.0  # aspect
        init_params[1] = 0.0  # angle
        
        try:
            # Local optimization with Nelder-Mead
            res = minimize(
                lambda p: hex_objective(p, rem_idx),
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            pts = hex_lattice_points(res.x, rem_idx)
            s = exact_score_from_pts(pts)
            
            if s > best_score:
                best_score = s
                best_x = pts.ravel()
                
        except Exception:
            continue

    # Phase 2: Global search with Differential Evolution on free coordinates
    if time.time() < deadline - 60:
        bounds = [(-2.0, 2.0)] * (n * d)
        
        try:
            res_de = differential_evolution(
                exact_objective,
                bounds,
                maxiter=200,
                popsize=30,
                seed=42,
                tol=1e-10,
                init='latinhypercube',
                polish=False,
                updating='immediate'
            )
            
            de_score = exact_score_from_pts(res_de.x.reshape(n, d))
            if de_score > best_score:
                best_score = de_score
                best_x = res_de.x.copy()
        except Exception:
            pass

    # Phase 3: Nelder-Mead local refinement on exact objective
    if best_x is not None and time.time() < deadline - 30:
        try:
            res_nm = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            nm_score = exact_score_from_pts(res_nm.x.reshape(n, d))
            if nm_score > best_score:
                best_score = nm_score
                best_x = res_nm.x.copy()
        except Exception:
            pass

    # Phase 4: Multi-start with diverse initializations + Nelder-Mead
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
                if s > best_score:
                    best_score = s
                    best_x = res.x.copy()
            except Exception:
                continue

    # Phase 5: Final polish with L-BFGS-B on smooth objective
    if best_x is not None and time.time() < deadline - 5:
        def smooth_objective(x, t=0.03):
            """Differentiable approximation for local optimization."""
            pts = x.reshape(n, d)
            dists = pdist(pts)
            d_max = np.max(dists)
            if d_max < 1e-12:
                return 0.0
            dists_norm = dists / d_max
            d_min_norm = -np.log(np.sum(np.exp(-dists_norm / t)) / len(dists_norm)) * t
            return -d_min_norm
        
        try:
            res_lbfgs = minimize(
                smooth_objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            s = exact_score_from_pts(res_lbfgs.x.reshape(n, d))
            if s > best_score:
                best_score = s
                best_x = res_lbfgs.x.copy()
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
