import time
import math
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_score(pts):
        """
        Computes the score: Delta / n^n
        Delta is product of ordered pair distances after scaling to diameter 2.
        """
        pts = np.asarray(pts)
        # Compute pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Get upper triangle indices
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        if np.min(d) < 1e-12:
            return 0.0
            
        D = np.max(d)
        if D < 1e-12:
            return 0.0
            
        # Scale distances to diameter 2
        scaled_d = d * (2.0 / D)
        
        # Product of unordered distances
        prod_unordered = np.prod(scaled_d)
        
        # Delta is product of ordered distances = (prod_unordered)^2
        Delta = prod_unordered ** 2
        
        return Delta / (n ** n)

    def neg_log_score_from_params_3d(params):
        """
        params: [r1, r2, phi]
        r1, r2 > 0.
        phi in [0, pi/2] due to symmetry.
        """
        if not np.all(np.isfinite(params)):
            return 1e10
        r1, r2, phi = params
        if r1 < 1e-6 or r2 < 1e-6:
            return 1e10
            
        # Generate points
        pts = np.zeros((8, 2))
        # Square 1
        for i in range(4):
            theta = i * np.pi / 2
            pts[2*i] = [r1 * np.cos(theta), r1 * np.sin(theta)]
            # Wait, I need 8 points total.
            # Let's say pts[0..3] are Square 1, pts[4..7] are Square 2.
            # My loop above overwrites. Let's fix indexing.
            
        pts = np.zeros((8, 2))
        for i in range(4):
            theta = i * np.pi / 2
            pts[i] = [r1 * np.cos(theta), r1 * np.sin(theta)]
            
        for i in range(4):
            theta = i * np.pi / 2 + phi
            pts[4+i] = [r2 * np.cos(theta), r2 * np.sin(theta)]
            
        if not np.all(np.isfinite(pts)):
            return 1e10
            
        # Check distinctness
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-8:
            return 1e10
            
        D = np.max(d)
        if D < 1e-12:
            return 1e10
            
        log_sum = np.sum(np.log(d))
        # Maximize log_sum - 14*log(D)
        # Minimize -(log_sum - 14*log(D))
        val = -(log_sum - 14.0 * np.log(D))
        return val

    def get_points_from_3d(params):
        r1, r2, phi = params
        pts = np.zeros((8, 2))
        for i in range(4):
            theta = i * np.pi / 2
            pts[i] = [r1 * np.cos(theta), r1 * np.sin(theta)]
        for i in range(4):
            theta = i * np.pi / 2 + phi
            pts[4+i] = [r2 * np.cos(theta), r2 * np.sin(theta)]
        return pts

    # 1. Optimize the 3D "Two Rotated Squares" family
    best_score = 0.0
    best_pts = None
    
    # Initial guesses for r1, r2, phi
    # Regular octagon: r1=r2=1, phi=pi/4
    # Two squares aligned: r1=r2=1, phi=0
    # Two squares interleaved: r1=r2=1, phi=pi/4
    # Try r1 != r2
    inits_3d = [
        [1.0, 1.0, np.pi/4],
        [1.0, 1.0, 0.0],
        [0.8, 1.0, np.pi/4],
        [1.0, 0.8, np.pi/4],
        [0.5, 1.0, np.pi/4],
        [1.5, 1.0, np.pi/4],
        [1.0, 1.5, np.pi/4],
        [0.7, 1.0, 0.0],
        [1.0, 0.7, 0.0],
        [1.0, 1.0, np.pi/8],
        [1.0, 1.0, 3*np.pi/8],
    ]
    
    # Add some random ones
    np.random.seed(42)
    for _ in range(20):
        r1 = np.random.uniform(0.5, 1.5)
        r2 = np.random.uniform(0.5, 1.5)
        phi = np.random.uniform(0, np.pi/2)
        inits_3d.append([r1, r2, phi])

    for init in inits_3d:
        if time.time() > deadline - 20.0:
            break
        try:
            # Use L-BFGS-B for the 3D problem
            res = minimize(
                neg_log_score_from_params_3d,
                init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < 1e9:
                pts = get_points_from_3d(res.x)
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass

    # Ensure we have a valid fallback
    if best_pts is None:
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
        best_score = compute_score(best_pts)

    # 2. Refine with full 16D optimization
    # We will run L-BFGS-B on the raw coordinates starting from best_pts
    # and also from the regular octagon, to see if we can escape the symmetry.
    
    def objective_16d(coords):
        if not np.all(np.isfinite(coords)):
            return 1e10
        pts = coords.reshape((8, 2))
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_sum = np.sum(np.log(d))
        return -(log_sum - 14.0 * np.log(D))

    def try_full_optim(start_pts):
        nonlocal best_score, best_pts
        start_coords = start_pts.flatten()
        
        # Run L-BFGS-B
        try:
            res = minimize(
                objective_16d,
                start_coords,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < 1e9:
                pts = res.x.reshape((8, 2))
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass

    # Start from the best 3D result
    if time.time() < deadline - 10.0:
        try_full_optim(best_pts)

    # Start from Regular Octagon
    if time.time() < deadline - 10.0:
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
        try_full_optim(reg_pts)

    # Start from a perturbed Regular Octagon
    if time.time() < deadline - 10.0:
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
        perturbed = reg_pts + np.random.randn(8, 2) * 0.1
        try_full_optim(perturbed)

    # Final validation
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])

    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6:
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])

    # Scale to ensure coords are small
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5
    elif max_coord < 1e-6 and max_coord > 0:
        best_pts = best_pts / max_coord * 1.0

    return best_pts.tolist()
