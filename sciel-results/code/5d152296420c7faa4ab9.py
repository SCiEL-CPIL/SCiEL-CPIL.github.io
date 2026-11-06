import numpy as np
from scipy.optimize import minimize, differential_evolution
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n = 8
    
    def compute_score(points):
        """
        Compute Delta/n^n where Delta is product of all ordered pair distances
        after scaling to diameter 2.
        """
        pts = np.asarray(points, dtype=float)
        if len(pts) != n:
            return -np.inf
        
        # Check for distinct points
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-10:
                    return -np.inf
        
        # Compute diameter
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                max_d = max(max_d, d)
        
        if max_d < 1e-15:
            return -np.inf
        
        # Scale to diameter 2
        scale = 2.0 / max_d
        pts_scaled = pts * scale
        
        # Compute log of product of all ordered pair distances
        # Ordered pairs means (i,j) and (j,i) both count, so it's 2 * sum over i<j of log(d_ij)
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return -np.inf
                log_prod += math.log(d)
        
        log_prod_ordered = 2.0 * log_prod
        
        # Delta = exp(log_prod_ordered), score = Delta / n^n
        # log(score) = log_prod_ordered - n*log(n)
        log_score = log_prod_ordered - n * math.log(n)
        
        return math.exp(log_score)

    def params_to_points(params):
        """
        Convert 8 parameters (4 radii, 4 angles) to 8 points.
        Enforces central symmetry (antipodal pairs).
        """
        radii = params[:4]
        thetas = params[4:]
        
        # Ensure radii are positive
        radii = np.abs(radii) + 1e-6
        
        pts = np.zeros((n, 2))
        for i in range(4):
            r = radii[i]
            theta = thetas[i]
            pts[i] = [r * math.cos(theta), r * math.sin(theta)]
            pts[i + 4] = [-r * math.cos(theta), -r * math.sin(theta)]
        
        # Sort points by angle to ensure consistent ordering (optional but helpful for debugging)
        # Actually, sorting might break the pairing if we just sort all 8.
        # But the score is permutation invariant.
        # Let's just return the points as constructed.
        return pts

    def negative_log_score(params):
        """Objective for minimization: -log(score)"""
        if time.time() > deadline:
            return 1e10
            
        pts = params_to_points(params)
        
        # Check for distinct points
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                    return 1e10
        
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                max_d = max(max_d, d)
        
        if max_d < 1e-15:
            return 1e10
        
        scale = 2.0 / max_d
        pts_scaled = pts * scale
        
        log_prod = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts_scaled[i] - pts_scaled[j])
                if d < 1e-15:
                    return 1e10
                log_prod += math.log(d)
        
        log_prod_ordered = 2.0 * log_prod
        log_score = log_prod_ordered - n * math.log(n)
        
        return -log_score

    # Regular octagon as baseline
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    reg_score = compute_score(reg_pts)
    
    best_pts = reg_pts.copy()
    best_score = reg_score
    
    # Symmetric parameterization: 4 pairs of antipodal points
    # Parameters: r1, r2, r3, r4, t1, t2, t3, t4
    # Regular octagon in this parameterization:
    # r = [1, 1, 1, 1]
    # t = [0, pi/4, pi/2, 3pi/4] -> wait, angles must be in [0, pi/2] for unique representation?
    # If we allow angles in [0, 2pi), we have redundancy.
    # Let's constrain angles to [0, pi/2] and rely on the fact that 
    # the set of 4 points in the first quadrant, mirrored, covers the plane.
    # However, the regular octagon has points at 0, 45, 90, 135...
    # In the first quadrant, we have 0 and 45. 90 is on the axis, 135 is in 2nd quadrant.
    # So for the regular octagon, the 4 points in the "first quadrant sector" (0 to 90) are at 0, 45, 90?
    # No, the antipodal symmetry means we define 4 points $P_1, P_2, P_3, P_4$ and their negatives.
    # For the regular octagon, we can pick $P_1$ at angle 0, $P_2$ at 45, $P_3$ at 90, $P_4$ at 135?
    # If we pick $P_3$ at 90, $-P_3$ is at 270.
    # If we pick $P_4$ at 135, $-P_4$ is at 315.
    # This covers all 8.
    # So angles can be in [0, 2pi).
    
    x0_sym = np.array([1.0, 1.0, 1.0, 1.0,
                        0.0, np.pi/4, np.pi/2, 3.0*np.pi/4])
    
    # Try differential evolution on the 8 symmetric parameters
    # Bounds: radii (0.1, 5.0), angles (-pi, pi)
    bounds_de = [(0.1, 5.0)] * 4 + [(-np.pi, np.pi)] * 4
    
    best_de_score = -np.inf
    best_de_params = x0_sym.copy()
    
    if time.time() < deadline - 10:
        try:
            # Use DE to find a good starting point
            res_de = differential_evolution(
                negative_log_score, bounds_de,
                maxiter=50, popsize=20, tol=1e-4,
                seed=42, polish=False, updating='immediate'
            )
            
            if np.all(np.isfinite(res_de.x)):
                best_de_params = res_de.x
                s_de = compute_score(params_to_points(res_de.x))
                if s_de > best_score:
                    best_score = s_de
                    best_pts = params_to_points(res_de.x).copy()
        except Exception:
            pass

    # Polish with L-BFGS-B from multiple starts
    # Start 1: Regular Octagon
    # Start 2: DE result
    # Start 3: Perturbed Regular Octagon
    
    starts = [
        x0_sym,
        best_de_params
    ]
    
    # Add some random perturbations of the regular octagon
    np.random.seed(42)
    for _ in range(5):
        noise_r = np.random.normal(0, 0.2, size=4)
        noise_t = np.random.normal(0, 0.2, size=4)
        pert = x0_sym.copy()
        pert[:4] += noise_r
        pert[4:] += noise_t
        starts.append(pert)
    
    for x0 in starts:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(negative_log_score, x0, method='L-BFGS-B',
                          bounds=bounds_de,
                          options={'maxiter': 500, 'ftol': 1e-12})
            
            if np.all(np.isfinite(res.x)):
                pts = params_to_points(res.x)
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()
        except Exception:
            pass
    
    # Final validation
    best_pts = np.asarray(best_pts, dtype=float)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts
        best_score = reg_score
    
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5
    
    return best_pts.tolist()
