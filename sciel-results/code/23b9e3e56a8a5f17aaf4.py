"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    t_start = time.time()
    deadline = 135.0  # safety margin before 150s kill

    n = 8

    # Score: product of distances over ordered pairs (i != j) scaled to diameter 2.
    # Score = (prod_{i!=j} d_ij) / 8^8
    # Maximize log(score) = sum_{i!=j} log(d_ij) - 8*log(8)
    # Since scaling to diameter 2 is done at the end, we can maximize the product
    # of distances for any configuration, then scale.
    # Actually, the score is scale invariant?
    # Let D be the diameter. We scale by 2/D.
    # New distances d' = d * (2/D).
    # Product of new distances = (prod d) * (2/D)^{56}
    # Score = (prod d) * (2/D)^{56} / 8^8
    # To maximize score, we maximize (prod d) / D^{56}.
    # log score = sum log(d) - 56 log(D) - 8 log(8)
    
    def score_from_points(pts):
        # pts: (8, 2)
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, 1.0)
        
        # Diameter
        D = np.max(d)
        if D < 1e-10:
            return -np.inf, D, d
            
        # Product of off-diagonal distances
        # sum log(d)
        log_prod = np.sum(np.log(d))
        
        # log score = log_prod - 56 * log(D) - 8 * log(8)
        # We want to maximize this.
        # For optimization, we can just maximize log_prod - 56*log(D)
        val = log_prod - 56.0 * np.log(D)
        return val, D, d

    # Two-square parameterization
    # Square 1: radius r1, angle a1
    # Square 2: radius r2, angle a2
    # We can fix a1 = 0 due to rotational symmetry of the whole set?
    # The score is invariant under global rotation.
    # So fix a1 = 0.
    # Variables: r1, r2, delta_angle = a2 - a1
    # r1, r2 > 0. delta_angle in [0, pi/2) (symmetry of square)
    
    def params_to_pts(params):
        r1, r2, da = params
        # Ensure positive radii
        r1 = max(r1, 1e-6)
        r2 = max(r2, 1e-6)
        
        # Square 1 points
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        # Square 2 points
        angles2 = angles1 + da
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        pts = np.vstack([pts1, pts2])
        return pts

    def neg_score(params):
        pts = params_to_pts(params)
        val, D, d = score_from_points(pts)
        if not np.isfinite(val):
            return 1e10
        return -val

    # Initial guesses
    # Regular octagon: r1 = r2 = R, da = pi/8
    # R such that diameter is 2? No, R is just a parameter.
    # In regular octagon, all points on circle.
    # Distances:
    # Between adjacent: 2R sin(pi/8)
    # Diameter: 2R
    # Let's just use R=1, da=pi/8
    best_params = np.array([1.0, 1.0, np.pi/8])
    best_val = -neg_score(best_params)

    # Local optimization around regular octagon
    try:
        res = minimize(neg_score, best_params, method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 2000})
        if -res.fun > best_val:
            best_val = -res.fun
            best_params = res.x
    except Exception:
        pass

    # Try other initializations
    # r1 != r2
    inits = [
        [1.0, 0.5, 0.0],
        [1.0, 0.5, np.pi/4],
        [1.0, 1.5, 0.0],
        [1.0, 1.5, np.pi/4],
        [1.0, 0.7, np.pi/8],
        [1.0, 1.2, np.pi/8],
        [0.5, 1.0, np.pi/8],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, np.pi/4],
    ]
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(neg_score, np.array(init), method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 2000})
            if -res.fun > best_val:
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass

    # Refine with BFGS if possible (needs gradient? No, Nelder-Mead is fine, but BFGS with numerical grad might be faster)
    # Let's try a final polish with COBYLA or similar robust method
    try:
        res = minimize(neg_score, best_params, method='COBYLA',
                       options={'rhobeg': 1e-4, 'maxiter': 1000, 'maxfev': 1000})
        if -res.fun > best_val:
            best_val = -res.fun
            best_params = res.x
    except Exception:
        pass

    # Generate points from best_params
    pts = params_to_pts(best_params)

    # Scale to diameter 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1.0)
    D = np.max(d)
    
    if D < 1e-10:
        # Fallback
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        
    pts_scaled = pts * (2.0 / D)

    # Check validity
    if not np.all(np.isfinite(pts_scaled)) or np.max(np.abs(pts_scaled)) >= 1e6:
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    return pts_scaled.tolist()
# EVOLVE-BLOCK-END
