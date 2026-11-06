import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
import cmath

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def compute_score(pts):
        """Compute the score for a set of points."""
        pts = np.asarray(pts, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        # Scale to diameter 2
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2  # Product over ordered pairs
        return Delta / (n ** n)

    def objective_general(x):
        """
        Objective for general optimization.
        Minimizes -sum(log(d_ij)) + penalty for diameter > 2.
        """
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        if np.min(d) < 1e-10:
            return 1e10
            
        D = np.max(d)
        if D < 1e-10:
            return 1e10
            
        # Maximize sum(log(d)) -> Minimize -sum(log(d))
        log_sum = np.sum(np.log(d))
        
        # Penalty for exceeding diameter 2
        penalty = 0.0
        if D > 2.0:
            penalty = 100.0 * (D - 2.0)**2
            
        return -log_sum + penalty

    best_score = 0.0
    best_pts = None

    def try_candidate(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            return 0.0
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # 1. Regular Octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    try_candidate(reg_pts)

    # 2. Two Concentric Squares Strategy (from Inspiration)
    # This structure often outperforms regular polygons for n=8.
    # We parameterize as two squares with radii r1, r2 and relative rotation theta.
    
    def objective_two_squares(params):
        r1, r2, theta = params
        if r1 < 0.001 or r2 < 0.001:
            return 1e10
            
        # Square 1: angles 0, pi/2, pi, 3pi/2
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        
        # Square 2: angles theta, theta+pi/2, ...
        angles2 = angles1 + theta
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        
        pts = np.vstack([pts1, pts2])
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
            
        # Maximize log product of scaled distances
        # log(Delta/n^n) = 2 * (sum(log(d_scaled)) - n*log(n))
        # d_scaled = d * (2/D)
        # sum(log(d_scaled)) = sum(log(d)) + 28*log(2) - 28*log(D)
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    # Initial guesses for r1, r2, theta - expanded around known good region
    guesses = [
        [1.0, 0.0, np.pi/4], # Degenerate (one square shrunk)
        [0.5, 1.0, np.pi/4],
        [0.7, 1.0, np.pi/4],
        [0.8, 0.8, np.pi/4], # Two identical squares rotated
        [1.0, 1.0, 0.0],     # Overlapping
        [0.6, 1.2, np.pi/8],
        [0.9, 0.9, np.pi/4],
        [1.0, 0.5, np.pi/4],
        [1.0, 0.7, np.pi/4],
        [0.8, 1.0, np.pi/8],
        # Additional targeted guesses near promising regions
        [0.75, 1.05, np.pi/4],
        [0.85, 0.95, np.pi/4],
        [0.95, 0.85, np.pi/4],
        [0.70, 1.10, np.pi/5],
        [0.80, 1.00, np.pi/6],
        [1.00, 0.80, np.pi/6],
        [0.65, 1.15, np.pi/7],
        [0.90, 0.90, np.pi/4],
        [0.75, 1.00, np.pi/3],
        [1.00, 0.75, np.pi/3],
    ]
    
    for guess in guesses:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(
                objective_two_squares,
                guess,
                method='L-BFGS-B',
                bounds=[(0.01, 1.5), (0.01, 1.5), (0.0, np.pi/2)],
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res.success or res.fun < 1e9:
                r1, r2, theta = res.x
                angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                angles2 = angles1 + theta
                pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
                pts = np.vstack([pts1, pts2])
                try_candidate(pts)
        except Exception:
            pass

    # 3. General Optimization with Penalty
    # Start from several random and structured initializations
    
    np.random.seed(42)
    inits = []
    
    # Random in disk
    for _ in range(10):
        r = np.random.rand(n) * 1.0
        theta = np.random.rand(n) * 2 * np.pi
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)
        
    # Perturbed Regular Octagon
    for _ in range(5):
        angles = angles_reg + 0.1 * np.random.randn(n)
        radii = 1.0 + 0.1 * np.random.randn(n)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)

    # Two squares random init
    for _ in range(5):
        r1 = 0.5 + np.random.rand() * 0.5
        r2 = 0.5 + np.random.rand() * 0.5
        theta = np.random.rand() * np.pi/2
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        angles2 = angles1 + theta
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        inits.append(np.vstack([pts1, pts2]))

    # Add best_pts from two-squares as an initial guess for general optimization
    if best_pts is not None:
        inits.append(best_pts.copy())
        # Perturbed versions
        for scale in [0.05, 0.1, 0.2]:
            inits.append(best_pts + scale * np.random.randn(n, 2))

    bounds_general = [(-1.5, 1.5)] * (2 * n)

    for init in inits:
        if time.time() > deadline - 5.0:
            break
        try:
            x0 = init.flatten()
            res = minimize(
                objective_general,
                x0,
                method='L-BFGS-B',
                bounds=bounds_general,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                try_candidate(refined_pts)
        except Exception:
            pass

    # 4. Complex-number / polynomial representation
    # Represent 8 points as roots of a degree-8 real polynomial.
    # With conjugate symmetry, we have 4 complex conjugate pairs.
    # Each pair is determined by its center (a+bi) and half-difference (c+di).
    # So 8 real parameters total (4 pairs × 2 real + 2 imag each... wait).
    # Actually: 4 pairs, each pair = (a, b, c, d) where roots are (a+c) + i(b+d) and (a-c) + i(b-d)
    # That's 4 parameters per pair × 4 pairs = 16 params. Not reduced.
    # 
    # Alternative: parameterize by 4 complex numbers z1..z4 (the "centers" of conjugate pairs),
    # and 4 complex numbers w1..w4 (the "half-differences").
    # But if we fix the polynomial to have real coefficients, the roots come in conjugate pairs.
    # So we only need to specify 4 roots (one from each conjugate pair) = 8 real params.
    # The other 4 roots are their conjugates.
    #
    # Let's parameterize: 4 complex numbers (8 real params) define 4 roots.
    # The other 4 roots are conjugates. Total 8 points.
    
    def poly_objective(params):
        """
        params: 8 real numbers representing 4 complex numbers z1..z4
        Each zi = params[2*i] + 1j*params[2*i+1]
        The 8 points are: z1, conj(z1), z2, conj(z2), z3, conj(z3), z4, conj(z4)
        """
        if not np.all(np.isfinite(params)):
            return 1e10
        
        # Construct 8 points from 4 complex numbers and their conjugates
        complex_roots = []
        for i in range(4):
            re = params[2*i]
            im = params[2*i+1]
            z = complex(re, im)
            complex_roots.append(z)
            complex_roots.append(z.conjugate())
        
        pts = np.array([[z.real, z.imag] for z in complex_roots])
        
        if not np.all(np.isfinite(pts)):
            return 1e10
        
        # Check for duplicate points
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        
        # Maximize log product of scaled distances
        log_sum = np.sum(np.log(d))
        
        # Scale to diameter 2
        # log(Delta/n^n) = 2 * (sum(log(d_scaled)) - n*log(n))
        # d_scaled = d * (2/D)
        # sum(log(d_scaled)) = sum(log(d)) + 28*log(2) - 28*log(D)
        log_score = 2.0 * (log_sum + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        
        # Penalty for large coordinates (to keep points bounded)
        coord_penalty = 0.0
        if np.max(np.abs(pts)) > 5.0:
            coord_penalty = 10.0 * (np.max(np.abs(pts)) - 5.0)**2
        
        return -log_score + coord_penalty

    # Initial guesses for the 4 complex numbers
    # Start with regular octagon configuration
    # Regular octagon: 8 points on unit circle
    # Group into 4 conjugate pairs:
    # Pair 1: angle 0 and pi (real axis) -> z = 1, conj = 1 (same!) - degenerate
    # Better: use angles that aren't symmetric about real axis
    # Angles: 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4
    # Conjugate pairs: (0, 0), (pi/4, 7pi/4), (pi/2, 3pi/2), (3pi/4, 5pi/4)
    # So z1 = e^{i*0} = 1, z2 = e^{i*pi/4}, z3 = e^{i*pi/2} = i, z4 = e^{i*3pi/4}
    
    poly_inits = [
        # Regular octagon
        [1.0, 0.0, np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 1.0, np.cos(3*np.pi/4), np.sin(3*np.pi/4)],
        # Perturbed
        [1.0, 0.1, np.cos(np.pi/4)+0.1, np.sin(np.pi/4)-0.1, 0.1, 1.0, np.cos(3*np.pi/4)-0.1, np.sin(3*np.pi/4)+0.1],
        # Two concentric squares style
        [1.0, 0.0, 0.7, 0.7, 0.0, 0.7, -0.7, 0.7],
        [1.0, 0.0, 0.8, 0.6, 0.0, 0.8, -0.6, 0.8],
        # Random
    ]
    
    np.random.seed(123)
    for _ in range(5):
        re_vals = np.random.uniform(-1.5, 1.5, 4)
        im_vals = np.random.uniform(-1.5, 1.5, 4)
        params = np.concatenate([re_vals, im_vals]).tolist()
        # Interleave: [re1, im1, re2, im2, ...]
        params = [re_vals[0], im_vals[0], re_vals[1], im_vals[1], re_vals[2], im_vals[2], re_vals[3], im_vals[3]]
        poly_inits.append(params)
    
    bounds_poly = [(-2.0, 2.0)] * 8
    
    for init in poly_inits:
        if time.time() > deadline - 10.0:
            break
        try:
            res = minimize(
                poly_objective,
                init,
                method='L-BFGS-B',
                bounds=bounds_poly,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if np.all(np.isfinite(res.x)):
                # Extract points
                complex_roots = []
                for i in range(4):
                    re = res.x[2*i]
                    im = res.x[2*i+1]
                    z = complex(re, im)
                    complex_roots.append(z)
                    complex_roots.append(z.conjugate())
                pts = np.array([[z.real, z.imag] for z in complex_roots])
                try_candidate(pts)
        except Exception:
            pass
    
    # Also try differential evolution in the 8D polynomial space
    if time.time() < deadline - 15.0:
        try:
            res_de = differential_evolution(
                poly_objective,
                bounds_poly,
                maxiter=200,
                popsize=30,
                tol=1e-10,
                seed=42,
                polish=True
            )
            if res_de.success or res_de.fun < 1e9:
                complex_roots = []
                for i in range(4):
                    re = res_de.x[2*i]
                    im = res_de.x[2*i+1]
                    z = complex(re, im)
                    complex_roots.append(z)
                    complex_roots.append(z.conjugate())
                pts = np.array([[z.real, z.imag] for z in complex_roots])
                try_candidate(pts)
        except Exception:
            pass
    
    # Additional: Try a grid search in the two-squares family with fine resolution
    if time.time() < deadline - 10.0:
        try:
            r1_vals = np.linspace(0.5, 1.2, 15)
            r2_vals = np.linspace(0.5, 1.2, 15)
            theta_vals = np.linspace(0.1, np.pi/2 - 0.1, 10)
            
            for r1 in r1_vals:
                for r2 in r2_vals:
                    for theta in theta_vals:
                        if time.time() > deadline - 5.0:
                            break
                        guess = [r1, r2, theta]
                        try:
                            res = minimize(
                                objective_two_squares,
                                guess,
                                method='L-BFGS-B',
                                bounds=[(0.01, 1.5), (0.01, 1.5), (0.0, np.pi/2)],
                                options={'maxiter': 200, 'ftol': 1e-14}
                            )
                            if res.success or res.fun < 1e9:
                                r1_opt, r2_opt, theta_opt = res.x
                                angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                                angles2 = angles1 + theta_opt
                                pts1 = np.column_stack([r1_opt * np.cos(angles1), r1_opt * np.sin(angles1)])
                                pts2 = np.column_stack([r2_opt * np.cos(angles2), r2_opt * np.sin(angles2)])
                                pts = np.vstack([pts1, pts2])
                                try_candidate(pts)
                        except Exception:
                            pass
        except Exception:
            pass

    # 5. Final Refinement of the Best Candidate Found (multiple passes)
    if best_pts is not None:
        # Pass 1: L-BFGS-B with tight tolerance
        try:
            res = minimize(
                objective_general,
                best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds_general,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                try_candidate(refined_pts)
        except Exception:
            pass
        
        # Pass 2: Try from slightly perturbed versions of best
        if best_pts is not None and time.time() < deadline - 10.0:
            for perturb_scale in [0.01, 0.005, 0.001]:
                if time.time() > deadline - 5.0:
                    break
                try:
                    perturbed = best_pts + perturb_scale * np.random.randn(n, 2)
                    res = minimize(
                        objective_general,
                        perturbed.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds_general,
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
                    )
                    if np.all(np.isfinite(res.x)):
                        refined_pts = res.x.reshape(n, 2)
                        try_candidate(refined_pts)
                except Exception:
                    pass
        
        # Pass 3: Nelder-Mead polish for final refinement
        if best_pts is not None and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    objective_general,
                    best_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'fatol': 1e-14, 'xatol': 1e-14, 'adaptive': True}
                )
                if np.all(np.isfinite(res.x)):
                    refined_pts = res.x.reshape(n, 2)
                    try_candidate(refined_pts)
            except Exception:
                pass

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Final validation
    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Scale to ensure coords are small (optional, but good for stability)
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
