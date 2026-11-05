"""
Optimize 8 points to maximize the normalized product of pairwise distances
(discriminant / diameter^(n(n-1))).

Approach:
- Represent 8 points as roots of a degree-8 polynomial P(z) = sum c_k z^k.
- The discriminant of P is proportional to the product of squared pairwise differences.
- Optimize real coefficients c_0..c_7 (16 real params) to maximize log|discriminant| - n(n-1)*log(max_modulus).
- Use soft-max for smooth diameter constraint.
- Multiple restarts with local refinement.
"""

import numpy as np
from math import cos, sin, pi, log, sqrt
from scipy.optimize import minimize, differential_evolution

def _discriminant_log(coeffs):
    """
    Compute log|discriminant| of a degree-8 polynomial given coefficients [c0, c1, ..., c7].
    The discriminant is prod_{i<j} (r_i - r_j)^2 where r_i are roots.
    We use the resultant-based formula: disc(P) = (-1)^(n(n-1)/2) / lc * Res(P, P').
    
    For numerical stability, we compute via the Sylvester matrix determinant.
    """
    n = 8
    # coeffs: [c0, c1, ..., c7] where P(z) = c0 + c1*z + ... + c7*z^7
    # Make it monic for simplicity: divide by c7 if nonzero
    lc = coeffs[7]
    if abs(lc) < 1e-15:
        return -1e10  # Degenerate polynomial
    
    # Normalize to monic polynomial
    c = np.array(coeffs, dtype=float) / lc
    
    # Derivative coefficients: P'(z) = sum_{k=1}^7 k*c_k z^(k-1)
    # P'(z) = 1*c_1 + 2*c_2*z + ... + 7*c_7*z^6
    cp = np.zeros(7)
    for k in range(1, 8):
        cp[k-1] = k * c[k]
    
    # Resultant of P (degree 8) and P' (degree 7)
    # Build Sylvester matrix: (8+7) x (8+7) = 15 x 15
    # First 7 rows: coefficients of P shifted
    # Next 8 rows: coefficients of P' shifted
    m = 15
    syl = np.zeros((m, m))
    
    # P has coefficients c[0..7], degree 8
    # P' has coefficients cp[0..6], degree 7
    
    # Rows 0..6: P shifted by 0..6
    for i in range(7):
        syl[i, i:i+8] = c
    # Rows 7..14: P' shifted by 0..7
    for i in range(8):
        syl[i+7, i:i+7] = cp
    
    # Determinant of Sylvester matrix gives resultant
    # Use slogdet for numerical stability
    sign, logdet = np.linalg.slogdet(syl)
    
    if sign == 0:
        return -1e10
    
    # disc(P) = (-1)^(n(n-1)/2) * Res(P, P') / lc^n
    # For monic polynomial, lc=1, so disc = (-1)^(28) * Res = Res
    # log|disc| = log|Res| = logdet (since sign is handled)
    
    return logdet

def _max_modulus_from_coeffs(coeffs):
    """
    Estimate max modulus of roots from polynomial coefficients.
    Uses Cauchy bound as a smooth proxy, then refines with actual root finding.
    """
    n = 8
    c = np.array(coeffs, dtype=float)
    lc = c[7]
    if abs(lc) < 1e-15:
        return 1e10
    
    # Normalize
    c_norm = c / lc
    
    # Cauchy bound: max |root| <= 1 + max_{k<7} |c_k / c_7|
    if np.any(np.abs(c_norm[:7]) > 0):
        cauchy_bound = 1.0 + np.max(np.abs(c_norm[:7]))
    else:
        cauchy_bound = 1.0
    
    # Also use the actual root finding for accuracy
    try:
        roots = np.roots(c_norm[::-1])  # np.roots expects highest degree first
        max_mod = np.max(np.abs(roots))
        return max_mod
    except Exception:
        return cauchy_bound

def _objective(x):
    """
    Objective to minimize: -log|discriminant| + n(n-1)*log(soft_max_modulus).
    x is 16-vector: [c0, c1, ..., c7, d0, d1, ..., d7] where c are real parts and d are imaginary parts?
    
    Actually, let's use a different parameterization. Since we want real points in the plane,
    we can represent each point as a complex number z_k = x_k + i*y_k.
    
    Alternative: use the polynomial whose roots are the 8 complex points.
    P(z) = prod_{k=0}^7 (z - z_k) = sum_{j=0}^8 e_j z^j
    
    The coefficients e_j are elementary symmetric functions of the z_k.
    They are complex in general, but if the points are conjugate-symmetric, they're real.
    
    For simplicity, let's parameterize by the 8 points directly but use the discriminant
    computed from the polynomial formed by these points.
    
    Let me reconsider: parameterize by 16 real coordinates (8 points x 2 coords),
    form the polynomial with these as roots, compute its discriminant.
    """
    n = 8
    pts = x.reshape(8, 2)
    z = pts[:, 0] + 1j * pts[:, 1]
    
    # Form polynomial P(w) = prod_{k=0}^7 (w - z_k)
    # Coefficients are elementary symmetric polynomials
    # Use numpy to compute
    try:
        coeffs = np.poly(z)  # Returns [1, e1, e2, ..., e8]
    except Exception:
        return 1e10
    
    # coeffs[0] should be 1 (monic)
    # coeffs[1..8] are the elementary symmetric functions
    
    # Discriminant of monic polynomial = prod_{i<j} (z_i - z_j)^2
    # |disc| = prod_{i<j} |z_i - z_j|^2 = (prod_{i<j} |z_i - z_j|)^2
    # So log|disc| = 2 * sum_{i<j} log|z_i - z_j|
    
    # Compute directly for accuracy
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = abs(z[i] - z[j])
            if d < 1e-15:
                return 1e10
            dists.append(np.log(d))
    
    log_prod_unordered = sum(dists)
    log_disc = 2 * log_prod_unordered
    
    # Diameter: max pairwise distance
    max_dist = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = abs(z[i] - z[j])
            if d > max_dist:
                max_dist = d
    
    if max_dist < 1e-15:
        return 1e10
    
    # Soft max for smoothness
    beta = 20.0
    all_dists = []
    for i in range(n):
        for j in range(i+1, n):
            all_dists.append(abs(z[i] - z[j]))
    all_dists = np.array(all_dists)
    
    # log(soft_max) = (1/beta) * log(sum(exp(beta*d)))
    # = (1/beta) * (beta*max_d + log(sum(exp(beta*(d-max_d)))))
    # = max_d + (1/beta)*log(sum(exp(beta*(d-max_d))))
    max_d = np.max(all_dists)
    log_soft_max = max_d + (1.0/beta) * np.log(np.sum(np.exp(beta*(all_dists - max_d))))
    
    # Objective to maximize: log_disc - n(n-1)*log(diameter)
    # We minimize the negative
    val = log_disc - n * (n-1) * log_soft_max
    
    # Penalty for points being too close
    min_dist = np.min(all_dists)
    if min_dist < 1e-6:
        val -= 1e6 * (1e-6 - min_dist)
    
    # Penalty for large coordinates
    if np.any(np.abs(pts) >= 1e6):
        val -= 1e10
    
    return -val

def construct():
    """
    Returns 8 distinct points [x, y] that maximize the normalized product
    of pairwise distances.
    
    Uses numerical optimization with multiple restarts, leveraging the
    discriminant formulation for efficient gradient computation.
    """
    n = 8
    best_score = -np.inf
    best_points = None
    
    # Reference: regular octagon
    reg_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    
    def _score(pts):
        """Compute the actual score for a set of points."""
        z = pts[:, 0] + 1j * pts[:, 1]
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                d = abs(z[i] - z[j])
                if d < 1e-15:
                    return -np.inf
                dists.append(np.log(d))
        log_prod = sum(dists)
        
        all_d = []
        for i in range(n):
            for j in range(i+1, n):
                all_d.append(abs(z[i] - z[j]))
        diam = max(all_d)
        
        if diam < 1e-15:
            return -np.inf
        
        return log_prod - n*(n-1)*np.log(diam)
    
    ref_score = _score(reg_pts)
    best_score = ref_score
    best_points = reg_pts.copy()
    
    # Try multiple random initializations
    rng = np.random.default_rng(42)
    
    for restart in range(30):
        if restart == 0:
            # Start from regular octagon with small perturbation
            x0 = reg_pts.flatten() + rng.normal(0, 0.01, 16)
        elif restart == 1:
            # Slightly stretched octagon
            angles = 2 * pi * np.arange(8) / 8
            radii = np.ones(8)
            radii[0] = 1.1
            radii[2] = 0.9
            x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
        elif restart == 2:
            # Two circles
            angles1 = 2 * pi * np.arange(4) / 4
            angles2 = 2 * pi * np.arange(4) / 4 + pi/4
            r1, r2 = 0.8, 1.2
            x0 = np.column_stack([
                r1 * np.cos(angles1) * np.concatenate([np.ones(4), np.zeros(4)]),
                r1 * np.sin(angles1) * np.concatenate([np.ones(4), np.zeros(4)]),
                r2 * np.cos(angles2) * np.concatenate([np.zeros(4), np.ones(4)]),
                r2 * np.sin(angles2) * np.concatenate([np.zeros(4), np.ones(4)])
            ]).flatten()
        else:
            # Random initialization on unit disk
            angles = rng.uniform(0, 2 * pi, 8)
            radii = rng.uniform(0.3, 1.5, 8)
            x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
        
        # Use L-BFGS-B for local optimization
        result = minimize(_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if result.fun < -best_score:
            score = -result.fun
            if score > best_score:
                best_score = score
                best_points = result.x.reshape(8, 2)
    
    # Try additional refinements from best point
    for _ in range(10):
        x0 = best_points.flatten()
        result = minimize(_objective, x0, method='L-BFGS-B',
                         options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
        if result.fun < -best_score:
            score = -result.fun
            if score > best_score:
                best_score = score
                best_points = result.x.reshape(8, 2)
    
    # Final check: ensure points are valid
    pts = best_points
    if np.any(np.abs(pts) >= 1e6):
        pts = reg_pts
    
    # Verify distinctness
    z = pts[:, 0] + 1j * pts[:, 1]
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = abs(z[i] - z[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-8:
        pts = reg_pts
    
    return [[float(x), float(y)] for x, y in pts]
