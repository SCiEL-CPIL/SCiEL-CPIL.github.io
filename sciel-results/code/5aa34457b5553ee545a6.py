"""Optimized program combining analytic bound computation with global search on the sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import math

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    """
    # c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330
    
    # P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    a0 = c0 + 12*c4 + 168*c8 + 1330*c12
    # a0 should be 0 by construction
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    # We need I_k = \int x^k P(x)^2 e^{-x^2} dx
    # P(x)^2 = (a12 x^12 + ... + a0)^2
    # This results in a polynomial of degree 24.
    # Let Q(x) = P(x)^2 = \sum_{j=0}^{12} b_{2j} x^{2j}
    # We need integrals \int x^{2j} e^{-x^2} dx = 0.5 * Gamma(j+0.5)
    
    # Calculate coefficients b_{2j} of P(x)^2
    # P(x) = \sum_{i=0}^6 p_i x^{2i}
    # p = [a0, a2, a4, a6, a8, a10, a12]
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Convolution to get coefficients of P(x)^2
    # q[k] = sum_{i+j=k} p[i]p[j] for x^{2k}
    q = np.convolve(p, p)
    # q has length 13, indices 0 to 12 corresponding to x^0 to x^24
    
    # Integrate x^{2k} e^{-x^2}
    # \int_{-inf}^{inf} x^{2k} e^{-x^2} dx = \sqrt{\pi} (2k-1)!! = \sqrt{\pi} 2^k k!
    # Let's compute these factors
    factors = np.zeros(13)
    for k in range(13):
        # (2k-1)!! = 2^k * k!
        factors[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I = q * factors
    # I[k] = \int x^{2k} P(x)^2 e^{-x^2} dx
    
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10 # Invalid
    
    # Variance <x^2> = I[1] / I[0]
    x2 = I[1] / I[0]
    
    # Calculate g''(x)
    # g = P e^{-x^2/2}
    # g' = (P' - xP) e^{-x^2/2}
    # g'' = (P'' - 2xP' + (x^2 - 1)P) e^{-x^2/2}
    
    # Let R(x) = P''(x) - 2xP'(x) + (x^2 - 1)P(x)
    # We need \int R(x)^2 e^{-x^2} dx.
    
    # P(x) = \sum p_i x^{2i}
    # P'(x) = \sum 2i p_i x^{2i-1}
    # P''(x) = \sum 2i(2i-1) p_i x^{2i-2}
    
    # R(x) is an even polynomial.
    # Coefficient r_j of x^{2j} in R(x):
    # From P'': term is 2(j+1)(2j+1) p_{j+1} (for j=0..5, 0 for j=6)
    # From -2xP': term is -4j p_j (for j=1..6, 0 for j=0)
    # From x^2 P: term is p_{j-1} (for j=1..7, 0 for j=0)
    # From -P: term is -p_j (for j=0..6, 0 for j=7)
    
    r = np.zeros(8) # indices 0..7
    for j in range(8):
        val = 0.0
        # P'' contribution
        if j <= 5:
            val += 2*(j+1)*(2*j+1)*p[j+1]
        # -2xP' contribution
        if j >= 1 and j <= 6:
            val += -4*j*p[j]
        # x^2 P contribution
        if j >= 1 and j <= 7:
            val += p[j-1]
        # -P contribution
        if j <= 6:
            val += -p[j]
        r[j] = val
        
    # Now compute \int R(x)^2 e^{-x^2} dx
    # R(x) = \sum_{j=0}^7 r_j x^{2j}
    # R(x)^2 = \sum_{k=0}^{14} s_k x^{2k}
    # s = convolve(r, r)
    s = np.convolve(r, r)
    
    # Integrate s_k x^{2k} e^{-x^2}
    # Need factors up to k=14
    factors_R = np.zeros(15)
    for k in range(15):
        factors_R[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I_R = s * factors_R
    norm_R_sq = I_R[0]
    
    # <xi^2> = norm_R_sq / (4 * pi^2 * norm_sq)
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    
    # Product
    product = math.sqrt(x2 * xi2)
    
    return product

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 # 150s limit, safe margin
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    # We will optimize on the unit sphere because the objective function is scale-invariant.
    # Let c = r * u, where u is a unit vector. The bound depends only on u.
    # We parameterize u using spherical coordinates (theta, phi).
    
    def bound_from_spherical(theta, phi):
        c0 = np.sin(theta) * np.cos(phi)
        c4 = np.sin(theta) * np.sin(phi)
        c8 = np.cos(theta)
        return _compute_bound(c0, c4, c8)
    
    # Convert seed to spherical
    c_seed_norm = seed_coeffs / np.linalg.norm(seed_coeffs)
    c0, c4, c8 = c_seed_norm
    theta_init = np.arccos(np.clip(c8, -1.0, 1.0))
    phi_init = np.arctan2(c4, c0)
    if phi_init < 0:
        phi_init += 2 * np.pi

    best_val = bound_from_spherical(theta_init, phi_init)
    best_theta = theta_init
    best_phi = phi_init
    
    # 1. Global search using Differential Evolution on the 2D spherical parameter space
    # Symmetry: f(c) = f(-c), so we can restrict theta to [0, pi/2]
    if time.time() < deadline - 40.0:
        try:
            # bounds for theta: (0, pi/2), phi: (0, 2pi)
            bounds = [(1e-6, np.pi / 2 - 1e-6), (1e-6, 2 * np.pi - 1e-6)]
            
            de_result = scipy.optimize.differential_evolution(
                bound_from_spherical,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-10,
                popsize=30,
                polish=True
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_theta, best_phi = de_result.x
        except Exception:
            pass
    
    # 2. Local refinement with Nelder-Mead
    if time.time() < deadline - 15.0:
        try:
            res = scipy.optimize.minimize(
                bound_from_spherical,
                x0=[best_theta, best_phi],
                method='Nelder-Mead',
                options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 10000, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_theta, best_phi = res.x
        except Exception:
            pass
            
    # 3. Further polish with L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            res2 = scipy.optimize.minimize(
                bound_from_spherical,
                x0=[best_theta, best_phi],
                method='L-BFGS-B',
                bounds=[(1e-6, np.pi / 2 - 1e-6), (1e-6, 2 * np.pi - 1e-6)],
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_theta, best_phi = res2.x
        except Exception:
            pass

    # Convert back to Cartesian coefficients
    c0 = np.sin(best_theta) * np.cos(best_phi)
    c4 = np.sin(best_theta) * np.sin(best_phi)
    c8 = np.cos(best_theta)
    
    # The task asks for "three coefficients". The scale is arbitrary for the bound, 
    # but the seed program returned specific values. 
    # However, the interface says "construct() -> three coefficients".
    # Usually, returning the normalized vector is fine if the evaluator normalizes,
    # or if the bound is scale-invariant (which it is).
    # To be safe and match the "spirit" of the seed which had specific magnitudes,
    # we could scale it back, but since the bound is invariant, any scale works.
    # Let's return the unit vector coefficients.
    
    return np.array([c0, c4, c8])
# EVOLVE-BLOCK-END
