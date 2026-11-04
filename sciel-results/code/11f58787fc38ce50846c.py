"""Optimized program: analytic exactness + refined multi-stage global search."""
import numpy as np
import time
import scipy.optimize
import math
import itertools

# Pre-compute factorials up to 24 for speed
_FACTS = [1.0]
for i in range(1, 25):
    _FACTS.append(_FACTS[-1] * i)

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    Returns the product sqrt(<x^2> * <xi^2>).
    """
    # Hermite values at 0: H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=1330
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    # a0 = c0 + 12c4 + 168c8 + 1330c12 = 0
    a0 = 0.0
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Convolution to get coefficients of P(x)^2
    # q[k] is coeff of x^{2k} in P(x)^2
    q = np.convolve(p, p)
    # q has length 13, indices 0 to 12 corresponding to x^0 to x^24
    
    # Integrate x^{2k} e^{-x^2}
    # \int_{-inf}^{inf} x^{2k} e^{-x^2} dx = \sqrt{\pi} (2k-1)!! = \sqrt{\pi} 2^k k!
    # factors[k] = sqrt(pi) * 2^k * k!
    # We can compute this vectorized
    k = np.arange(13)
    factors = math.sqrt(math.pi) * (2.0 ** k) * np.array(_FACTS[:13])
        
    I = q * factors
    # I[k] = \int x^{2k} P(x)^2 e^{-x^2} dx
    
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10 # Invalid
    
    # Variance <x^2> = I[1] / I[0]
    x2 = I[1] / I[0]
    
    # Calculate g'(x) to get <xi^2>
    # g(x) = P(x) e^{-x^2/2}
    # g'(x) = (P'(x) - xP(x)) e^{-x^2/2}
    # Let R(x) = P'(x) - xP(x), which is odd.
    # We need \int R(x)^2 e^{-x^2} dx.
    
    # P(x) = \sum_{i=0}^6 p_i x^{2i}
    # P'(x) = \sum_{i=1}^6 2i p_i x^{2i-1}
    # xP(x) = \sum_{i=0}^6 p_i x^{2i+1}
    # R(x) = \sum_{i=1}^6 2i p_i x^{2i-1} - \sum_{i=0}^6 p_i x^{2i+1}
    # R(x) = \sum_{j=0}^6 r_j x^{2j+1} where:
    #   For x^{2j+1}: coefficient from P' is 2(j+1)p_{j+1} (if j+1 <= 6, i.e., j <= 5)
    #   Coefficient from -xP is -p_j
    # So r_j = 2(j+1)p_{j+1} - p_j for j=0..5, and r_6 = -p_6
    
    r = np.zeros(7) # coefficients for x^1, x^3, ..., x^13
    for j in range(7):
        val = 0.0
        # From P': 2(j+1)p_{j+1} if j+1 <= 6
        if j <= 5:
            val += 2*(j+1)*p[j+1]
        # From -xP: -p_j
        val += -p[j]
        r[j] = val
    
    # Compute \int R(x)^2 e^{-x^2} dx
    # R(x)^2 = \sum_{j,k} r_j r_k x^{2j+1} x^{2k+1} = \sum_{j,k} r_j r_k x^{2(j+k)+2}
    # Let m = j+k, then coefficient of x^{2m+2} is \sum_{j+k=m} r_j r_k
    # This is the convolution of r with r, but shifted.
    # Actually, \int x^{2(j+k)+2} e^{-x^2} dx = \sqrt{\pi} 2^{j+k+1} (j+k+1)!
    
    # Let's compute the sum directly
    integral_R_sq = 0.0
    for j in range(7):
        for k in range(7):
            m = j + k
            # \int x^{2m+2} e^{-x^2} dx = \sqrt{\pi} 2^{m+1} (m+1)!
            integral_R_sq += r[j] * r[k] * math.sqrt(math.pi) * (2**(m+1)) * _FACTS[m+1]
    
    # <xi^2> = (1/4\pi^2) * \int |g'|^2 dx / \int |g|^2 dx
    xi2 = integral_R_sq / (4 * math.pi**2 * norm_sq)
    
    # Product
    product = math.sqrt(x2 * xi2)
    
    return product

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # 175s limit to be safe
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    def _objective(x):
        return _compute_bound(x[0], x[1], x[2])
    
    # 1. Local Optimization from Seed
    try:
        res_local = scipy.optimize.minimize(
            _objective,
            seed_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if res_local.fun < best_val:
            best_val = res_local.fun
            best_coeffs = res_local.x.copy()
    except Exception:
        pass
    
    # 2. Global Search using Differential Evolution
    # Bounds are set wide enough to explore the space
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    if time.time() < deadline - 40.0:
        try:
            # DE is stochastic, use fixed seed for reproducibility
            de_res = scipy.optimize.differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-10,
                popsize=20,
                polish=False
            )
            
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_coeffs = de_res.x.copy()
        except Exception:
            pass
            
    # 2b. Nelder-Mead Global Exploration
    # Try a few random starting points with Nelder-Mead to catch different basins
    if time.time() < deadline - 20.0:
        np.random.seed(123)
        for _ in range(5):
            if time.time() > deadline - 15.0:
                break
            start_point = np.random.uniform(-3, 3, 3)
            try:
                res_nm = scipy.optimize.minimize(
                    _objective,
                    start_point,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                if res_nm.fun < best_val:
                    best_val = res_nm.fun
                    best_coeffs = res_nm.x.copy()
            except Exception:
                pass

    # 3. Multi-start Local Optimization around Best
    if time.time() < deadline - 10.0:
        np.random.seed(456)
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            # Start near the current best with some noise
            noise = np.random.normal(0, 0.1, 3)
            start_point = best_coeffs + noise
            try:
                res_restart = scipy.optimize.minimize(
                    _objective,
                    start_point,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14}
                )
                if res_restart.fun < best_val:
                    best_val = res_restart.fun
                    best_coeffs = res_restart.x.copy()
            except Exception:
                pass

    # 4. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        try:
            res_final = scipy.optimize.minimize(
                _objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass

    return best_coeffs
