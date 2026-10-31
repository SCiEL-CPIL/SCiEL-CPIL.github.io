"""Program exploiting scale invariance for the uncertainty inequality task."""
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar
import time

# Precompute Gauss-Hermite nodes and weights 
# 15 points is exact for polynomials of degree up to 29, which is sufficient for our degree 12 product (degree 24)
_N_GQ = 15
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)

# Precompute powers and Hermite polynomials
_X2 = _X_GQ * _X_GQ
_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = _X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3 + 55440 * _X2**2 - 100800 * _X2 + 10395

# Precompute derivatives of Hermite polynomials
# H_n'(x) = 2n H_{n-1}(x)
# H4'(x) = 8 H3(x) = 8(4x^3 - 12x) = 32x^3 - 96x ? No, standard physics Hermite polynomials differ by scaling.
# The code uses probabilists' or physicists'?
# H4(x) = x^4 - 6x^2 + 3. This is the probabilists' Hermite polynomial He_4(x) scaled?
# Actually, He_4(x) = x^4 - 6x^2 + 3.
# Derivative He_4'(x) = 4x^3 - 12x.
# Let's verify the precomputed derivatives in the original code.
# _DH4 = 4 * _X3 - 12 * _X_GQ. This matches d/dx (x^4 - 6x^2 + 3) = 4x^3 - 12x. Correct.
# _DH8: H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105.
# d/dx = 8x^7 - 168x^5 + 840x^3 - 840x.
# _DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ. Correct.
# _DH12: H12(x) = x^12 - 84x^10 + 1260x^8 - 12600x^6 + 55440x^4 - 100800x^2 + 10395.
# d/dx = 12x^11 - 840x^9 + 10080x^7 - 75600x^5 + 221760x^3 - 201600x.
# _DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ. Correct.

_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11

_DH4 = 4 * _X3 - 12 * _X_GQ
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
_DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ

# Weights for integrals
_W_I0 = _W_GQ
_W_I2 = _W_GQ * _X2

# Precompute derivative basis vectors for efficiency
# d = -(a + 3b + 105c) / 10395
# dP/da = H0 - H12/10395
# dP/db = H4 - 3H12/10395
# dP/dc = H8 - 105H12/10395
_dP_da = _H0 - _H12 / 10395.0
_dP_db = _H4 - 3 * _H12 / 10395.0
_dP_dc = _H8 - 105 * _H12 / 10395.0

# dP_prime/da = -DH12/10395
# dP_prime/db = DH4 - 3DH12/10395
# dP_prime/dc = DH8 - 105DH12/10395
_dPp_da = -_DH12 / 10395.0
_dPp_db = _DH4 - 3 * _DH12 / 10395.0
_dPp_dc = _DH8 - 105 * _DH12 / 10395.0

# d(g_prime)/da = dPp_da - x * dP_da
_dgp_da = _dPp_da - _X_GQ * _dP_da
_dgp_db = _dPp_db - _X_GQ * _dP_db
_dgp_dc = _dPp_dc - _X_GQ * _dP_dc


def _objective_and_grad(params):
    """
    Compute the uncertainty product and its gradient.
    params: [a, b, c]
    Returns: (objective_value, gradient_vector)
    """
    a, b, c = params
    
    # Calculate d from constraint P(0)=0
    # P(0) = a*H0(0) + b*H4(0) + c*H8(0) + d*H12(0) = 0
    # H0(0)=1, H4(0)=3, H8(0)=105, H12(0)=10395
    d = -(a + 3 * b + 105 * c) / 10395.0
    
    # Construct P and P_prime
    P = a * _H0 + b * _H4 + c * _H8 + d * _H12
    P_prime = b * _DH4 + c * _DH8 + d * _DH12
    
    # Compute Integrals
    I0 = np.dot(_W_I0, P * P)
    I2 = np.dot(_W_I2, P * P)
    
    g_prime_poly = P_prime - _X_GQ * P
    J0 = np.dot(_W_I0, g_prime_poly * g_prime_poly)
    
    # Check for validity
    if I0 <= 1e-15 or J0 <= 1e-15 or I2 <= 1e-15:
        return 1e10, np.array([0.0, 0.0, 0.0])
    
    # Objective: sqrt(I2 * J0) / I0
    product = np.sqrt(I2 * J0) / I0
    
    # Compute Gradient
    # dI0/da = 2 * sum(w * P * dP_da)
    dI0_da = 2 * np.dot(_W_I0, P * _dP_da)
    dI0_db = 2 * np.dot(_W_I0, P * _dP_db)
    dI0_dc = 2 * np.dot(_W_I0, P * _dP_dc)
    
    # dI2/da = 2 * sum(w*x^2 * P * dP_da)
    dI2_da = 2 * np.dot(_W_I2, P * _dP_da)
    dI2_db = 2 * np.dot(_W_I2, P * _dP_db)
    dI2_dc = 2 * np.dot(_W_I2, P * _dP_dc)
    
    # dJ0/da = 2 * sum(w * g_pp * dgp_da)
    dJ0_da = 2 * np.dot(_W_I0, g_prime_poly * _dgp_da)
    dJ0_db = 2 * np.dot(_W_I0, g_prime_poly * _dgp_db)
    dJ0_dc = 2 * np.dot(_W_I0, g_prime_poly * _dgp_dc)
    
    # Gradient of product f = sqrt(I2*J0)/I0
    # df/da = (dI2_da * J0 + I2 * dJ0_da) / (2 * I2 * J0) - dI0_da / I0
    # This formula is derived from d(sqrt(u)/v) where u=I2*J0, v=I0
    # d(sqrt(u)/v) = (1/(2sqrt(u)v)) du - sqrt(u)/v^2 dv
    # du = dI2*J0 + I2*dJ0
    # dv = dI0
    # term1 = (dI2*J0 + I2*dJ0) / (2 * sqrt(I2*J0) * I0)
    # term2 = sqrt(I2*J0) / I0^2 * dI0
    # Note: sqrt(I2*J0)/I0 = product
    # term1 = (dI2*J0 + I2*dJ0) / (2 * product * I0^2) ? No.
    # Let's re-verify the derivative formula used in the seed.
    # Seed used: grad_a = (dI2_da * J0 + I2 * dJ0_da) / (2.0 * I2 * J0) - dI0_da / I0
    # Let f = (I2 J0)^0.5 I0^-1
    # ln f = 0.5 ln I2 + 0.5 ln J0 - ln I0
    # d ln f = 0.5 dI2/I2 + 0.5 dJ0/J0 - dI0/I0
    # df = f * (0.5 dI2/I2 + 0.5 dJ0/J0 - dI0/I0)
    # df/da = f * (0.5 dI2_da/I2 + 0.5 dJ0_da/J0 - dI0_da/I0)
    # The seed formula: (dI2 J0 + I2 dJ0) / (2 I2 J0) = 0.5 dI2/I2 + 0.5 dJ0/J0.
    # So Seed Formula = f * (0.5 dI2/I2 + 0.5 dJ0/J0 - dI0/I0).
    # This is correct.
    
    grad_a = product * (0.5 * dI2_da / I2 + 0.5 * dJ0_da / J0 - dI0_da / I0)
    grad_b = product * (0.5 * dI2_db / I2 + 0.5 * dJ0_db / J0 - dI0_db / I0)
    grad_c = product * (0.5 * dI2_dc / I2 + 0.5 * dJ0_dc / J0 - dI0_dc / I0)
    
    return product, np.array([grad_a, grad_b, grad_c])


def construct():
    start_time = time.time()
    deadline = 170.0  # Safety margin before 180s limit
    
    # Seed value
    seed_result = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_params = seed_result.copy()
    best_score, _ = _objective_and_grad(best_params)
    
    # Exploit scale invariance: fix a=1, search over ratios (b/a, c/a)
    # Seed ratios: b/a = (1/25)/(-113/100) = -100/(25*113) = -4/113 ≈ -0.0354
    #             c/a = (1/3240)/(-113/100) = -100/(3240*113) = -1/(324*1.13) ≈ -0.00273
    
    # Phase 1: Dense grid search over (b/a, c/a) ratios
    # b/a ranges: seed is ~-0.035, let's search [-0.5, 0.5]
    # c/a ranges: seed is ~-0.0027, let's search [-0.1, 0.1]
    
    n_b = 200
    n_c = 100
    
    b_vals = np.linspace(-0.5, 0.5, n_b)
    c_vals = np.linspace(-0.1, 0.1, n_c)
    
    grid_best_score = best_score
    grid_best_b = seed_result[1] / seed_result[0]
    grid_best_c = seed_result[2] / seed_result[0]
    
    for i_b, b_val in enumerate(b_vals):
        if time.time() > deadline - 20:
            break
        for i_c, c_val in enumerate(c_vals):
            # Fix a = 1, so params = [1, b_val, c_val]
            params = np.array([1.0, b_val, c_val])
            score, _ = _objective_and_grad(params)
            if score < grid_best_score:
                grid_best_score = score
                grid_best_b = b_val
                grid_best_c = c_val
    
    if grid_best_score < best_score:
        best_score = grid_best_score
        best_params = np.array([1.0, grid_best_b, grid_best_c])
    
    # Phase 2: Local refinement using L-BFGS-B from the best grid point and seed
    points_to_refine = [best_params.copy(), seed_result.copy()]
    
    # Add some neighbors of the best grid point
    if grid_best_score < best_score or grid_best_score < 1e9:
        neighbors = [
            [1.0, grid_best_b + 0.01, grid_best_c],
            [1.0, grid_best_b - 0.01, grid_best_c],
            [1.0, grid_best_b, grid_best_c + 0.005],
            [1.0, grid_best_b, grid_best_c - 0.005],
        ]
        points_to_refine.extend(neighbors)
    
    for p in points_to_refine:
        if time.time() > deadline - 5:
            break
        
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                p,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 2:
        try:
            result = minimize(
                lambda x: _objective_and_grad(x)[0],
                best_params,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final verification
    final_score, _ = _objective_and_grad(best_params)
    if final_score < best_score:
        best_score = final_score
    
    # Ensure we don't return worse than seed
    seed_score, _ = _objective_and_grad(seed_result)
    if best_score > seed_score:
        best_params = seed_result.copy()
    
    return best_params
# EVOLVE-BLOCK-END
