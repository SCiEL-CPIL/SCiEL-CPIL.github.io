"""Optimized program for uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill

    # Hermite polynomial values at 0
    h0_0 = 1.0
    h4_0 = 3.0
    h8_0 = 105.0
    h12_0 = 9240.0

    def get_c12(c0, c4, c8):
        return -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0

    # Precompute grid - use a smaller grid for speed, accuracy is sufficient
    x = np.linspace(-12, 12, 10000)
    dx = x[1] - x[0]
    
    # Precompute Hermite polynomials on grid
    H0 = eval_hermite(0, x)
    H4 = eval_hermite(4, x)
    H8 = eval_hermite(8, x)
    H12 = eval_hermite(12, x)
    gauss = np.exp(-x**2 / 2.0)
    
    # Precompute weighted basis vectors for efficient norm computation
    # w_i * H_i * gauss
    wH0 = gauss * H0
    wH4 = gauss * H4
    wH8 = gauss * H8
    wH12 = gauss * H12
    
    # Precompute inner products for norm calculation
    # norm_sq = sum(weights * (c0*wH0 + c4*wH4 + c8*wH8 + c12*wH12)^2) * dx
    # = c0^2 * <wH0,wH0> + c4^2 * <wH4,wH4> + ... + cross terms
    # where <a,b> = sum(weights * a * b) * dx
    w = np.ones_like(x)
    w[0] = 0.5
    w[-1] = 0.5
    
    # Precompute all pairwise inner products
    I00 = np.sum(w * wH0 * wH0) * dx
    I04 = np.sum(w * wH0 * wH4) * dx
    I08 = np.sum(w * wH0 * wH8) * dx
    I012 = np.sum(w * wH0 * wH12) * dx
    I44 = np.sum(w * wH4 * wH4) * dx
    I48 = np.sum(w * wH4 * wH8) * dx
    I412 = np.sum(w * wH4 * wH12) * dx
    I88 = np.sum(w * wH8 * wH8) * dx
    I812 = np.sum(w * wH8 * wH12) * dx
    I1212 = np.sum(w * wH12 * wH12) * dx
    
    # Precompute absolute values for L1 norm (vectorized)
    abs_wH0 = np.abs(wH0)
    abs_wH4 = np.abs(wH4)
    abs_wH8 = np.abs(wH8)
    abs_wH12 = np.abs(wH12)

    def compute_bound(c0, c4, c8):
        if time.time() > deadline:
            return 1e10
        c12 = get_c12(c0, c4, c8)

        # Compute norm squared using precomputed inner products
        norm_sq = (c0*c0*I00 + c4*c4*I44 + c8*c8*I88 + c12*c12*I1212 +
                   2*c0*c4*I04 + 2*c0*c8*I08 + 2*c0*c12*I012 +
                   2*c4*c8*I48 + 2*c4*c12*I412 + 2*c8*c12*I812)
        
        if norm_sq < 1e-15:
            return 1e10
        
        norm = np.sqrt(norm_sq)
        
        # L1 norm: sum(weights * |g|) * dx where g = (poly * gauss) / norm
        # = (1/norm) * sum(weights * |c0*wH0 + c4*wH4 + c8*wH8 + c12*wH12|) * dx
        poly_gauss = c0 * wH0 + c4 * wH4 + c8 * wH8 + c12 * wH12
        l1_raw = np.sum(w * np.abs(poly_gauss)) * dx
        l1_g = l1_raw / norm
        
        return l1_g * l1_g

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0

    best_score = compute_bound(c0_init, c4_init, c8_init)
    best_coeffs = [c0_init, c4_init, c8_init]

    # Generate diverse starting points with adaptive refinement
    starts = [[c0_init, c4_init, c8_init]]
    
    # Multi-scale exploration
    np.random.seed(42)
    
    # Coarse scale
    for i in range(6):
        if time.time() > deadline - 40.0:
            break
        x0 = [
            c0_init + np.random.randn() * 0.5,
            c4_init + np.random.randn() * 0.1,
            c8_init + np.random.randn() * 0.005
        ]
        starts.append(x0)
    
    # Fine scale
    for i in range(6):
        if time.time() > deadline - 35.0:
            break
        x0 = [
            c0_init + np.random.randn() * 0.1,
            c4_init + np.random.randn() * 0.02,
            c8_init + np.random.randn() * 0.001
        ]
        starts.append(x0)
    
    # Structured single-axis perturbations
    for delta_c0 in [-0.3, -0.15, 0.15, 0.3]:
        if time.time() > deadline - 30.0:
            break
        starts.append([c0_init + delta_c0, c4_init, c8_init])
    
    for delta_c4 in [-0.04, -0.02, 0.02, 0.04]:
        if time.time() > deadline - 30.0:
            break
        starts.append([c0_init, c4_init + delta_c4, c8_init])

    for delta_c8 in [-0.0015, -0.0005, 0.0005, 0.0015]:
        if time.time() > deadline - 30.0:
            break
        starts.append([c0_init, c4_init, c8_init + delta_c8])

    # Phase 1: L-BFGS-B from multiple starts
    for x0 in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            res_lbfgs = minimize(
                lambda c: compute_bound(*c),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_lbfgs.fun < best_score:
                best_score = res_lbfgs.fun
                best_coeffs = res_lbfgs.x.tolist()
        except Exception:
            pass

    # Phase 2: Nelder-Mead polish
    if time.time() < deadline - 10.0 and best_score < 1e10:
        try:
            res_nm = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_coeffs = res_nm.x.tolist()
        except Exception:
            pass

    # Phase 3: Final L-BFGS-B polish with very tight tolerances
    if time.time() < deadline - 5.0 and best_score < 1e10:
        try:
            res_final = minimize(
                lambda c: compute_bound(*c),
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_coeffs = res_final.x.tolist()
        except Exception:
            pass

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
