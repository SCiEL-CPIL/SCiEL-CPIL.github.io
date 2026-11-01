"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_hermite
import time

def _hermite_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n(x)."""
    # H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, etc.
    # Use recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])  # 2x
    h_nm1 = np.array([1.0])  # H_0
    h_n = np.array([0.0, 2.0])  # H_1
    for k in range(2, n + 1):
        h_np1 = np.zeros(len(h_n) + 1)
        h_np1[1:] += 2.0 * h_n
        h_np1[:-1] -= 2.0 * (k - 1) * h_nm1
        h_nm1, h_n = h_n, h_np1
    return h_n

def _poly_eval(coeffs, x):
    """Evaluate polynomial with coefficients (ascending order) at x."""
    result = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        result += c * x**i
    return result

def construct():
    deadline = time.time() + 190  # 190s limit, 20s margin
    
    # Hermite polynomials H_0, H_4, H_8, H_12
    H0_coeffs = _hermite_coeffs(0)
    H4_coeffs = _hermite_coeffs(4)
    H8_coeffs = _hermite_coeffs(8)
    H12_coeffs = _hermite_coeffs(12)
    
    # Values at x=0 for the constraint
    H0_0 = _poly_eval(H0_coeffs, np.array([0.0]))[0]
    H4_0 = _poly_eval(H4_coeffs, np.array([0.0]))[0]
    H8_0 = _poly_eval(H8_coeffs, np.array([0.0]))[0]
    H12_0 = _poly_eval(H12_coeffs, np.array([0.0]))[0]
    
    # Constraint: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
    
    # Compute integrals for the expectation values
    # <x^2> = sum_nm c_n c_m * <H_n x^2 H_m> / sum_n c_n^2 * <H_n^2>
    # where <f g> = integral f(x) g(x) e^{-x^2} dx
    
    # Using orthogonality: <H_n H_m> = 2^n n! sqrt(pi) delta_nm
    # <H_n x^2 H_m> can be computed via x^2 H_n = 0.5 H_{n+2} + n H_{n-2}
    # So <H_n x^2 H_m> = 0.5 <H_n H_{m+2}> + n <H_n H_{m-2}>
    # = 0.5 * 2^m m! sqrt(pi) delta_{n,m+2} + n * 2^{m-2} (m-2)! sqrt(pi) delta_{n,m-2}
    
    modes = [0, 4, 8, 12]
    n_modes = len(modes)
    
    # Precompute norms: norm[n] = 2^n n! sqrt(pi)
    import math
    norms = {}
    for n in modes:
        norms[n] = (2.0 ** n) * math.factorial(n) * math.sqrt(math.pi)
    
    # Matrix M where M[i,j] = <H_{modes[i]} x^2 H_{modes[j]}>
    M = np.zeros((n_modes, n_modes))
    for i, ni in enumerate(modes):
        for j, mj in enumerate(modes):
            # <H_ni x^2 H_mj> = 0.5 * <H_ni H_{mj+2}> + ni * <H_ni H_{mj-2}>
            val = 0.0
            if mj + 2 in norms and ni == mj + 2:
                val += 0.5 * norms[mj + 2]
            if mj - 2 >= 0 and mj - 2 in norms and ni == mj - 2:
                val += ni * norms[mj - 2]
            M[i, j] = val
    
    # Norm matrix N where N[i,i] = norms[modes[i]]
    N = np.diag([norms[m] for m in modes])
    
    def compute_uncertainty(c0, c4, c8):
        """Compute the uncertainty product <x^2><xi^2> for given coefficients."""
        c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
        coeffs = np.array([c0, c4, c8, c12])
        
        # Normalization: <f^2> = coeffs^T N coeffs
        norm_sq = coeffs @ N @ coeffs
        if abs(norm_sq) < 1e-15:
            return np.inf
        
        # <x^2> = coeffs^T M coeffs / norm_sq
        x2 = coeffs @ M @ coeffs / norm_sq
        
        # For Fourier transform, <xi^2> = <x^2> (by symmetry of the basis)
        # So the product is x2^2
        return x2 * x2  # This is <x^2><xi^2>
    
    # Grid search over c0, c4, c8
    # The coefficients can be scaled, so we can fix one and search over ratios
    # Let's fix c0 = 1 and search over c4/c0 and c8/c0
    # Or better: search over a 2D grid of (c4, c8) with c0 = 1
    
    best_val = np.inf
    best_coeffs = None
    
    # Coarse grid
    n_grid = 200
    c4_range = np.linspace(-5, 5, n_grid)
    c8_range = np.linspace(-5, 5, n_grid)
    
    for i, c4 in enumerate(c4_range):
        if time.time() > deadline:
            break
        for c8 in c8_range:
            val = compute_uncertainty(1.0, c4, c8)
            if val < best_val:
                best_val = val
                best_coeffs = np.array([1.0, c4, c8])
    
    # Refine around the best point
    if best_coeffs is not None and time.time() < deadline - 30:
        c0_best, c4_best, c8_best = best_coeffs
        # Local refinement
        step = 0.1
        for _ in range(5):
            improved = False
            for dc0 in [-step, 0, step]:
                for dc4 in [-step, 0, step]:
                    for dc8 in [-step, 0, step]:
                        if dc0 == 0 and dc4 == 0 and dc8 == 0:
                            continue
                        test_c0 = c0_best + dc0
                        test_c4 = c4_best + dc4
                        test_c8 = c8_best + dc8
                        if abs(test_c0) < 1e-10:
                            continue
                        val = compute_uncertainty(test_c0, test_c4, test_c8)
                        if val < best_val:
                            best_val = val
                            best_coeffs = np.array([test_c0, test_c4, test_c8])
                            c0_best, c4_best, c8_best = test_c0, test_c4, test_c8
                            improved = True
            if not improved:
                break
            step *= 0.5
    
    # Extract the three coefficients (c0, c4, c8)
    if best_coeffs is not None:
        return np.array(best_coeffs[:3])
    else:
        # Fallback to original
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
# EVOLVE-BLOCK-END

