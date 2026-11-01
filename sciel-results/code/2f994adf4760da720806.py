"""Optimized Hermite-Gaussian uncertainty inequality solver."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _hermite_moments(c):
    """
    Compute Var(x) and Var(p) for a state with coefficients c on Hermite functions
    at indices [0, 4, 8, 12]. Returns (var_x, var_p).
    """
    indices = np.array([0, 4, 8, 12])
    n = len(indices)
    
    # Normalization
    norm_sq = np.sum(np.abs(c)**2)
    if norm_sq < 1e-30:
        return (np.inf, np.inf)
    c_norm = c / np.sqrt(norm_sq)
    
    # <x> and <p> are zero by parity (all even indices)
    # Compute <x^2> and <p^2>
    
    # x operator: x|k> = sqrt(k/2)|k-1> + sqrt((k+1)/2)|k+1>
    # For state |psi> = sum_i c_i |i>, 
    # <x^2> = sum_{i,j} c_i* c_j <i|x^2|j>
    # x^2|k> = sqrt(k/2)x|k-1> + sqrt((k+1)/2)x|k+1>
    #        = sqrt(k/2)[sqrt((k-1)/2)|k-2> + sqrt(k/2)|k>] + sqrt((k+1)/2)[sqrt((k+1)/2)|k> + sqrt((k+2)/2)|k+2>]
    #        = sqrt(k(k-1)/4)|k-2> + (k/2 + (k+1)/2)|k> + sqrt((k+1)(k+2)/4)|k+2>
    #        = sqrt(k(k-1)/4)|k-2> + (2k+1)/2 |k> + sqrt((k+1)(k+2)/4)|k+2>
    
    # Similarly for p^2, but p = -i d/dx for these, and p^2 has same diagonal as x^2
    # Actually for the standard harmonic oscillator, x^2 and p^2 have the same matrix elements
    # in the number basis up to the sign convention, but <p^2> = <x^2> for the same state
    # No wait, that's not right. Let me be careful.
    
    # For the standard quantized harmonic oscillator with H = x^2 + p^2 (in units where m=omega=1),
    # <n|x^2|n> = <n|p^2|n> = (2n+1)/2
    # And the off-diagonal elements are related but not identical.
    
    # Let me just compute x^2 matrix elements directly.
    # x^2|k> = sqrt(k(k-1)/4)|k-2> + (2k+1)/2 |k> + sqrt((k+1)(k+2)/4)|k+2>
    
    # For p: p|k> = i(sqrt(k/2)|k-1> - sqrt((k+1)/2)|k+1>)
    # p^2|k> = p[i(sqrt(k/2)|k-1> - sqrt((k+1)/2)|k+1>)]
    #         = i[sqrt(k/2)(i(sqrt((k-1)/2)|k-2> - sqrt(k/2)|k>)) - sqrt((k+1)/2)(i(sqrt((k+1)/2)|k> - sqrt((k+2)/2)|k+2>))]
    #         = -[sqrt(k(k-1)/4)|k-2> + k/2|k> + (k+1)/2|k> + sqrt((k+1)(k+2)/4)|k+2>]
    # Wait, let me redo this more carefully.
    
    # p|k> = i(sqrt(k/2)|k-1> - sqrt((k+1)/2)|k+1>)
    # p^2|k> = p(p|k>) = i * p(sqrt(k/2)|k-1> - sqrt((k+1)/2)|k+1>)
    #         = i * [sqrt(k/2) * p|k-1> - sqrt((k+1)/2) * p|k+1>]
    #         = i * [sqrt(k/2) * i(sqrt((k-1)/2)|k-2> - sqrt(k/2)|k>) - sqrt((k+1)/2) * i(sqrt((k+1)/2)|k> - sqrt((k+2)/2)|k+2>)]
    #         = -[sqrt(k(k-1)/4)|k-2> + k/2|k> + (k+1)/2|k> - sqrt((k+1)(k+2)/4)|k+2>]
    #         = -sqrt(k(k-1)/4)|k-2> - (2k+1)/2|k> + sqrt((k+1)(k+2)/4)|k+2>
    
    # Hmm, that gives a negative sign. Let me use a different approach.
    # Actually, for the uncertainty principle, we need <x^2> - <x>^2 and <p^2> - <p>^2.
    # By symmetry, <x> = <p> = 0 for even states.
    # And it turns out <x^2> = <p^2> for the same state in this basis? No.
    
    # Let me just use the known result: for a state in the number basis,
    # <x^2> = sum_{i,j} c_i* c_j <i|x^2|j>
    # <p^2> = sum_{i,j} c_i* c_j <i|p^2|j>
    
    # The matrix elements:
    # <i|x^2|j> is nonzero only if |i-j| = 0 or 2
    # <i|x^2|i> = (2i+1)/2
    # <i|x^2|i+2> = sqrt((i+1)(i+2)/4)
    # <i|x^2|i-2> = sqrt(i(i-1)/4)
    
    # For p^2:
    # <i|p^2|i> = (2i+1)/2
    # <i|p^2|i+2> = -sqrt((i+1)(i+2)/4)  [note the negative sign!]
    # <i|p^2|i-2> = -sqrt(i(i-1)/4)
    
    # Wait, I need to verify this. Let me use the creation/annihilation operators.
    # x = (a + a†)/sqrt(2), p = -i(a - a†)/sqrt(2)
    # x^2 = (a^2 + a a† + a† a + a†2)/2 = (a^2 + 2a†a + 1 + a†2)/2
    # p^2 = -1/2 (a^2 - a a† - a† a + a†2) = -1/2 (a^2 - 2a†a - 1 + a†2) = (2a†a + 1 - a^2 - a†2)/2
    
    # So <n|x^2|n> = (2n+1)/2, <n|p^2|n> = (2n+1)/2
    # <n|x^2|n+2> = <n|(a^2 + a†2)/2|n+2> = <n|a†2|n+2>/2 + <n|a^2|n+2>/2
    # a†2|n+2> = sqrt((n+3)(n+2))|n+4>, so <n|a†2|n+2> = 0
    # a^2|n+2> = sqrt((n+2)(n+1))|n>, so <n|a^2|n+2>/2 = sqrt((n+1)(n+2))/2
    # So <n|x^2|n+2> = sqrt((n+1)(n+2))/2
    
    # <n|p^2|n+2> = <n|(2a†a + 1 - a^2 - a†2)/2|n+2>
    # = (2(n+2) + 1)/2 * <n|n+2> - <n|a^2|n+2>/2 - <n|a†2|n+2>/2
    # = 0 - sqrt((n+1)(n+2))/2 - 0
    # = -sqrt((n+1)(n+2))/2
    
    # Great, so the off-diagonal elements have opposite signs.
    
    var_x = 0.0
    var_p = 0.0
    
    for i in range(n):
        for j in range(n):
            ni = indices[i]
            nj = indices[j]
            diff = ni - nj
            
            if diff == 0:
                diag = (2*ni + 1) / 2.0
                var_x += np.conj(c_norm[i]) * c_norm[j] * diag
                var_p += np.conj(c_norm[i]) * c_norm[j] * diag
            elif diff == 2:
                # <ni|x^2|nj> where ni = nj + 2
                offdiag_x = np.sqrt((nj + 1) * (nj + 2)) / 2.0
                offdiag_p = -offdiag_x
                var_x += np.conj(c_norm[i]) * c_norm[j] * offdiag_x
                var_p += np.conj(c_norm[i]) * c_norm[j] * offdiag_p
            elif diff == -2:
                # <ni|x^2|nj> where ni = nj - 2
                offdiag_x = np.sqrt(ni * (ni + 1)) / 2.0
                offdiag_p = -offdiag_x
                var_x += np.conj(c_norm[i]) * c_norm[j] * offdiag_x
                var_p += np.conj(c_norm[i]) * c_norm[j] * offdiag_p
            # Other differences: zero by parity/selection rules
    
    # Ensure real (should be real for real coefficients)
    var_x = float(np.real(var_x))
    var_p = float(np.real(var_p))
    
    if var_x < 0:
        var_x = 0.0
    if var_p < 0:
        var_p = 0.0
    
    return (var_x, var_p)


def _uncertainty_product(coeffs):
    """
    coeffs: array of [a0, a4, a8]
    Returns the upper bound on A(f)*A(F(f)), which is 4 * Var(x) * Var(p)
    for the normalized wavefunction.
    """
    a0, a4, a8 = coeffs
    
    # H_n(0) for n = 0, 4, 8, 12
    # H_0(0) = 1
    # H_4(0) = -3
    # H_8(0) = 105
    # H_12(0) = -10395
    H0_0 = 1.0
    H4_0 = -3.0
    H8_0 = 105.0
    H12_0 = -10395.0
    
    # Constraint: g(0) = 0 => a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    # a12 = -(a0 - 3*a4 + 105*a8) / (-10395) = (a0 - 3*a4 + 105*a8) / 10395
    a12 = (a0 - 3.0*a4 + 105.0*a8) / 10395.0
    
    c = np.array([a0, a4, a8, a12])
    
    var_x, var_p = _hermite_moments(c)
    
    # The uncertainty product bound is 4 * sqrt(Var(x) * Var(p))
    # But actually, A(f) * A(F(f)) where A is the "area" or width...
    # For a wavefunction, the uncertainty principle gives sigma_x * sigma_p >= 1/2
    # The "upper bound" here likely refers to the actual product sigma_x * sigma_p
    # which for the test function gives the constant in the inequality.
    
    # Looking at the task: "upper bound on A(f)*A(Fourier(f))"
    # This is likely sigma_x * sigma_p for the normalized g.
    
    product = var_x * var_p
    if product < 0:
        product = 0.0
    
    return np.sqrt(product) if product > 0 else 0.0


def _objective(coeffs):
    """Objective function to minimize."""
    try:
        val = _uncertainty_product(coeffs)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except:
        return 1e10


def construct():
    """
    Returns three coefficients [a0, a4, a8] that minimize the uncertainty product.
    The a12 coefficient is determined by the constraint g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    # Initial guess from prior state of the art
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    best_x = x0.copy()
    best_val = _objective(best_x)
    
    # Try multiple starting points
    start_points = [
        x0,
        np.array([1.0, -0.05, 0.0001]),
        np.array([-1.0, 0.05, -0.0001]),
        np.array([0.5, -0.02, 0.00005]),
        np.array([-0.5, 0.02, -0.00005]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
    ]
    
    for x_start in start_points:
        if time.time() > deadline - 30.0:
            break
        
        try:
            result = minimize(
                _objective,
                x_start,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except:
            pass
    
    # If we have time, do a finer local search around the best point
    if time.time() < deadline - 60.0:
        # Try Nelder-Mead for further refinement
        try:
            result2 = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_x = result2.x
        except:
            pass
    
    # Final check
    final_val = _objective(best_x)
    if final_val < best_val:
        best_val = final_val
    
    return np.array(best_x)

# EVOLVE-BLOCK-END
