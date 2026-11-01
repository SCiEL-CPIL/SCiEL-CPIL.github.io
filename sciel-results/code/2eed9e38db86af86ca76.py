"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

# Constants for the Hermite-Gaussian basis
# We work with degrees 0, 4, 8, 12
# g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
# Constraint: g(0) = 0

# Precompute constants
# H_n(0) for even n:
# H0(0) = 1
# H4(0) = 12
# H8(0) = 1680
# H12(0) = 39916800 / 120 = 332640? Let's compute carefully.
# Actually, let's use scipy to be safe, or compute analytically.

from scipy.special import eval_hermite

H0_0 = float(eval_hermite(0, 0))
H4_0 = float(eval_hermite(4, 0))
H8_0 = float(eval_hermite(8, 0))
H12_0 = float(eval_hermite(12, 0))

# Normalization factors: ||H_k(x) exp(-x^2/2)||^2 = 2^k k! sqrt(pi)
# Let N_k = 2^k k!
# We can ignore sqrt(pi) as it cancels in ratios.
N0 = 1.0
N4 = 2.0**4 * math.factorial(4)
N8 = 2.0**8 * math.factorial(8)
N12 = 2.0**12 * math.factorial(12)

# Inverse normalization factors for coefficients in orthonormal basis
# c_k = a_k / sqrt(N_k)
inv_sqrt_N0 = 1.0 / math.sqrt(N0)
inv_sqrt_N4 = 1.0 / math.sqrt(N4)
inv_sqrt_N8 = 1.0 / math.sqrt(N8)
inv_sqrt_N12 = 1.0 / math.sqrt(N12)


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 = A(f) * A(F).
    
    For a function g(x) = sum a_k H_k(x) exp(-x^2/2),
    the Fourier transform is F(g)(xi) = sum (-i)^k a_k H_k(xi) exp(-xi^2/2).
    
    The uncertainty A(f) is related to the standard deviation of x, and A(F) to the standard deviation of xi.
    For the Hermite-Gaussian basis, the variance of x for a state with coefficients c_k (in orthonormal basis) is:
    <x^2> = sum_k c_k^2 (k + 0.5) + off-diagonal terms.
    
    However, since our basis functions H_0, H_4, H_8, H_12 are not adjacent, the off-diagonal terms
    <m|x^2|n> for |m-n| != 2 are zero. Since the gap is 4, there are NO off-diagonal terms!
    
    So <x^2> = (c0^2 * 0.5 + c4^2 * 4.5 + c8^2 * 8.5 + c12^2 * 12.5) / (c0^2 + c4^2 + c8^2 + c12^2)
    
    Similarly for <xi^2> in frequency space.
    The Fourier transform maps H_k to (-i)^k H_k. So the coefficients in frequency space are
    d_k = (-i)^k c_k.
    Since we only have even k, (-i)^k = 1 for k=0,4,8,12. So d_k = c_k.
    Thus <xi^2> = <x^2>.
    
    Therefore, the product A(f)*A(F) is proportional to <x^2> (or the square root of it, depending on definition).
    The task asks for the upper bound on C4. Let's assume C4 = <x^2> * <xi^2> = <x^2>^2.
    Or perhaps A(f) = sqrt(<x^2>) and A(F) = sqrt(<xi^2>), so A(f)*A(F) = <x^2>.
    
    Let's compute <x^2> and return it.
    """
    # Compute a12 from constraint g(0) = 0
    # a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Compute orthonormal coefficients
    c0 = a0 * inv_sqrt_N0
    c4 = a4 * inv_sqrt_N4
    c8 = a8 * inv_sqrt_N8
    c12 = a12 * inv_sqrt_N12
    
    c0_sq = c0 * c0
    c4_sq = c4 * c4
    c8_sq = c8 * c8
    c12_sq = c12 * c12
    
    total_norm = c0_sq + c4_sq + c8_sq + c12_sq
    
    if total_norm < 1e-15:
        return float('inf')
    
    # <x^2> = (0.5*c0^2 + 4.5*c4^2 + 8.5*c8^2 + 12.5*c12^2) / total_norm
    x2 = (0.5 * c0_sq + 4.5 * c4_sq + 8.5 * c8_sq + 12.5 * c12_sq) / total_norm
    
    return x2


def _compute_gradient(a0, a4, a8):
    """
    Compute the gradient of the bound with respect to (a0, a4, a8).
    We use finite differences for simplicity, but analytic would be better.
    Let's try analytic.
    
    Let V = 0.5*c0^2 + 4.5*c4^2 + 8.5*c8^2 + 12.5*c12^2
    Let N = c0^2 + c4^2 + c8^2 + c12^2
    Bound = V/N
    
    d(Bound)/da_i = (dV/da_i * N - V * dN/da_i) / N^2
    
    c_k = a_k * inv_sqrt_N_k (for k=0,4,8)
    c12 = a12 * inv_sqrt_N12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0)/H12_0 * inv_sqrt_N12
    
    Let's define:
    k0 = inv_sqrt_N0
    k4 = inv_sqrt_N4
    k8 = inv_sqrt_N8
    k12 = inv_sqrt_N12
    
    c0 = a0 * k0
    c4 = a4 * k4
    c8 = a8 * k8
    a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0) / H12_0
    c12 = a12 * k12
    
    d c0 / d a0 = k0, d c0 / d a4 = 0, d c0 / d a8 = 0
    d c4 / d a0 = 0, d c4 / d a4 = k4, d c4 / d a8 = 0
    d c8 / d a0 = 0, d c8 / d a4 = 0, d c8 / d a8 = k8
    
    d a12 / d a0 = -H0_0 / H12_0
    d a12 / d a4 = -H4_0 / H12_0
    d a12 / d a8 = -H8_0 / H12_0
    
    d c12 / d a0 = k12 * (-H0_0 / H12_0)
    d c12 / d a4 = k12 * (-H4_0 / H12_0)
    d c12 / d a8 = k12 * (-H8_0 / H12_0)
    
    Let's denote:
    dc0_da0 = k0
    dc4_da4 = k4
    dc8_da8 = k8
    dc12_da0 = k12 * (-H0_0 / H12_0)
    dc12_da4 = k12 * (-H4_0 / H12_0)
    dc12_da8 = k12 * (-H8_0 / H12_0)
    
    dV/da0 = 2*0.5*c0*dc0_da0 + 2*12.5*c12*dc12_da0 = c0*k0 + 25*c12*dc12_da0
    dN/da0 = 2*c0*dc0_da0 + 2*c12*dc12_da0 = 2*c0*k0 + 2*c12*dc12_da0
    
    Similarly for a4 and a8.
    """
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    c0 = a0 * inv_sqrt_N0
    c4 = a4 * inv_sqrt_N4
    c8 = a8 * inv_sqrt_N8
    c12 = a12 * inv_sqrt_N12
    
    c0_sq = c0 * c0
    c4_sq = c4 * c4
    c8_sq = c8 * c8
    c12_sq = c12 * c12
    
    total_norm = c0_sq + c4_sq + c8_sq + c12_sq
    
    if total_norm < 1e-15:
        return np.array([0.0, 0.0, 0.0])
    
    V = 0.5 * c0_sq + 4.5 * c4_sq + 8.5 * c8_sq + 12.5 * c12_sq
    N = total_norm
    
    # Derivatives of c_k
    dc0_da0 = inv_sqrt_N0
    dc4_da4 = inv_sqrt_N4
    dc8_da8 = inv_sqrt_N8
    dc12_da0 = inv_sqrt_N12 * (-H0_0 / H12_0)
    dc12_da4 = inv_sqrt_N12 * (-H4_0 / H12_0)
    dc12_da8 = inv_sqrt_N12 * (-H8_0 / H12_0)
    
    # dV/da and dN/da
    dV_da0 = c0 * dc0_da0 + 25.0 * c12 * dc12_da0
    dN_da0 = 2.0 * c0 * dc0_da0 + 2.0 * c12 * dc12_da0
    
    dV_da4 = 9.0 * c4 * dc4_da4 + 25.0 * c12 * dc12_da4
    dN_da4 = 2.0 * c4 * dc4_da4 + 2.0 * c12 * dc12_da4
    
    dV_da8 = 17.0 * c8 * dc8_da8 + 25.0 * c12 * dc12_da8
    dN_da8 = 2.0 * c8 * dc8_da8 + 2.0 * c12 * dc12_da8
    
    # Gradient of V/N
    grad_a0 = (dV_da0 * N - V * dN_da0) / (N * N)
    grad_a4 = (dV_da4 * N - V * dN_da4) / (N * N)
    grad_a8 = (dV_da8 * N - V * dN_da8) / (N * N)
    
    return np.array([grad_a0, grad_a4, grad_a8])


def construct():
    """
    Returns three coefficients (a0, a4, a8) for the Hermite combination.
    """
    start_time = time.time()
    deadline = start_time + 180.0 - 30.0  # 30 second safety margin
    
    def objective(coeffs):
        if time.time() > deadline:
            return 1e10
        a0, a4, a8 = coeffs
        return _compute_bound(a0, a4, a8)
    
    def objective_grad(coeffs):
        if time.time() > deadline:
            return np.array([0.0, 0.0, 0.0])
        a0, a4, a8 = coeffs
        return _compute_gradient(a0, a4, a8)
    
    # Initial guess from the seed program
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Try multiple starting points
    start_points = [
        x0,
        np.array([-1.0, 0.1, 0.001]),
        np.array([-0.5, 0.05, 0.0005]),
        np.array([-1.5, 0.15, 0.0015]),
        np.array([-0.8, 0.08, 0.0008]),
        np.array([-1.2, 0.12, 0.0012]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.13, 0.04, 0.0003]),
        np.array([-1.13, 0.04, 0.0004]),
        np.array([-1.13, 0.04, 0.0002]),
    ]
    
    best_result = None
    best_obj = float('inf')
    
    for sp in start_points:
        if time.time() > deadline:
            break
        
        try:
            # Use L-BFGS-B with gradient
            result = minimize(objective, sp, jac=objective_grad, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # If we have time, try a high-precision polish pass on the best result
    if best_result is not None and time.time() < deadline - 10.0:
        try:
            result = minimize(objective, best_result, jac=objective_grad, method='L-BFGS-B',
                            options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-18})
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x
        except Exception:
            pass
    
    if best_result is None:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    return best_result
# EVOLVE-BLOCK-END
