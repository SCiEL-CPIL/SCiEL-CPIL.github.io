"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def _hermite_poly(n, x):
    """Compute physicist's Hermite polynomial Hn(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def construct():
    start_time = time.time()
    deadline = 180  # seconds, keep margin for evaluator kill at 210s

    # Indices for the basis functions H0, H4, H8, H12
    idxs = [0, 4, 8, 12]
    n_basis = len(idxs)
    
    # Precompute factorials
    fact = [math.factorial(k) for k in idxs]
    sqrt_pi = math.sqrt(math.pi)
    
    # Build the mass matrix M (norm squared) and the position moment matrix A
    # M_ij = delta_ij * i! * sqrt(pi)
    # A_ij = <x^2 H_i H_j e^-x^2>
    M = np.zeros((n_basis, n_basis))
    A = np.zeros((n_basis, n_basis))
    
    for i_idx, i in enumerate(idxs):
        for j_idx, j in enumerate(idxs):
            # Norm term
            if i == j:
                M[i_idx, j_idx] = fact[i_idx] * sqrt_pi
            
            # x^2 moment term
            # K_ij = sqrt(pi) * [ 1/4 (i+2)! delta_{j, i+2} + (i+1/2) i! delta_{j, i} + 1/4 i(i-1)(i-2)! delta_{j, i-2} ]
            val = 0.0
            if j == i + 2:
                # i+2 must be in idxs. Since idxs are 0,4,8,12, i+2 is never in idxs for even i.
                # Wait. 0->2 (no), 4->6 (no), 8->10 (no), 12->14 (no).
                # So the off-diagonal terms connecting i and i+2 are ZERO because 2 is not in the basis.
                # The basis is {0, 4, 8, 12}. The step is 4.
                # The operator x^2 connects H_i to H_{i-2}, H_i, H_{i+2}.
                # Since the basis only contains multiples of 4, x^2 will NOT couple H_0 to H_4, etc.
                # It will only couple H_i to H_i (diagonal) and potentially H_{i-4} or H_{i+4} if the operator allowed step 4.
                # But x^2 is step 2.
                # Therefore, A is DIAGONAL!
                pass
            elif j == i:
                val = (i + 0.5) * fact[i_idx] * sqrt_pi
            elif j == i - 2:
                # i-2 is not in idxs.
                pass
            
            A[i_idx, j_idx] = val

    # Verify: Is A diagonal?
    # Basis: 0, 4, 8, 12.
    # x^2 H_0 = 1/2 H_0 + 1/4 H_2. H_2 not in basis. So <x^2 H_0 H_0> only.
    # x^2 H_4 = ... H_2, H_4, H_6. Only H_4 in basis.
    # So yes, A is diagonal.
    
    # Let's double check the constraint c_12.
    # g(0) = c_0 H_0(0) + c_4 H_4(0) + c_8 H_8(0) + c_12 H_12(0) = 0
    # H_0(0) = 1
    # H_4(0) = -12
    # H_8(0) = 1680
    # H_12(0) = -40320
    # c_0 - 12 c_4 + 1680 c_8 - 40320 c_12 = 0
    # c_12 = (c_0 - 12 c_4 + 1680 c_8) / 40320

    def objective(coeffs):
        c0, c4, c8 = coeffs
        c12 = (c0 - 12.0 * c4 + 1680.0 * c8) / 40320.0
        c = np.array([c0, c4, c8, c12])
        
        norm2 = c @ M @ c
        if norm2 <= 0 or not np.isfinite(norm2):
            return np.inf
            
        x2_norm = c @ A @ c
        if not np.isfinite(x2_norm) or x2_norm < 0:
            return np.inf
            
        # Delta x * Delta xi = (x2_norm / norm2)
        return x2_norm / norm2

    # Initial guess
    c0_init = -1.13
    c4_init = 0.04
    c8_init = 1.0 / 3240.0
    
    best_coeffs = np.array([c0_init, c4_init, c8_init])
    best_val = objective(best_coeffs)
    
    # Differential Evolution
    # Bounds for c0, c4, c8
    # c0 is around -1, c4 around 0.04, c8 around 0.0003
    # Let's set generous bounds
    bounds = [(-10, 10), (-1, 1), (-0.01, 0.01)]
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        try:
            de_res = differential_evolution(
                objective, bounds, 
                seed=42, 
                maxiter=1000, 
                tol=1e-12,
                polish=True,
                init='sobol',
                updating='immediate',
                workers=1
            )
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_coeffs = de_res.x
        except Exception:
            pass
    
    # Final Polish with L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            res = minimize(
                objective, best_coeffs, method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x
        except Exception:
            pass

    return best_coeffs
# EVOLVE-BLOCK-END
