"""Improved program: Exact generalized eigenvalue solve with high-precision polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 175.0  # Safety margin before 210s kill

    # Hermite polynomial values at 0 (physicists' convention)
    h0_0 = 1.0
    h4_0 = 12.0
    h8_0 = 1680.0
    h12_0 = 1039500.0

    # Variance (effective n) for each Hermite mode: <x^2> = n + 1/2
    n0 = 0.5
    n4 = 4.5
    n8 = 8.5
    n12 = 12.5

    # Coefficients for the constraint c12 = -k * (c0*h0_0 + c4*h4_0 + c8*h8_0)
    k = 1.0 / h12_0
    a = np.array([h0_0, h4_0, h8_0])

    # Matrix A for the numerator of the variance
    D = np.diag([n0, n4, n8])
    A = D + n12 * (k**2) * np.outer(a, a)

    # Matrix B for the denominator of the variance
    I = np.eye(3)
    B = I + (k**2) * np.outer(a, a)

    def rayleigh_quotient(c):
        c = np.asarray(c, dtype=float)
        num = c @ A @ c
        den = c @ B @ c
        if den <= 0:
            return 1e10
        return num / den

    def rayleigh_grad(c):
        c = np.asarray(c, dtype=float)
        Ac = A @ c
        Bc = B @ c
        num = c @ Ac
        den = c @ Bc
        if den <= 0:
            return np.zeros_like(c)
        grad = (2.0 * (Ac * den - Bc * num)) / (den * den)
        return grad

    def direct_objective(coeffs):
        """Direct formula for the uncertainty bound (var_x^2)"""
        c0, c4, c8 = coeffs
        c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
        
        w0 = c0**2
        w4 = c4**2
        w8 = c8**2
        w12 = c12**2
        
        total_w = w0 + w4 + w8 + w12
        if total_w < 1e-15:
            return 1e15
            
        var_x = (w0 * n0 + w4 * n4 + w8 * n8 + w12 * n12) / total_w
        return var_x * var_x

    # Solve the generalized eigenvalue problem A v = lambda B v
    best_val = np.inf
    best_vec = None

    try:
        # Use higher precision for the eigensolve
        eigvals, eigvecs = np.linalg.eigh(A, B)
        idx_min = np.argmin(eigvals)
        best_val = eigvals[idx_min]
        best_vec = eigvecs[:, idx_min].copy()
    except Exception:
        pass

    if best_vec is None:
        # Fallback to seed
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])

    # Verify with direct objective
    direct_val = direct_objective(best_vec)
    if direct_val < best_val:
        best_val = direct_val

    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5.0:
        try:
            from scipy.optimize import minimize
            
            # Stage 1: L-BFGS-B with the eigenvector as starting point
            starts = [
                best_vec,
                -best_vec,
                best_vec * 2.0,
                best_vec * 0.5,
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ]
            
            for s in starts:
                if time.time() - start_time > DEADLINE - 10.0:
                    break
                try:
                    res = minimize(
                        rayleigh_quotient,
                        s,
                        jac=rayleigh_grad,
                        method='L-BFGS-B',
                        options={'gtol': 1e-15, 'ftol': 1e-15, 'maxiter': 5000}
                    )
                    if res.fun < best_val:
                        best_val = res.fun
                        best_vec = res.x.copy()
                except Exception:
                    pass

            # Stage 2: Nelder-Mead polish from the best point
            if time.time() - start_time < DEADLINE - 5.0:
                try:
                    res_nm = minimize(
                        rayleigh_quotient,
                        best_vec,
                        method='Nelder-Mead',
                        options={'xatol': 1e-15, 'fatol': 1e-15, 'maxiter': 10000, 'adaptive': True}
                    )
                    if res_nm.fun < best_val:
                        best_val = res_nm.fun
                        best_vec = res_nm.x.copy()
                except Exception:
                    pass

            # Stage 3: Final ultra-precise polish using direct objective
            if time.time() - start_time < DEADLINE - 3.0:
                try:
                    res_final = minimize(
                        direct_objective,
                        best_vec,
                        method='Nelder-Mead',
                        options={'xatol': 1e-16, 'fatol': 1e-16, 'maxiter': 20000, 'adaptive': True}
                    )
                    if res_final.fun < best_val:
                        best_val = res_final.fun
                        best_vec = res_final.x.copy()
                except Exception:
                    pass

        except Exception:
            pass

    return np.array(best_vec)
# EVOLVE-BLOCK-END
