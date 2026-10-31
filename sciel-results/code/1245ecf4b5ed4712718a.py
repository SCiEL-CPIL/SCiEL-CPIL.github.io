"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import factorial, sqrt, pi
from scipy.optimize import fsolve


def _factorial(n):
    return factorial(n)


def construct():
    start_time = time.time()
    deadline = start_time + 180.0

    try:
        # Hermite-Gaussian test function: g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
        # Constraint: g(0) = 0 => c0 + 12*c4 + 1680*c8 + 665280*c12 = 0
        # So c12 = -(c0 + 12*c4 + 1680*c8) / 665280
        #
        # The uncertainty product bound is:
        # (∫ x^2 |g|^2 dx) * (∫ |g'|^2 dx) / (∫ |g|^2 dx)^2
        #
        # Using physicists' Hermite polynomials with weight e^{-x^2}:
        # ∫ H_m H_n e^{-x^2} dx = sqrt(pi) * 2^n * n! * delta_mn
        #
        # Let p(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        # Then:
        # ∫ |g|^2 dx = c^T M c
        # ∫ x^2 |g|^2 dx = c^T A c
        # ∫ |g'|^2 dx = c^T B c
        #
        # Using d/dx H_n = 2n H_{n-1} and x H_n = (1/2)H_{n+1} + n H_{n-1}:
        # p'(x) - x p(x) = sum c_k (k H_{k-1} - (1/2)H_{k+1})
        #
        # B is diagonal with:
        # B_{00} = sqrt(pi)/2
        # B_{ii} = sqrt(pi) * 2^{i-1} * (i-1)! * i * (2i + 1) for i >= 1

        degrees = [0, 4, 8, 12]
        n_vals = len(degrees)

        # Moment matrix M (diagonal)
        M = np.zeros((n_vals, n_vals))
        for i, deg in enumerate(degrees):
            M[i, i] = sqrt(pi) * (2 ** deg) * _factorial(deg)

        # Position matrix A
        # x^2 H_n = (1/4) H_{n+2} + n H_n + (n/2) H_{n-2}
        A = np.zeros((n_vals, n_vals))
        for i, deg_i in enumerate(degrees):
            for j, deg_j in enumerate(degrees):
                val = 0.0
                if deg_i + 2 == deg_j:
                    val += 0.25 * sqrt(pi) * (2 ** deg_j) * _factorial(deg_j)
                if deg_i == deg_j:
                    val += deg_i * sqrt(pi) * (2 ** deg_i) * _factorial(deg_i)
                if deg_i >= 2 and deg_i - 2 == deg_j:
                    val += (deg_i / 2.0) * sqrt(pi) * (2 ** deg_j) * _factorial(deg_j)
                A[i, j] = val

        # Momentum matrix B (diagonal)
        B = np.zeros((n_vals, n_vals))
        for i, deg in enumerate(degrees):
            if deg == 0:
                B[i, i] = sqrt(pi) / 2.0
            else:
                B[i, i] = sqrt(pi) * (2 ** (deg - 1)) * _factorial(deg - 1) * deg * (2 * deg + 1)

        # Constraint: c12 = -(c0 + 12*c4 + 1680*c8) / 665280
        # Parameterize with c = [c0, c4, c8, c12] where c12 is determined.
        # Let's use c0 = 1 (scale invariance of the Rayleigh quotient).
        # c = [1, a, b, -(1 + 12a + 1680b)/665280]

        c_base = np.array([1.0, 0.0, 0.0, -1.0 / 665280.0])
        v_a = np.array([0.0, 1.0, 0.0, -12.0 / 665280.0])
        v_b = np.array([0.0, 0.0, 1.0, -1680.0 / 665280.0])

        # Precompute quadratic form coefficients
        M0 = c_base @ M @ c_base
        A0 = c_base @ A @ c_base
        B0 = c_base @ B @ c_base

        Ma = c_base @ M @ v_a
        Aa = c_base @ A @ v_a
        Ba = c_base @ B @ v_a

        Mb = c_base @ M @ v_b
        Ab = c_base @ A @ v_b
        Bb = c_base @ B @ v_b

        Maa = v_a @ M @ v_a
        Aaa = v_a @ A @ v_a
        Baa = v_a @ B @ v_a

        Mbb = v_b @ M @ v_b
        Abb = v_b @ A @ v_b
        Bbb = v_b @ B @ v_b

        Mab = v_a @ M @ v_b
        Aab = v_a @ A @ v_b
        Bab = v_a @ B @ v_b

        def objective(params):
            a, b = params
            M_c = M0 + 2 * a * Ma + 2 * b * Mb + a * a * Maa + 2 * a * b * Mab + b * b * Mbb
            A_c = A0 + 2 * a * Aa + 2 * b * Ab + a * a * Aaa + 2 * a * b * Aab + b * b * Abb
            B_c = B0 + 2 * a * Ba + 2 * b * Bb + a * a * Baa + 2 * a * b * Bab + b * b * Bbb
            if M_c <= 0:
                return 1e10
            return (A_c * B_c) / (M_c * M_c)

        def gradient(params):
            a, b = params
            M_c = M0 + 2 * a * Ma + 2 * b * Mb + a * a * Maa + 2 * a * b * Mab + b * b * Mbb
            A_c = A0 + 2 * a * Aa + 2 * b * Ab + a * a * Aaa + 2 * a * b * Aab + b * b * Abb
            B_c = B0 + 2 * a * Ba + 2 * b * Bb + a * a * Baa + 2 * a * b * Bab + b * b * Bbb
            if M_c <= 0:
                return np.array([1e10, 1e10])
            
            # d/dx of (A_c * B_c) / (M_c^2)
            # = (A'_c * B_c * M_c^2 - A_c * B_c * 2 * M_c * M'_c) / M_c^4
            # = (A'_c * B_c * M_c - 2 * A_c * B_c * M'_c) / M_c^3
            
            dA_da = 2 * Aa + 2 * a * Aaa + 2 * b * Aab
            dA_db = 2 * Ab + 2 * b * Abb + 2 * a * Aab
            dB_da = 2 * Ba + 2 * a * Baa + 2 * b * Bab
            dB_db = 2 * Bb + 2 * b * Bbb + 2 * a * Bab
            dM_da = 2 * Ma + 2 * a * Maa + 2 * b * Mab
            dM_db = 2 * Mb + 2 * b * Mbb + 2 * a * Mab
            
            grad_a = (dA_da * B_c * M_c - 2 * A_c * B_c * dM_da) / (M_c ** 3)
            grad_b = (dA_db * B_c * M_c - 2 * A_c * B_c * dM_db) / (M_c ** 3)
            
            return np.array([grad_a, grad_b])

        # Use fsolve to find stationary points (gradient = 0)
        # This directly targets the exact critical points of the Rayleigh quotient
        best_score = float('inf')
        best_result = None

        # Try multiple starting points for fsolve
        start_points = []
        for a in np.linspace(-0.1, 0.1, 7):
            for b in np.linspace(-0.01, 0.01, 7):
                start_points.append(np.array([a, b]))

        # Add focused points around the seed solution
        seed_a = 1.0 / 25.0
        seed_b = 1.0 / 3240.0
        for da in np.linspace(-0.02, 0.02, 5):
            for db in np.linspace(-0.001, 0.001, 5):
                start_points.append(np.array([seed_a + da, seed_b + db]))

        for sp in start_points:
            if time.time() > deadline - 30:
                break
            try:
                solution = fsolve(gradient, sp, full_output=True)
                x_sol, info, ier, msg = solution
                if ier == 1:  # Converged
                    score = objective(x_sol)
                    if score < best_score:
                        best_score = score
                        best_result = x_sol
            except Exception:
                pass

        # If fsolve didn't find anything good, try minimize as fallback
        if best_result is None or best_score > 0.75:
            from scipy.optimize import minimize
            for sp in start_points[:20]:
                if time.time() > deadline - 30:
                    break
                try:
                    result = minimize(objective, sp, method='Nelder-Mead',
                                      options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16})
                    if result.success and result.fun < best_score:
                        best_score = result.fun
                        best_result = result.x
                except Exception:
                    pass

        if best_result is None:
            return np.array([-113 / 100, 1 / 25, 1 / 3240])

        a_opt, b_opt = best_result
        return np.array([1.0, a_opt, b_opt])

    except Exception:
        return np.array([-113 / 100, 1 / 25, 1 / 3240])
# EVOLVE-BLOCK-END
