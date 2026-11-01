"""
Analytic uncertainty bound with exact gradient + L-BFGS-B optimization.
Uses the eigenfunction property of the Fourier transform on Hermite functions
to compute frequency-space moments as closed-form algebraic functions of coefficients.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import factorial, pi, sqrt

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195  # Safety margin before 210s kill


def _precompute():
    """
    Precompute all constants needed for the analytic bound computation.
    
    The test function is g(x) = P(x) * exp(-x^2/2) where
    P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    
    Using the standard (physicist's) Hermite polynomials:
    H0(x) = 1
    H4(x) = x^4 - 6x^2 + 3
    H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    H12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    
    g(0) = 0 constraint: c0 + 3*c4 + 105*c8 + 945*c12 = 0
    => c12 = -(c0 + 3*c4 + 105*c8) / 945
    
    |g(x)|^2 = P(x)^2 * exp(-x^2)
    For the Fourier transform with convention F[f](k) = (1/sqrt(2*pi)) * int f(x) e^{-ikx} dx:
    F[H_n(x) e^{-x^2/2}] = (-i)^n H_n(x) e^{-x^2/2} (up to normalization)
    
    So |G(k)|^2 = (1/(2*pi)) * P(-ik)^2 * exp(-k^2) * (2*pi) = P(-ik)^2 * exp(-k^2) / (2*pi) * 2*pi
    Actually, let me be more careful.
    
    F[g](k) = (1/sqrt(2*pi)) * int P(x) e^{-x^2/2} e^{-ikx} dx
    
    Using the identity: int H_n(x) e^{-x^2/2} e^{-ikx} dx = sqrt(2*pi) * (-i)^n H_n(k) e^{-k^2/2}
    
    So F[g](k) = (1/sqrt(2*pi)) * sum c_n (-i)^n H_n(k) e^{-k^2/2} * sqrt(2*pi)
               = sum c_n (-i)^n H_n(k) e^{-k^2/2}
    
    Therefore |G(k)|^2 = |sum c_n (-i)^n H_n(k)|^2 * e^{-k^2}
    
    Since we're dealing with real coefficients and even polynomials (only even degrees),
    (-i)^n for n=0,4,8,12 gives: 1, 1, 1, 1 (since n is divisible by 4)
    
    Wait, (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1. Yes!
    So |G(k)|^2 = P(k)^2 * e^{-k^2}
    
    This means the position and frequency space integrals have the SAME form!
    A(f) * A(F(f)) = sqrt(M/N) * sqrt(M/N) = M/N
    
    where N = int P(x)^2 e^{-x^2} dx and M = int x^2 P(x)^2 e^{-x^2} dx
    """
    
    # I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n * n!) * sqrt(pi)
    # = (2n-1)!! * sqrt(pi) / 2^n
    # For n=0: sqrt(pi)
    # For n=1: sqrt(pi)/2
    # For n=2: 3*sqrt(pi)/4
    # For n=3: 15*sqrt(pi)/8
    # etc.
    
    sp = sqrt(pi)
    I = np.zeros(15)  # Need up to n=12 for x^{24}
    for n in range(15):
        # (2n)! / (4^n * n!) * sqrt(pi)
        if n == 0:
            I[n] = sp
        else:
            # Use recurrence: I[n] = (2n-1)/(2n) * I[n-1]
            I[n] = (2*n - 1) / (2*n) * I[n-1]
    
    # P(x) = sum_{k=0}^{6} p_k x^{2k}
    # p_k depends on c0, c4, c8 (and c12 which is determined)
    # 
    # H0: p0 += c0
    # H4: p0 += 3*c4, p1 += -6*c4, p2 += c4
    # H8: p0 += 105*c8, p1 += -420*c8, p2 += 210*c8, p3 += -28*c8, p4 += c8
    # H12: p0 += 945*c12, p1 += -7560*c12, p2 += 13860*c12, p3 += -6930*c12, 
    #       p4 += 1155*c12, p5 += -66*c12, p6 += c12
    
    # Coefficient matrix: rows are p_k, columns are [c0, c4, c8, c12]
    # p0 = 1*c0 + 3*c4 + 105*c8 + 945*c12
    # p1 = 0*c0 - 6*c4 - 420*c8 - 7560*c12
    # p2 = 0*c0 + 1*c4 + 210*c8 + 13860*c12
    # p3 = 0*c0 + 0*c4 - 28*c8 - 6930*c12
    # p4 = 0*c0 + 0*c4 + 1*c8 + 1155*c12
    # p5 = 0*c0 + 0*c4 + 0*c8 - 66*c12
    # p6 = 0*c0 + 0*c4 + 0*c8 + 1*c12
    
    coeff_matrix = np.array([
        [1.0, 3.0, 105.0, 945.0],
        [0.0, -6.0, -420.0, -7560.0],
        [0.0, 1.0, 210.0, 13860.0],
        [0.0, 0.0, -28.0, -6930.0],
        [0.0, 0.0, 1.0, 1155.0],
        [0.0, 0.0, 0.0, -66.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # c12 = -(c0 + 3*c4 + 105*c8) / 945
    # So [c0, c4, c8, c12] = [c0, c4, c8, -(c0 + 3*c4 + 105*c8)/945]
    # 
    # Let's express p in terms of c0, c4, c8 only:
    # c12 = -c0/945 - 3*c4/945 - 105*c8/945
    #      = -c0/945 - c4/315 - c8/9
    #
    # p0 = c0 + 3*c4 + 105*c8 + 945*(-c0/945 - c4/315 - c8/9)
    #    = c0 + 3*c4 + 105*c8 - c0 - 3*c4 - 105*c8 = 0  (as expected from g(0)=0)
    #
    # p1 = -6*c4 - 420*c8 - 7560*(-c0/945 - c4/315 - c8/9)
    #    = -6*c4 - 420*c8 + 7560*c0/945 + 7560*c4/315 + 7560*c8/9
    #    = -6*c4 - 420*c8 + 8*c0 + 24*c4 + 840*c8
    #    = 8*c0 + 18*c4 + 420*c8
    #
    # p2 = c4 + 210*c8 + 13860*(-c0/945 - c4/315 - c8/9)
    #    = c4 + 210*c8 - 13860*c0/945 - 13860*c4/315 - 13860*c8/9
    #    = c4 + 210*c8 - 14.6667*c0 - 44*c4 - 1540*c8
    #    = -14.6667*c0 - 43*c4 - 1330*c8
    #
    # Let me compute these exactly:
    # p1: 7560/945 = 8, 7560/315 = 24, 7560/9 = 840
    # p1 = 8*c0 + (-6+24)*c4 + (-420+840)*c8 = 8*c0 + 18*c4 + 420*c8
    #
    # p2: 13860/945 = 14.6667 = 44/3, 13860/315 = 44, 13860/9 = 1540
    # p2 = -44/3*c0 + (1-44)*c4 + (210-1540)*c8 = -44/3*c0 - 43*c4 - 1330*c8
    #
    # p3 = -28*c8 - 6930*(-c0/945 - c4/315 - c8/9)
    #    = -28*c8 + 6930*c0/945 + 6930*c4/315 + 6930*c8/9
    #    = -28*c8 + 7.3333*c0 + 22*c4 + 770*c8
    #    = 22/3*c0 + 22*c4 + 742*c8
    #
    # p4 = c8 + 1155*(-c0/945 - c4/315 - c8/9)
    #    = c8 - 1155*c0/945 - 1155*c4/315 - 1155*c8/9
    #    = c8 - 7/3*c0 - 11/3*c4 - 128.333*c8
    #    = -7/3*c0 - 11/3*c4 + (1-385/3)*c8
    #    = -7/3*c0 - 11/3*c4 - 382/3*c8
    #
    # p5 = -66*(-c0/945 - c4/315 - c8/9)
    #    = 66*c0/945 + 66*c4/315 + 66*c8/9
    #    = 2/31.5*c0 + 2/10.5*c4 + 22/3*c8
    #    = 2/31.5*c0 + 2/10.5*c4 + 22/3*c8
    #    Let me redo: 66/945 = 2/31.5 = 4/63, 66/315 = 2/10.5 = 4/21, 66/9 = 22/3
    #    p5 = 4/63*c0 + 4/21*c4 + 22/3*c8
    #
    # p6 = -c0/945 - c4/315 - c8/9
    
    # Let me just use the matrix approach with c12 expressed in terms of c0,c4,c8
    # c_vec = [c0, c4, c8, c12] where c12 = A@c0 + B@c4 + C@c8
    # A = -1/945, B = -1/315, C = -1/9
    
    # p = coeff_matrix @ [c0, c4, c8, c12]
    #   = coeff_matrix[:, :3] @ [c0, c4, c8] + coeff_matrix[:, 3] @ c12
    #   = coeff_matrix[:, :3] @ [c0, c4, c8] + coeff_matrix[:, 3] @ (A*c0 + B*c4 + C*c8)
    #   = (coeff_matrix[:, :3] + np.outer(coeff_matrix[:, 3], [A, B, C])) @ [c0, c4, c8]
    
    A_c12 = -1.0/945.0
    B_c12 = -1.0/315.0
    C_c12 = -1.0/9.0
    
    reduced_matrix = coeff_matrix[:, :3] + np.outer(coeff_matrix[:, 3], [A_c12, B_c12, C_c12])
    
    return I, reduced_matrix


_I, _REDUCED_MATRIX = _precompute()


def _compute_moments(params):
    """
    Compute N = int P(x)^2 e^{-x^2} dx and M = int x^2 P(x)^2 e^{-x^2} dx
    analytically.
    """
    c0, c4, c8 = params
    p = _REDUCED_MATRIX @ np.array([c0, c4, c8])
    
    # N = sum_{i,j=0}^{6} p_i * p_j * I[i+j]
    # M = sum_{i,j=0}^{6} p_i * p_j * I[i+j+1]
    
    N = 0.0
    M = 0.0
    for i in range(7):
        if p[i] == 0.0:
            continue
        for j in range(7):
            if p[j] == 0.0:
                continue
            pp = p[i] * p[j]
            N += pp * _I[i + j]
            M += pp * _I[i + j + 1]
    
    return N, M


def _bound_and_grad(params):
    """
    Compute the uncertainty product bound and its analytic gradient.
    bound = M/N where N = int P^2 e^{-x^2} dx, M = int x^2 P^2 e^{-x^2} dx
    """
    N, M = _compute_moments(params)
    
    if N <= 0 or not np.isfinite(N) or not np.isfinite(M):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = M / N
    
    # Gradient: d(M/N)/dc = (dM/dc * N - M * dN/dc) / N^2
    # dN/dc_k = 2 * sum_j p_j * (dp/dc_k)_j * I[i+j] summed over i
    # Actually: N = sum_{i,j} p_i p_j I_{i+j}
    # dN/dc_k = sum_{i,j} (dp_i/dc_k * p_j + p_i * dp_j/dc_k) I_{i+j}
    #          = 2 * sum_i dp_i/dc_k * sum_j p_j I_{i+j}
    
    p = _REDUCED_MATRIX @ np.array(params)
    dp = _REDUCED_MATRIX  # This is dp/dc_k for each k
    
    grad = np.zeros(3)
    for k in range(3):
        dN = 0.0
        dM = 0.0
        for i in range(7):
            dpi = dp[i, k]
            if dpi == 0.0:
                continue
            sum_j_N = 0.0
            sum_j_M = 0.0
            for j in range(7):
                if p[j] == 0.0:
                    continue
                sum_j_N += p[j] * _I[i + j]
                sum_j_M += p[j] * _I[i + j + 1]
            dN += 2.0 * dpi * sum_j_N
            dM += 2.0 * dpi * sum_j_M
        
        grad[k] = (dM * N - M * dN) / (N * N)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    """
    Return three coefficients (c0, c4, c8) that minimize the uncertainty bound.
    c12 is determined by the constraint g(0) = 0.
    """
    # Seed initial guess
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Multiple diverse starting points
    starts = [
        seed,
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
        np.array([-0.8, 0.03, 0.0003]),
        np.array([-1.2, 0.06, 0.0006]),
    ]
    
    # Add random starts
    rng = np.random.default_rng(42)
    for _ in range(30):
        starts.append(rng.normal(0, 1.0, 3))
    
    best_val = np.inf
    best_params = seed.copy()
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with Nelder-Mead from best
    if best_params is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Additional random restarts if time permits
    if time.time() < _DEADLINE - 10:
        for _ in range(20):
            if time.time() > _DEADLINE - 10:
                break
            x_rand = rng.normal(0, 0.5, 3)
            try:
                res = minimize(_objective_grad, x_rand, method='L-BFGS-B',
                              jac=True, options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    return best_params

# EVOLVE-BLOCK-END
