"""Exact matrix approach for uncertainty product minimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200


def _build_exact_matrices():
    """
    Construct the exact overlap, position, and momentum matrices
    for the basis of Hermite functions with degrees 0, 4, 8, 12.
    
    Basis: B_n(x) = H_n(x) exp(-x^2/2)
    Norm: <B_n, B_n> = sqrt(pi) * 2^n * n!
    
    X^2 B_n = 1/4 B_{n+2} + (n + 1/2) B_n + n(n-1) B_{n-2}
    Since our degrees are 0, 4, 8, 12, the off-diagonal terms (n +/- 2)
    do not overlap with the basis vectors. Thus, the matrices are diagonal.
    
    M_0[k, k] = sqrt(pi) * 2^n_k * n_k!
    M_pos[k, k] = (n_k + 0.5) * M_0[k, k]
    M_mom[k, k] = (n_k + 0.5) * M_0[k, k]  (since signs don't affect diagonal)
    """
    degrees = [0, 4, 8, 12]
    N = len(degrees)
    
    M_0 = np.zeros((N, N))
    M_pos = np.zeros((N, N))
    M_mom = np.zeros((N, N))
    
    for k, n in enumerate(degrees):
        norm = np.sqrt(np.pi) * (2 ** n) * np.math.factorial(n)
        M_0[k, k] = norm
        # X^2 expectation value for eigenstate n is n + 0.5
        # Here we are in the non-orthonormal basis B_n, but the ratio is invariant.
        # <B_n, X^2 B_n> = (n + 0.5) <B_n, B_n>
        M_pos[k, k] = (n + 0.5) * norm
        M_mom[k, k] = (n + 0.5) * norm
        
    return M_0, M_pos, M_mom


def _bound_and_grad_exact(params, M_0, M_pos, M_mom):
    """
    Compute bound and gradient for the exact diagonal case.
    params = (a4, a8, a12)
    a0 is determined by g(0)=0: a0 = -(3*a4 + 105*a8 + 945*a12)
    Note: H_0(0)=1, H_4(0)=3, H_8(0)=105, H_12(0)=945.
    So a0*1 + a4*3 + a8*105 + a12*945 = 0 => a0 = -3*a4 - 105*a8 - 945*a12.
    
    Wait, the original code used a12 as the dependent variable.
    "The notebook derives the H12 coefficient to impose P(0)=0."
    "Three free coefficients multiply H0,H4,H8; the degree-12 weight is fixed"
    So the interface expects (a0, a4, a8) as the return value?
    "construct() -> three coefficients of the Hermite combination (degrees 0, 4, 8; the degree-12 weight is fixed so g(0) = 0)"
    
    The current code returns `np.array([a0, a4, a8])`.
    And inside, it calculates `a12 = -(a0 + 3*a4 + 105*a8)/945`.
    This is equivalent to $a_0 + 3 a_4 + 105 a_8 + 945 a_{12} = 0$.
    So I must keep the parameterization: inputs are $(a_0, a_4, a_8)$, and $a_{12}$ is derived.
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    # Since matrices are diagonal:
    # norm2 = sum(a_k^2 * M_0[k,k])
    # pos_var = sum(a_k^2 * M_pos[k,k])
    # mom_var = sum(a_k^2 * M_mom[k,k])
    
    M0_diag = np.diag(M_0)
    Mpos_diag = np.diag(M_pos)
    Mmom_diag = np.diag(M_mom)
    
    a2 = a**2
    norm2 = np.sum(a2 * M0_diag)
    pos_val = np.sum(a2 * Mpos_diag)
    mom_val = np.sum(a2 * Mmom_diag)
    
    if norm2 <= 1e-30 or not np.isfinite(norm2):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = pos_val / norm2
    vF = mom_val / norm2
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient calculation
    # bound = sqrt( (P/N) * (Q/N) ) = sqrt(PQ) / N
    # Let F = bound.
    # dF/da_i = (1/(2F)) * d(PQ/N^2)/da_i
    # P = sum a_k^2 p_k, Q = sum a_k^2 q_k, N = sum a_k^2 n_k
    # dP/da_i = 2 a_i p_i, etc.
    # d(PQ/N^2)/da_i = (dP Q + P dQ) N^2 - PQ 2 N dN / N^4
    # = (dP Q + P dQ)/N^2 - 2 PQ dN / N^3
    
    dN = 2 * a * M0_diag
    dP = 2 * a * Mpos_diag
    dQ = 2 * a * Mmom_diag
    
    # We need da/dparams.
    # a = [a0, a4, a8, a12]
    # params = [a0, a4, a8]
    # a12 = -(a0 + 3a4 + 105a8)/945
    # da0/dp0 = 1, da0/dp1 = 0, da0/dp2 = 0
    # da4/dp0 = 0, da4/dp1 = 1, da4/dp2 = 0
    # da8/dp0 = 0, da8/dp1 = 0, da8/dp2 = 1
    # da12/dp0 = -1/945, da12/dp1 = -3/945, da12/dp2 = -105/945
    
    # dN/dparams = sum_k (dN_k / da_k) * (da_k / dparams)
    # dN_k / da_k = 2 a_k M0_k
    
    dN_dp = np.zeros(3)
    dP_dp = np.zeros(3)
    dQ_dp = np.zeros(3)
    
    # k=0 (a0)
    dN_dp += (2 * a[0] * M0_diag[0]) * np.array([1, 0, 0])
    dP_dp += (2 * a[0] * Mpos_diag[0]) * np.array([1, 0, 0])
    dQ_dp += (2 * a[0] * Mmom_diag[0]) * np.array([1, 0, 0])
    
    # k=1 (a4)
    dN_dp += (2 * a[1] * M0_diag[1]) * np.array([0, 1, 0])
    dP_dp += (2 * a[1] * Mpos_diag[1]) * np.array([0, 1, 0])
    dQ_dp += (2 * a[1] * Mmom_diag[1]) * np.array([0, 1, 0])
    
    # k=2 (a8)
    dN_dp += (2 * a[2] * M0_diag[2]) * np.array([0, 0, 1])
    dP_dp += (2 * a[2] * Mpos_diag[2]) * np.array([0, 0, 1])
    dQ_dp += (2 * a[2] * Mmom_diag[2]) * np.array([0, 0, 1])
    
    # k=3 (a12)
    dN_dp += (2 * a[3] * M0_diag[3]) * np.array([-1/945.0, -3/945.0, -105/945.0])
    dP_dp += (2 * a[3] * Mpos_diag[3]) * np.array([-1/945.0, -3/945.0, -105/945.0])
    dQ_dp += (2 * a[3] * Mmom_diag[3]) * np.array([-1/945.0, -3/945.0, -105/945.0])
    
    N = norm2
    P = pos_val
    Q = mom_val
    
    dPQN2 = (dP_dp * Q + P * dQ_dp) / (N * N) - 2.0 * P * Q * dN_dp / (N * N * N)
    
    grad = dPQN2 / (2.0 * bound)
    
    if not np.all(np.isfinite(grad)):
        grad = np.array([1e18, 1e18, 1e18])
        
    return bound, grad


def construct():
    try:
        M_0, M_pos, M_mom = _build_exact_matrices()
    except Exception:
        return np.array([-1.13, 0.04, 0.000308642])

    def _obj_grad(params):
        val, grad = _bound_and_grad_exact(params, M_0, M_pos, M_mom)
        return val, grad

    # Starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
    ]
    
    best_val = np.inf
    best_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
        try:
            res = minimize(_obj_grad, start, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return np.array(best_params)
# EVOLVE-BLOCK-END
