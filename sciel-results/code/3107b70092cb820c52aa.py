"""Chebyshev collocation approach to minimize uncertainty product upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 170

def _chebyshev_matrices(N, L):
    """
    Compute Chebyshev collocation points, weights, and derivative matrices on [-L, L].
    N is the number of intervals, so there are N+1 points.
    """
    j = np.arange(N + 1)
    t = np.cos(np.pi * j / N)
    t = t[::-1]  # Ascending order from -1 to 1
    x = L * t
    
    # Weights for Chebyshev-Gauss-Lobatto quadrature
    w = np.ones(N + 1) * np.pi / N
    w[0] *= 0.5
    w[-1] *= 0.5
    w = L * w
    
    # Derivative matrix D on [-1, 1]
    c = np.ones(N + 1)
    c[0] = 0.5
    c[-1] = 0.5
    c = c * (-1.0) ** np.arange(N + 1)
    
    X = np.tile(t, (N + 1, 1))
    dX = X - X.T
    dX[np.abs(dX) < 1e-15] = 1.0
    
    # Off-diagonal entries
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    # Diagonal entries
    D = D - np.diag(D.sum(axis=1))
    
    # Scale to [-L, L]
    D1 = D / L
    D2 = D1 @ D1
    
    return x, w, D1, D2

def _compute_matrices(N, L):
    """
    Compute the Gram matrices M0, Mpos, Mmom for the basis functions.
    Basis: phi_n(x) = H_n(x) exp(-x^2/2) for n in {0, 4, 8, 12}.
    """
    x, w, D1, D2 = _chebyshev_matrices(N, L)
    
    # Hermite Polynomials
    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    def H12(x): return x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 13860*x**4 - 7560*x**2 + 945
    
    exp_factor = np.exp(-x**2 / 2.0)
    phi0 = H0(x) * exp_factor
    phi4 = H4(x) * exp_factor
    phi8 = H8(x) * exp_factor
    phi12 = H12(x) * exp_factor
    
    phis = np.array([phi0, phi4, phi8, phi12])
    
    # M0[i,j] = <phi_i, phi_j>
    M0 = np.zeros((4, 4))
    # Mpos[i,j] = <phi_i, x^2 phi_j>
    Mpos = np.zeros((4, 4))
    # Mmom[i,j] = <phi_i, -D2 phi_j>  (since p^2 = -d^2/dx^2)
    Mmom = np.zeros((4, 4))
    
    x2 = x**2
    
    for i in range(4):
        for j in range(4):
            M0[i, j] = np.sum(w * phis[i] * phis[j])
            Mpos[i, j] = np.sum(w * x2 * phis[i] * phis[j])
            # Compute -D2 phi_j
            d2_phi_j = D2 @ phis[j]
            Mmom[i, j] = -np.sum(w * phis[i] * d2_phi_j)
            
    return M0, Mpos, Mmom

def _objective_and_grad(params, M0, Mpos, Mmom):
    """
    Compute the objective value and gradient for parameters (a0, a4, a8).
    a12 is determined by the constraint g(0)=0 => a0 + 3*a4 + 105*a8 + 945*a12 = 0.
    So a12 = -(a0 + 3*a4 + 105*a8) / 945.
    """
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    N0 = a @ M0 @ a
    P = a @ Mpos @ a
    Q = a @ Mmom @ a
    
    if N0 <= 1e-30 or P <= 1e-30 or Q <= 1e-30 or not np.isfinite(N0) or not np.isfinite(P) or not np.isfinite(Q):
        return 1e10, np.zeros(3)
        
    val = np.sqrt(P * Q / (N0 * N0))
    
    # Gradient calculation
    # val^2 = P Q / N0^2
    # d(val^2)/da = (dP Q + P dQ) / N0^2 - 2 P Q dN0 / N0^3
    # d(val)/da = d(val^2)/da / (2 val)
    
    # Chain rule for a12
    # da/dparams is 4x3
    da_dparams = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/945.0, -3.0/945.0, -105.0/945.0]
    ])
    
    M0a = M0 @ a
    Mposa = Mpos @ a
    Mmoma = Mmom @ a
    
    dN0 = 2.0 * (da_dparams @ M0a)
    dP = 2.0 * (da_dparams @ Mposa)
    dQ = 2.0 * (da_dparams @ Mmoma)
    
    d_val2 = (dP * Q + P * dQ) / (N0 * N0) - 2.0 * P * Q * dN0 / (N0 * N0 * N0)
    grad = d_val2 / (2.0 * val)
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros(3)
        
    return val, grad

def construct():
    N = 500
    L = 10.0
    
    try:
        M0, Mpos, Mmom = _compute_matrices(N, L)
    except Exception:
        return np.array([-1.13, 0.04, 0.000308642])
        
    # Multi-start optimization
    rng = np.random.default_rng(12345)
    best_val = np.inf
    best_params = None
    
    # Generate diverse starting points
    starts = []
    # Seed solution
    starts.append(np.array([-1.13, 0.04, 0.000308642]))
    # Random points in a reasonable box
    for _ in range(20):
        a0 = rng.uniform(-3, 3)
        a4 = rng.uniform(-0.5, 0.5)
        a8 = rng.uniform(-0.01, 0.01)
        starts.append(np.array([a0, a4, a8]))
        
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
            
        try:
            # Use SLSQP as directed, with analytic gradient
            res = minimize(
                lambda p: _objective_and_grad(p, M0, Mpos, Mmom)[0],
                start,
                jac=lambda p: _objective_and_grad(p, M0, Mpos, Mmom)[1],
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
            
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    # Final polish with tighter tolerance
    try:
        res = minimize(
            lambda p: _objective_and_grad(p, M0, Mpos, Mmom)[0],
            best_params,
            jac=lambda p: _objective_and_grad(p, M0, Mpos, Mmom)[1],
            method='SLSQP',
            options={'maxiter': 5000, 'ftol': 1e-15}
        )
        if res.fun < best_val and np.isfinite(res.fun):
            best_val = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
        
    return np.array(best_params)
# EVOLVE-BLOCK-END
