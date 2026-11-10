# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize

_DEADLINE = time.time() + 615

def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    cos_theta = x_gl
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    dphi = 2.0 * np.pi / 130
    weights_1d = w_gl * dphi
    return cos_theta, phi, weights_1d

def _compute_harmonics_matrix(cos_theta, phi):
    """
    Compute the matrix X where X[i, j] = Y_j(ct_i, phi_j).
    j ranges from 0 to 20, corresponding to m = -10 to 10.
    """
    N_pts_ct = len(cos_theta)
    N_pts_phi = len(phi)
    X = np.zeros((N_pts_ct * N_pts_phi, 21))
    
    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    ct_flat = ct.flatten()
    ph_flat = ph.flatten()
    
    for k, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) + 
                       gammaln(10 - abs_m + 1) - gammaln(10 + abs_m + 1))
        N_m = np.exp(log_N)
        
        P = lpmv(abs_m, 10, ct_flat)
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * ph_flat)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(abs_m * ph_flat)
            
        X[:, k] = Y
        
    return X

def _precompute_tensors(X, w_flat):
    """Precompute Q (2x2) and T (4x4x4x4) tensors for spectral L2/L4 norm computation."""
    Q = X.T @ (w_flat[:, None] * X)
    # T_ijkl = sum_p w_p X_pi X_pj X_pk X_pl
    T = np.einsum('pi,pj,pk,pl,p->ijkl', X, X, X, X, w_flat, optimize=True)
    return Q, T

def _objective_and_grad(c, Q, T):
    """
    Minimize S4(c) = c^T T c (quartic) subject to c^T Q c = 1.
    Returns S4 and gradient of S4.
    """
    S2 = c @ Q @ c
    if S2 < 1e-10:
        return 1e10, np.zeros(21)
        
    S4 = np.einsum('ijkl,i,j,k,l->', T, c, c, c, c, optimize=True)
    
    # Gradient of S4: 4 * T_ijkl c_j c_k c_l
    grad_S4 = 4.0 * np.einsum('ijkl,j,k,l->i', T, c, c, c, optimize=True)
    
    return S4, grad_S4

def _constrained_minimize(start, Q, T, maxiter=1000):
    """
    Minimize S4 subject to c^T Q c = 1 using SLSQP.
    """
    def obj(c):
        val, _ = _objective_and_grad(c, Q, T)
        return val
    
    def grad(c):
        _, g = _objective_and_grad(c, Q, T)
        return g
        
    def constraint_fun(c):
        return c @ Q @ c - 1.0
        
    def constraint_jac(c):
        return 2.0 * (Q @ c)
        
    cons = ({'type': 'eq', 'fun': constraint_fun, 'jac': constraint_jac})
    
    # Initial scaling to satisfy constraint roughly
    norm_sq = start @ Q @ start
    if norm_sq > 1e-10:
        start = start / np.sqrt(norm_sq)
    else:
        start = np.random.default_rng(0).standard_normal(21)
        start = start / np.sqrt(start @ Q @ start)
        
    try:
        res = minimize(
            obj,
            start,
            jac=grad,
            method='SLSQP',
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-14}
        )
        return res.x, res.fun
    except Exception:
        return start, np.inf

def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE
    
    cos_theta, phi, w_1d = _build_quadrature()
    X = _compute_harmonics_matrix(cos_theta, phi)
    w_flat = np.tile(w_1d, 130)
    
    Q, T = _precompute_tensors(X, w_flat)
    
    best_coeffs = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    starts = []
    
    # Structured starts: Pure m
    for m in range(-10, 11):
        c = np.zeros(21)
        c[m + 10] = 1.0
        starts.append(c)
        
    # Structured starts: Pairs with symmetry
    for m1 in range(-10, 11):
        for m2 in range(m1 + 1, 11):
            if (m1 + m2) % 2 == 0:
                for w in [0.5, 0.7]:
                    c = np.zeros(21)
                    c[m1 + 10] = w
                    c[m2 + 10] = np.sqrt(1 - w*w)
                    starts.append(c.copy())
                    c[m2 + 10] = -np.sqrt(1 - w*w)
                    starts.append(c.copy())
                    
    # Random starts
    for _ in range(500):
        c = rng.standard_normal(21)
        starts.append(c)
        
    # Sparse random starts
    for _ in range(200):
        c = np.zeros(21)
        n_nonzero = rng.integers(2, 8)
        indices = rng.choice(21, size=n_nonzero, replace=False)
        c[indices] = rng.standard_normal(n_nonzero)
        starts.append(c)
        
    # Run minimization
    for start in starts:
        if time.time() > _DEADLINE - 10:
            break
            
        c_res, val = _constrained_minimize(start, Q, T, maxiter=500)
        
        # Validate and update best
        if np.isfinite(val) and val < best_val:
            # Re-evaluate to ensure accuracy
            val_check, _ = _objective_and_grad(c_res, Q, T)
            if val_check < best_val:
                best_val = val_check
                best_coeffs = c_res.copy()
                
    # Refine best result with higher precision
    if best_coeffs is not None and time.time() < _DEADLINE - 5:
        c_res, val = _constrained_minimize(best_coeffs, Q, T, maxiter=2000)
        if np.isfinite(val) and val < best_val:
            val_check, _ = _objective_and_grad(c_res, Q, T)
            if val_check < best_val:
                best_val = val_check
                best_coeffs = c_res.copy()
                
    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    # Normalize to satisfy constraints strictly for the return value
    # The problem statement says "returning 21 real coefficients c".
    # It does not require ||c||=1, but the ratio is scale invariant.
    # However, the validity check says max|c| < 1e-3 or > 1e5 is invalid.
    # Our c is normalized to c^T Q c = 1. Since Q ~ I, ||c|| ~ 1.
    # So max|c| should be around 1/sqrt(21) ~ 0.2, which is safe.
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (0.01 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e4 / max_abs)
        
    return best_coeffs

# EVOLVE-BLOCK-END
