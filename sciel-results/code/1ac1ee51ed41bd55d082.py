# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, gammaln
import time

def _build_harmonics():
    """
    Build the 21 real orthonormal degree-10 spherical harmonics evaluated
    at 70 Gauss-Legendre nodes in cos(theta) x 130 equally-spaced phi points.
    """
    k = 10
    n_theta = 70
    n_phi = 130
    
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    phi_weight = 2 * np.pi / n_phi
    weights_2d = np.outer(w, np.full(n_phi, phi_weight))
    
    # Y shape: (n_theta, n_phi, 21)
    Y = np.zeros((n_theta, n_phi, 2 * k + 1), dtype=np.float64)
    
    for m in range(-k, k + 1):
        m_idx = m + k
        abs_m = abs(m)
        
        # Associated Legendre P_l^m(x) with Condon-Shortley phase
        P = lpmv(abs_m, k, x)
        
        # Normalization N_m
        # N_m = sqrt( (2l+1)/(4pi) * (l-m)!/(l+m)! )
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + gammaln(k - abs_m + 1) - gammaln(k + abs_m + 1))
        N = np.exp(log_N)
        
        if m == 0:
            Y[:, :, m_idx] = N * P[:, np.newaxis]
        elif m > 0:
            # Y_m = sqrt(2) N_m P^m cos(m phi)
            Y[:, :, m_idx] = np.sqrt(2) * N * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            # Y_m = sqrt(2) N_|m| P^|m| sin(|m| phi)
            Y[:, :, m_idx] = np.sqrt(2) * N * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return Y, weights_2d

def _objective_and_grad(c, Y, weights):
    """
    Minimize ||f||_4 subject to ||c||_2 = 1.
    Returns ||f||_4 and its gradient.
    """
    f = np.tensordot(c, Y, axes=([0], [2]))
    
    # Weights are in 'weights'
    f2 = f * f
    f4 = f2 * f2
    
    sum_f4 = np.sum(f4 * weights)
    
    if sum_f4 < 1e-30:
        return 1e10, np.zeros_like(c)
    
    norm4 = sum_f4 ** 0.25
    
    # Gradient of ||f||_4
    # d/dc_k (sum f^4 w)^{1/4} = 1/4 (sum f^4 w)^{-3/4} * sum 4 f^3 Y_k w
    # = (sum f^4 w)^{-3/4} * sum f^3 Y_k w
    # = norm4^{-3} * sum f^3 Y_k w
    
    f3_w = f * f2 * weights
    # sum over grid points for each harmonic component
    # Y is (n_theta, n_phi, 21)
    # f3_w is (n_theta, n_phi)
    grad_terms = np.tensordot(f3_w, Y, axes=([0, 1], [0, 1]))
    
    grad = (norm4 ** -3) * grad_terms
    
    return norm4, grad

def _constraint(c):
    """Constraint: ||c||_2^2 - 1 = 0"""
    return np.array([np.dot(c, c) - 1.0])

def _constraint_jac(c):
    """Jacobian of constraint"""
    return np.array([2.0 * c])

def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing ||f||_4/||f||_2 for degree-10 spherical harmonics.
    Equivalent to minimizing ||f||_4 subject to ||c||_2 = 1.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 630s kill
    
    Y, weights = _build_harmonics()
    
    best_coeffs = None
    best_val = np.inf
    
    # Define constraints for trust-constr
    cons = {'type': 'eq', 'fun': _constraint, 'jac': _constraint_jac}
    
    # Initializations
    inits = []
    
    # 1. m=0 only (Zonal)
    c0 = np.zeros(n_coeffs)
    c0[k] = 1.0
    inits.append(c0)
    
    # 2. m=10 only (Highest order)
    c10 = np.zeros(n_coeffs)
    c10[k] = 1.0 # m=10 is index 20
    c10[20] = 1.0
    c10[k] = 0.0
    inits.append(c10)
    
    # 3. m=-10 only
    c_10 = np.zeros(n_coeffs)
    c_10[0] = 1.0
    inits.append(c_10)
    
    # 4. Symmetric m=0 and m=10
    c_sym = np.zeros(n_coeffs)
    c_sym[k] = 1.0 / np.sqrt(2)
    c_sym[20] = 1.0 / np.sqrt(2)
    inits.append(c_sym)
    
    # 5. All cos terms (m >= 0)
    c_cos = np.zeros(n_coeffs)
    for m in range(0, 11):
        c_cos[10 + m] = 1.0 / np.sqrt(11)
    inits.append(c_cos)
    
    # 6. All sin terms (m < 0)
    c_sin = np.zeros(n_coeffs)
    for m in range(-10, 0):
        c_sin[10 + m] = 1.0 / np.sqrt(10)
    inits.append(c_sin)
    
    # 7. Random inits
    np.random.seed(42)
    for _ in range(10):
        c_rand = np.random.randn(n_coeffs)
        c_rand /= np.linalg.norm(c_rand)
        inits.append(c_rand)
        
    # 8. Perturbations of m=0
    for _ in range(5):
        c_pert = np.zeros(n_coeffs)
        c_pert[k] = 1.0
        c_pert += 0.1 * np.random.randn(n_coeffs)
        c_pert /= np.linalg.norm(c_pert)
        inits.append(c_pert)

    for c_init in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        # Check initial value
        val_init, _ = _objective_and_grad(c_init, Y, weights)
        if val_init < best_val:
            best_val = val_init
            best_coeffs = c_init.copy()
        
        # Optimize
        try:
            # Use trust-constr to handle the equality constraint robustly
            result = minimize(
                lambda c: _objective_and_grad(c, Y, weights)[0],
                c_init,
                jac=lambda c: _objective_and_grad(c, Y, weights)[1],
                method='trust-constr',
                constraints=[cons],
                options={'maxiter': 1000, 'gtol': 1e-12, 'xtol': 1e-12, 'verbose': 0}
            )
            
            if result.fun < best_val and np.isfinite(result.fun):
                # Ensure constraint is satisfied
                c_res = result.x
                c_res /= np.linalg.norm(c_res)
                val_res, _ = _objective_and_grad(c_res, Y, weights)
                if val_res < best_val:
                    best_val = val_res
                    best_coeffs = c_res.copy()
        except Exception:
            pass

    # Polish phase: High precision run on the best candidate
    if best_coeffs is not None and time.time() - start_time < deadline - 10:
        try:
            result_polish = minimize(
                lambda c: _objective_and_grad(c, Y, weights)[0],
                best_coeffs,
                jac=lambda c: _objective_and_grad(c, Y, weights)[1],
                method='trust-constr',
                constraints=[cons],
                options={'maxiter': 2000, 'gtol': 1e-14, 'xtol': 1e-14, 'verbose': 0}
            )
            if result_polish.fun < best_val and np.isfinite(result_polish.fun):
                c_polish = result_polish.x
                c_polish /= np.linalg.norm(c_polish)
                val_polish, _ = _objective_and_grad(c_polish, Y, weights)
                if val_polish < best_val:
                    best_val = val_polish
                    best_coeffs = c_polish.copy()
        except Exception:
            pass

    if best_coeffs is None:
        best_coeffs = np.zeros(n_coeffs)
        best_coeffs[k] = 1.0

    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.zeros(n_coeffs)
        best_coeffs[k] = 1.0

    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = np.zeros(n_coeffs)
        best_coeffs[k] = 1.0
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5

    return best_coeffs

# EVOLVE-BLOCK-END
