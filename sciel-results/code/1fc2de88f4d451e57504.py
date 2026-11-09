# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import lpmv
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    k = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Leave 20s margin before 600s limit

    # Set up quadrature points
    # 70-point Gauss-Legendre in cos(theta)
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    
    # 130 equally spaced phi points
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Precompute sin/cos of m*phi for m = 0..10
    cos_m_phi = np.zeros((11, n_phi))
    sin_m_phi = np.zeros((11, n_phi))
    for m in range(11):
        if m == 0:
            cos_m_phi[m] = 1.0
            sin_m_phi[m] = 0.0
        else:
            cos_m_phi[m] = np.cos(m * phi)
            sin_m_phi[m] = np.sin(m * phi)
    
    # Precompute normalization constants N_m
    N = np.zeros(11)
    for m in range(11):
        N[m] = np.sqrt(21.0 / (4.0 * np.pi) * math.factorial(10 - m) / math.factorial(10 + m))
    
    # Precompute associated Legendre polynomials P_10^m(x) for x in x_gl
    P_legendre = np.zeros((11, 70))
    for m in range(11):
        P_legendre[m] = lpmv(m, 10, x_gl)
    
    # Build the full matrix Y_flat of shape (70*130, 21)
    # Indexing: row = i*n_phi + j, col = m+10
    N_q = 70 * n_phi
    Y_flat = np.zeros((N_q, 21), dtype=np.float64)
    
    # Weights for each quadrature point
    w_flat = np.zeros(N_q, dtype=np.float64)
    
    for i in range(70):
        row_start = i * n_phi
        row_end = (i + 1) * n_phi
        w_flat[row_start:row_end] = w_gl[i] * dphi
        
        # m = 0
        Y_flat[row_start:row_end, 10] = N[0] * P_legendre[0, i]
        
        # m > 0 and m < 0
        for m in range(1, 11):
            coeff_m_i = np.sqrt(2.0) * N[m] * P_legendre[m, i]
            # c_m (index 10+m) corresponds to cos(m*phi)
            Y_flat[row_start:row_end, 10 + m] = coeff_m_i * cos_m_phi[m]
            # c_-m (index 10-m) corresponds to sin(m*phi)
            Y_flat[row_start:row_end, 10 - m] = coeff_m_i * sin_m_phi[m]

    def objective_J(c_vec):
        """
        Compute J = S4 / S2^2 where S2 = sum(w f^2), S4 = sum(w f^4).
        This is proportional to R^4. Minimizing J is equivalent to minimizing R.
        """
        f = Y_flat @ c_vec
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        if S2 <= 1e-300:
            return 1e100
        return S4 / (S2 * S2)

    def jac_J(c_vec):
        """
        Gradient of J = S4 / S2^2.
        dJ/dc = (1/S2^2) dS4 - (2 S4 / S2^3) dS2
        dS2/dc = 2 Y^T (w f)
        dS4/dc = 4 Y^T (w f^3)
        """
        f = Y_flat @ c_vec
        f2 = f * f
        S2 = np.sum(w_flat * f2)
        S4 = np.sum(w_flat * f2 * f2)
        
        if S2 <= 1e-300:
            return np.zeros(21)
            
        wf = w_flat * f
        wf3 = w_flat * f * f2
        
        dS2 = 2.0 * (Y_flat.T @ wf)
        dS4 = 4.0 * (Y_flat.T @ wf3)
        
        # J = S4 * S2^-2
        # dJ = dS4 * S2^-2 + S4 * (-2) * S2^-3 * dS2
        grad = (dS4 * (S2 * S2) - 2.0 * S4 * dS2) / (S2 * S2 * S2)
        return grad

    # 11-dim amplitude optimization
    # u = [c_0, A_1, ..., A_10] where A_m = sqrt(c_m^2 + c_-m^2) >= 0
    # The objective J is phase-independent, so we can reduce to 11 dims.
    
    def u_to_c(u):
        c = np.zeros(21)
        c[10] = u[0]
        for m in range(1, 11):
            A = abs(u[m])
            c[10 + m] = A / np.sqrt(2.0)
            c[10 - m] = A / np.sqrt(2.0)
        return c

    def objective_u(u):
        c = u_to_c(u)
        return objective_J(c)

    def jac_u(u):
        c = u_to_c(u)
        grad_c = jac_J(c)
        grad_u = np.zeros(11)
        grad_u[0] = grad_c[10]
        for m in range(1, 11):
            A = abs(u[m])
            if A > 1e-15:
                sign = 1.0 if u[m] > 0 else -1.0
                grad_u[m] = (grad_c[10+m] + grad_c[10-m]) * sign / np.sqrt(2.0)
            else:
                grad_u[m] = 0.0
        return grad_u

    def norm_constraint_u(u):
        return u[0]**2 + np.sum(u[1:]**2) - 1.0

    def norm_jac_u(u):
        g = np.zeros(11)
        g[0] = 2.0 * u[0]
        g[1:] = 2.0 * u[1:]
        return g

    cons_u = [{'type': 'eq', 'fun': norm_constraint_u, 'jac': norm_jac_u}]
    cons_21 = [{'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2.0 * c}]

    best_result = None
    best_obj = np.inf

    # Starting points in 11-dim amplitude space
    starts_u = []
    
    # Structured starts
    starts_u.append(np.array([1.0] + [0.0]*10))  # c_0 only
    starts_u.append(np.array([0.0]*10 + [1.0]))  # A_10 only
    
    for m_target in range(1, 11):
        u = np.zeros(11)
        u[m_target] = 1.0
        starts_u.append(u)
    
    # Pairs
    for m1 in range(11):
        for m2 in range(m1+1, 11):
            u = np.zeros(11)
            u[m1] = 1.0 / np.sqrt(2)
            u[m2] = 1.0 / np.sqrt(2)
            starts_u.append(u)
    
    # Uniform
    starts_u.append(np.ones(11) / np.sqrt(11.0))
    
    # Low-m emphasis
    u_low = np.array([1.0/(m+1) for m in range(11)])
    starts_u.append(u_low / np.linalg.norm(u_low))
    
    # High-m emphasis
    u_high = np.array([(m+1)/11.0 for m in range(11)])
    starts_u.append(u_high / np.linalg.norm(u_high))
    
    # Random starts
    for seed in range(100):
        np.random.seed(seed)
        u_rand = np.random.randn(11)
        u_rand = u_rand / np.linalg.norm(u_rand)
        starts_u.append(u_rand)

    # Optimize in 11-dim space
    for i, u_start in enumerate(starts_u):
        if time.time() > deadline - 60.0:
            break
            
        try:
            result = minimize(
                objective_u,
                u_start,
                method='SLSQP',
                jac=jac_u,
                constraints=cons_u,
                options={
                    'maxiter': 150,
                    'ftol': 1e-12
                }
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = u_to_c(result.x)
                
        except Exception:
            pass
            
    # Full 21-dim polish from best 11-dim result
    if best_result is not None and time.time() < deadline - 30.0:
        try:
            result_21 = minimize(
                objective_J,
                best_result,
                method='SLSQP',
                jac=jac_J,
                constraints=cons_21,
                options={
                    'maxiter': 500,
                    'ftol': 1e-14
                }
            )
            if result_21.fun < best_obj:
                best_obj = result_21.fun
                best_result = result_21.x.copy()
        except Exception:
            pass
    
    # Additional random 21-dim starts
    for seed in range(40):
        if time.time() > deadline - 15.0:
            break
        np.random.seed(5000 + seed)
        c_rand = np.random.randn(21)
        c_rand = c_rand / np.linalg.norm(c_rand)
        
        try:
            result = minimize(
                objective_J,
                c_rand,
                method='SLSQP',
                jac=jac_J,
                constraints=cons_21,
                options={
                    'maxiter': 150,
                    'ftol': 1e-12
                }
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result.x.copy()
                
        except Exception:
            pass

    if best_result is None:
        best_result = np.ones(21, dtype=np.float64)
        best_result /= np.linalg.norm(best_result)
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            result_polish = minimize(
                objective_J,
                best_result,
                method='SLSQP',
                jac=jac_J,
                constraints=cons_21,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-15
                }
            )
            if result_polish.fun < best_obj:
                best_obj = result_polish.fun
                best_result = result_polish.x.copy()
        except Exception:
            pass
            
    c = best_result
    
    # Ensure valid range
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c / max_abs * 1e-2
    elif max_abs > 1e5:
        c = c / max_abs * 1e4
        
    return c

# EVOLVE-BLOCK-END
