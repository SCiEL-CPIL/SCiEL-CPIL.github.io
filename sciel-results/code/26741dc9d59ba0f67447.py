"""Combined: Jacobi matrix parameterization + gradient-based L-BFGS-B optimization.
Parameterize 12 double roots as eigenvalues of a tridiagonal Jacobi matrix from
modified Laguerre recurrence with tunable weight exponent. This enforces positive,
ordered roots inherently. Use analytic gradient for fast convergence."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre, roots_genlaguerre, factorial

_FACTS = np.array([math.factorial(k) for k in range(60)], dtype=np.float64)

def _poly_moment(coeffs):
    """Compute integral of poly (ascending coeff order) * e^{-x} dx over [0,inf)."""
    n = len(coeffs)
    if n == 0:
        return 0.0
    m = min(n, 60)
    return float(np.dot(coeffs[:m], _FACTS[:m]))

def _build_q(roots):
    """Build Q(x) = prod (x-x_i) and return Q_coeffs in ascending order."""
    from numpy.polynomial.polynomial import polyfromroots
    return polyfromroots(roots)

def _compute_c4(roots, need_grad=False):
    """Compute C4 = I2 / I1^2 where I1 = int P e^{-x} dx, I2 = int P^2 e^{-x} dx.
    P = Q^2, P^2 = Q^4. Uses analytic gradient via polynomial division."""
    roots = np.asarray(roots, dtype=np.float64)
    n = len(roots)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf

    Q = _build_q(roots)
    P = np.convolve(Q, Q)      # P = Q^2
    P2 = np.convolve(P, P)     # P^2 = Q^4
    
    I1 = _poly_moment(P)
    I2 = _poly_moment(P2)
    if I1 <= 0 or not np.isfinite(I1) or not np.isfinite(I2):
        if need_grad:
            return np.inf, np.zeros_like(roots)
        return np.inf
    c4 = I2 / (I1 * I1)
    if not need_grad:
        return float(c4)
    
    # Analytic gradient:
    # dP/dx_i = 2*P/(x-x_i), d(P^2)/dx_i = 4*P^2/(x-x_i)
    # dI1/dx_i = 2 * int [P/(x-x_i)] e^{-x} dx
    # dI2/dx_i = 4 * int [P^2/(x-x_i)] e^{-x} dx
    # dC4/dx_i = dI2/I1^2 - 2*I2*dI1/I1^3
    
    from numpy.polynomial.polynomial import polydiv
    
    dI1 = np.zeros(n)
    dI2 = np.zeros(n)
    
    for i in range(n):
        divisor = np.array([-roots[i], 1.0])
        S_i, _ = polydiv(P, divisor)
        T_i, _ = polydiv(P2, divisor)
        
        int_S = _poly_moment(S_i)
        int_T = _poly_moment(T_i)
        
        dI1[i] = 2.0 * int_S
        dI2[i] = 4.0 * int_T
    
    I1_sq = I1 * I1
    I1_cu = I1_sq * I1
    grad = dI2 / I1_sq - 2.0 * I2 * dI1 / I1_cu
    
    return float(c4), grad

def _jacobi_matrix_from_beta(beta):
    """Construct 12x12 Jacobi matrix from modified Laguerre recurrence with weight exponent beta.
    
    For generalized Laguerre L_k^{(alpha)}(x) with weight x^alpha e^{-x},
    the three-term recurrence is:
    (k+1) L_{k+1}^{(alpha)}(x) = (2k + alpha + 1 - x) L_k^{(alpha)}(x) - (k+alpha) L_{k-1}^{(alpha)}(x)
    
    So: a_k = 2k + alpha + 1, b_k = k + alpha (for k >= 1), b_0 = alpha
    Jacobi matrix J has J[k,k] = a_k, J[k,k+1] = J[k+1,k] = sqrt(b_{k+1})
    
    For modified weight x^beta e^{-x}, set alpha = beta.
    """
    n = 12
    alpha = float(beta)
    J = np.zeros((n, n))
    for k in range(n):
        J[k, k] = 2.0 * k + alpha + 1.0
        if k < n - 1:
            b_next = (k + 1) + alpha
            if b_next > 0:
                J[k, k+1] = math.sqrt(b_next)
                J[k+1, k] = math.sqrt(b_next)
    return J

def _objective_jacobi(params):
    """Objective: first 12 params are Jacobi diagonal entries (a_k),
    next 11 params are log of off-diagonal entries (log_b_k for k=1..11).
    This ensures positive off-diagonals. Eigenvalues of resulting symmetric
    matrix are the roots."""
    n = 12
    a = params[:n]
    log_b = params[n:]
    b = np.exp(log_b)
    
    # Build Jacobi matrix
    J = np.zeros((n, n))
    for k in range(n):
        J[k, k] = a[k]
        if k < n - 1:
            J[k, k+1] = b[k]
            J[k+1, k] = b[k]
    
    # Compute eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(J)
    except Exception:
        return 1e10, np.zeros_like(params)
    
    roots = np.sort(eigvals)
    
    # Check positivity
    if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
        return 1e10, np.zeros_like(params)
    
    c4, grad_roots = _compute_c4(roots, need_grad=True)
    if not np.isfinite(c4):
        return 1e10, np.zeros_like(params)
    
    # Chain rule: dC4/dJ = V^T (dC4/droots) V where V is eigenvector matrix
    # For symmetric matrix: d(lambda_i)/dJ_{pq} = v_i[p] * v_i[q]
    # So dC4/dJ_{pq} = sum_i (dC4/dlambda_i) * v_i[p] * v_i[q]
    try:
        eigvals_full, V = np.linalg.eigh(J)
    except Exception:
        return 1e10, np.zeros_like(params)
    
    # dC4/dJ as matrix
    dC4_dJ = (V * grad_roots[np.newaxis, :]) @ V.T
    
    # Map back to parameters
    grad_params = np.zeros_like(params)
    for k in range(n):
        grad_params[k] = dC4_dJ[k, k]
        if k < n - 1:
            # dJ[k,k+1]/dlog_b_k = b_k
            grad_params[n + k] = dC4_dJ[k, k+1] * b[k]
    
    if not np.all(np.isfinite(grad_params)):
        return 1e10, np.zeros_like(params)
    
    return c4, grad_params

def _objective_jacobi_safe(params):
    """Safe wrapper for scipy minimize."""
    try:
        c4, g = _objective_jacobi(params)
        if not np.isfinite(c4) or not np.all(np.isfinite(g)):
            return 1e10, np.zeros_like(params)
        return c4, g
    except Exception:
        return 1e10, np.zeros_like(params)

def _init_jacobi_params_from_roots(roots):
    """Initialize Jacobi parameters from given roots using Stieltjes procedure
    with the discrete measure at the roots (equal weights)."""
    n = 12
    roots = np.sort(roots).astype(np.float64)
    
    # Use Stieltjes procedure with discrete measure at roots
    # mu = sum_i delta_{r_i} (equal weights)
    # Moments: m_k = sum_i r_i^k
    
    # Compute moments up to order 2n-1
    moments = np.zeros(2 * n)
    for k in range(2 * n):
        moments[k] = np.sum(roots ** k)
    
    # Stieltjes procedure to get Jacobi parameters
    # p_0 = 1, h_0 = m_0
    # p_{k+1} = (x - a_k) p_k - b_k p_{k-1}
    # where a_k = <x p_k, p_k> / h_k, b_k = h_{k+1} / h_k
    
    a_k = np.zeros(n)
    b_k = np.zeros(n - 1)
    
    # h_k = <p_k, p_k>
    # For discrete measure: <f, g> = sum_i f(r_i) g(r_i)
    
    # p_0 = 1
    p0_vals = np.ones(n)
    h0 = np.sum(p0_vals ** 2)  # = n
    
    if h0 < 1e-15:
        # Fallback: use standard Laguerre
        x0, _ = roots_laguerre(n)
        roots_fb = np.sort(x0)
        a_k[:] = np.mean(roots_fb)
        b_k[:] = np.std(roots_fb)
        params = np.zeros(2 * n - 1)
        for k in range(n):
            params[k] = a_k[k]
            if k < n - 1:
                params[n + k] = math.log(max(b_k[k], 1e-10))
        return params
    
    # a_0 = <x, 1> / <1, 1> = m_1 / m_0
    a_k[0] = moments[1] / moments[0]
    
    # p_1 = (x - a_0) * 1 = x - a_0
    p1_vals = roots - a_k[0]
    h1 = np.sum(p1_vals ** 2)
    
    if h1 < 1e-15:
        b_k[0] = 1e-10
    else:
        b_k[0] = h1 / h0
    
    for k in range(1, n):
        if k < n - 1:
            # Compute p_k values at roots
            # p_k = (x - a_{k-1}) p_{k-1} - b_{k-1} p_{k-2}
            if k == 1:
                pk_vals = p1_vals
                pkm1_vals = p0_vals
            else:
                # Need to track polynomials
                # This is getting complex; let me use a simpler approach
                pass
        
        # For k >= 1, use the moment-based formula
        # a_k = m_{k+1} adjusted... this is complex.
        # Let me just use the direct approach:
        # Build the companion matrix from the monic polynomial
    
    # Simpler fallback: construct companion matrix of the monic polynomial
    from numpy.polynomial.polynomial import polyfromroots
    p_coeffs = polyfromroots(roots)  # ascending order, monic
    
    # Companion matrix for polynomial p(x) = x^n + c_{n-1} x^{n-1} + ... + c_0
    # The eigenvalues of the companion matrix are the roots
    # Companion matrix (Frobenius form):
    # [0 0 ... 0 -c_0]
    # [1 0 ... 0 -c_1]
    # [0 1 ... 0 -c_2]
    # ...
    # [0 0 ... 1 -c_{n-1}]
    
    # But this isn't symmetric. For a symmetric Jacobi matrix, we need
    # the orthogonal polynomial recurrence.
    
    # Alternative: use scipy's approach - the Jacobi matrix for discrete measure
    # at points r_i with weights w_i can be computed via the Stieltjes procedure.
    
    # Let me just do the Stieltjes procedure properly:
    a_k = np.zeros(n)
    b_k = np.zeros(n - 1)
    
    # p_0(x) = 1
    p_prev = np.ones(n)  # values of p_{k-1} at roots
    h_prev = np.sum(p_prev ** 2)
    
    a_k[0] = np.sum(roots * p_prev ** 2) / h_prev
    
    # p_1(x) = (x - a_0) p_0(x)
    p_curr = (roots - a_k[0]) * p_prev
    h_curr = np.sum(p_curr ** 2)
    
    if h_curr < 1e-15:
        b_k[0] = 1e-10
    else:
        b_k[0] = h_curr / h_prev
    
    for k in range(1, n):
        if k < n - 1:
            # p_{k+1} = (x - a_k) p_k - b_k p_{k-1}
            # But we need a_k first
            # a_k = <x p_k, p_k> / <p_k, p_k>
            a_k[k] = np.sum(roots * p_curr ** 2) / h_curr
            
            p_next = (roots - a_k[k]) * p_curr - b_k[k] * p_prev
            h_next = np.sum(p_next ** 2)
            
            if h_next < 1e-15:
                b_k[k] = 1e-10
            else:
                b_k[k] = h_next / h_curr
            
            p_prev = p_curr
            p_curr = p_next
            h_prev = h_curr
            h_curr = h_next
    
    # Build initial Jacobi parameters
    params = np.zeros(2 * n - 1)
    for k in range(n):
        params[k] = a_k[k]
        if k < n - 1:
            params[n + k] = math.log(max(b_k[k], 1e-10))
    
    return params

def _init_jacobi_from_beta(beta):
    """Initialize Jacobi parameters from a given beta value."""
    J = _jacobi_matrix_from_beta(beta)
    n = 12
    params = np.zeros(2*n - 1)
    for k in range(n):
        params[k] = J[k, k]
        if k < n - 1:
            params[n + k] = math.log(max(J[k, k+1], 1e-10))
    return params

def construct():
    start = time.time()
    DEADLINE = 630.0
    SAFETY = 15.0

    def time_ok():
        return (time.time() - start) < DEADLINE - SAFETY

    best_roots = None
    best_c4 = np.inf

    # Phase 1: Search over beta for good initial Jacobi matrices
    beta_values = np.unique(np.concatenate([
        np.logspace(-2, 3, 50),
        np.linspace(0.1, 50, 100),
    ]))
    
    best_beta = None
    for beta in beta_values:
        if not time_ok():
            break
        try:
            J = _jacobi_matrix_from_beta(beta)
            eigvals = np.linalg.eigvalsh(J)
            roots = np.sort(eigvals)
            if np.all(roots > 0) and np.isfinite(roots).all():
                c4 = _compute_c4(roots)
                if np.isfinite(c4) and c4 < best_c4:
                    best_c4 = c4
                    best_roots = roots.copy()
                    best_beta = beta
        except Exception:
            pass
    
    # Fallback
    if best_roots is None:
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
        best_c4 = _compute_c4(best_roots)
        best_beta = 1.0

    # Phase 2: L-BFGS-B in Jacobi parameter space from best roots
    if time_ok():
        try:
            init_params = _init_jacobi_params_from_roots(best_roots)
            res = minimize(
                _objective_jacobi_safe,
                init_params,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 200}
            )
            if np.all(np.isfinite(res.x)):
                n = 12
                a = res.x[:n]
                log_b = res.x[n:]
                b = np.exp(log_b)
                J = np.zeros((n, n))
                for k in range(n):
                    J[k, k] = a[k]
                    if k < n - 1:
                        J[k, k+1] = b[k]
                        J[k+1, k] = b[k]
                eigvals = np.linalg.eigvalsh(J)
                candidate = np.sort(eigvals)
                if np.all(candidate > 0) and np.isfinite(candidate).all():
                    c4 = _compute_c4(candidate)
                    if np.isfinite(c4) and c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
        except Exception:
            pass

    # Phase 3: Multi-start from different beta values and perturbations
    if time_ok():
        candidates_to_refine = []
        
        # From best roots
        try:
            candidates_to_refine.append(_init_jacobi_params_from_roots(best_roots))
        except Exception:
            pass
        
        # From various betas
        for beta in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
            try:
                candidates_to_refine.append(_init_jacobi_from_beta(beta))
            except Exception:
                pass
        
        # Perturb best params
        rng = np.random.default_rng(42)
        try:
            base_params = _init_jacobi_params_from_roots(best_roots)
            for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
                pert = base_params + scale * rng.standard_normal(23)
                candidates_to_refine.append(pert)
        except Exception:
            pass
        
        for init in candidates_to_refine:
            if not time_ok():
                break
            try:
                res = minimize(
                    _objective_jacobi_safe,
                    init,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 200}
                )
                if np.all(np.isfinite(res.x)):
                    n = 12
                    a = res.x[:n]
                    log_b = res.x[n:]
                    b = np.exp(log_b)
                    J = np.zeros((n, n))
                    for k in range(n):
                        J[k, k] = a[k]
                        if k < n - 1:
                            J[k, k+1] = b[k]
                            J[k+1, k] = b[k]
                    eigvals = np.linalg.eigvalsh(J)
                    candidate = np.sort(eigvals)
                    if np.all(candidate > 0) and np.isfinite(candidate).all():
                        c4 = _compute_c4(candidate)
                        if np.isfinite(c4) and c4 < best_c4:
                            best_c4 = c4
                            best_roots = candidate.copy()
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    for polish_iter in range(5):
        if not time_ok():
            break
        try:
            init_params = _init_jacobi_params_from_roots(best_roots)
            res = minimize(
                _objective_jacobi_safe,
                init_params,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-17, 'maxls': 300}
            )
            if np.all(np.isfinite(res.x)):
                n = 12
                a = res.x[:n]
                log_b = res.x[n:]
                b = np.exp(log_b)
                J = np.zeros((n, n))
                for k in range(n):
                    J[k, k] = a[k]
                    if k < n - 1:
                        J[k, k+1] = b[k]
                        J[k+1, k] = b[k]
                eigvals = np.linalg.eigvalsh(J)
                candidate = np.sort(eigvals)
                if np.all(candidate > 0) and np.isfinite(candidate).all():
                    c4 = _compute_c4(candidate)
                    if np.isfinite(c4) and c4 < best_c4:
                        best_c4 = c4
                        best_roots = candidate.copy()
                    else:
                        break
        except Exception:
            break

    # Final safety check
    if best_roots is None or not np.all(np.isfinite(best_roots)) or not np.all(best_roots > 0):
        x0, _ = roots_laguerre(12)
        best_roots = np.sort(x0)
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
