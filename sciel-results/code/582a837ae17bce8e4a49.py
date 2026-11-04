"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _precompute_factorials(n):
    """Precompute factorials 0! through n!."""
    facts = np.ones(n + 1, dtype=np.float64)
    for k in range(1, n + 1):
        facts[k] = facts[k - 1] * k
    return facts

def _poly_moment(coeffs, facts, shift=0):
    """
    Compute sum_i coeffs[i] * (i + shift)! where coeffs[i] is coefficient of x^i.
    This equals integral_0^inf P(x) * x^shift * e^{-x} dx.
    """
    max_deg = len(coeffs) - 1
    needed = max_deg + shift + 1
    if needed > len(facts):
        # Extend factorials
        extra = needed - len(facts) + 1
        new_facts = np.ones(extra, dtype=np.float64)
        start_val = facts[-1]
        start_idx = len(facts)
        for k in range(extra):
            start_val *= (start_idx + k)
            new_facts[k] = start_val
        facts = np.concatenate([facts, new_facts])
    return np.dot(coeffs, facts[shift:shift + len(coeffs)])

def _compute_c4(roots, facts=None):
    """
    Compute the C4 upper bound for given 12 double-root locations.
    C4 = integral_0^inf P(x)^2 * x^2 * e^{-x} dx where P(x) = prod (x - r_i)^2
    """
    roots = np.asarray(roots, dtype=np.float64)
    m = len(roots)
    
    # Build P(x) = prod_i (x - r_i)^2
    # Coefficients in ascending order: coeffs[k] = coeff of x^k
    p_coeffs = np.array([1.0], dtype=np.float64)
    for r in roots:
        factor = np.array([r * r, -2.0 * r, 1.0], dtype=np.float64)
        p_coeffs = np.convolve(p_coeffs, factor)
    
    # P(x)^2
    p2_coeffs = np.convolve(p_coeffs, p_coeffs)
    
    # C4 = sum_k p2_coeffs[k] * (k+2)!
    if facts is None:
        facts = _precompute_factorials(len(p2_coeffs) + 2)
    
    return _poly_moment(p2_coeffs, facts, shift=2)

def _compute_c4_and_grad(roots, facts=None):
    """
    Compute C4 and its gradient with respect to root positions.
    dC4/dr_i = -4 * integral Q_i(x) * (x - r_i) * x^2 * e^{-x} dx
    where Q_i(x) = prod_{j != i} (x - r_j)^2
    """
    roots = np.asarray(roots, dtype=np.float64)
    m = len(roots)
    
    # Precompute factorials
    max_deg = 2 * m  # degree of P(x)^2
    if facts is None:
        facts = _precompute_factorials(max_deg + 2)
    
    # Build P(x)
    p_coeffs = np.array([1.0], dtype=np.float64)
    for r in roots:
        factor = np.array([r * r, -2.0 * r, 1.0], dtype=np.float64)
        p_coeffs = np.convolve(p_coeffs, factor)
    
    # C4
    p2_coeffs = np.convolve(p_coeffs, p_coeffs)
    c4 = _poly_moment(p2_coeffs, facts, shift=2)
    
    # Gradient
    grad = np.zeros(m, dtype=np.float64)
    for i in range(m):
        # Build Q_i(x) = prod_{j != i} (x - r_j)^2
        q_coeffs = np.array([1.0], dtype=np.float64)
        for j in range(m):
            if j == i:
                continue
            factor = np.array([roots[j] * roots[j], -2.0 * roots[j], 1.0], dtype=np.float64)
            q_coeffs = np.convolve(q_coeffs, factor)
        
        # Multiply by (x - r_i): q_coeffs * [0, 1, -r_i] in ascending order
        # (x - r_i) = -r_i + x, so coeffs = [-r_i, 1]
        prod_coeffs = np.convolve(q_coeffs, np.array([-roots[i], 1.0], dtype=np.float64))
        
        # integral = sum_k prod_coeffs[k] * (k+2)!
        integral_val = _poly_moment(prod_coeffs, facts, shift=2)
        grad[i] = -4.0 * integral_val
    
    return c4, grad

def _c4_from_alpha(alpha):
    """
    Compute C4 for roots that are eigenvalues of the Jacobi matrix
    associated with generalized Laguerre polynomials L_m^(alpha)(x).
    
    The Jacobi matrix for Laguerre polynomials has:
    - Diagonal: 2*i + alpha + 1 for i = 0, ..., m-1
    - Off-diagonal: sqrt(i * (i + alpha)) for i = 0, ..., m-2
    
    For m=12, this gives 12 eigenvalues (the roots).
    """
    m = 12
    # Build tridiagonal Jacobi matrix
    J = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        J[i, i] = 2.0 * i + alpha + 1.0
    for i in range(m - 1):
        offdiag = math.sqrt(i * (i + alpha)) if i > 0 else 0.0
        if offdiag > 0:
            J[i, i + 1] = offdiag
            J[i + 1, i] = offdiag
    
    # Eigenvalues are the roots
    roots = np.linalg.eigvalsh(J)
    
    # Ensure positive and sorted
    roots = np.sort(roots)
    if np.any(roots <= 0):
        return np.inf, None
    
    c4, grad = _compute_c4_and_grad(roots)
    return c4, roots

def construct():
    """Return 12 positive double-root locations minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 600.0  # 600s safety margin
    
    best_x = None
    best_val = np.inf
    
    # Precompute factorials for efficiency
    max_deg = 48  # degree of P(x)^2 for m=12
    facts = _precompute_factorials(max_deg + 2)
    
    def objective(x):
        val, _ = _compute_c4_and_grad(x, facts)
        return val
    
    def gradient(x):
        _, grad = _compute_c4_and_grad(x, facts)
        return grad
    
    # Strategy 1: Optimize alpha parameter (1D search)
    # The alpha parameter controls the spread of the Laguerre roots
    alpha_candidates = []
    # Try a range of alpha values
    for a in np.linspace(-0.9, 5.0, 50):
        if time.time() > deadline - 30.0:
            break
        try:
            c4, roots = _c4_from_alpha(a)
            if np.isfinite(c4) and c4 < best_val:
                best_val = c4
                best_x = roots.copy()
                alpha_candidates.append((a, roots.copy()))
        except Exception:
            continue
    
    # Strategy 2: Multi-start direct optimization on root positions
    if best_x is not None:
        initializations = [best_x.copy()]
    else:
        initializations = [np.array([4.0 * (i + 1) for i in range(12)])]
    
    # Add more diverse initializations
    if best_x is not None:
        initializations.append(best_x + np.random.randn(12) * 0.5)
        initializations.append(best_x * (1.0 + np.random.randn(12) * 0.1))
    
    initializations.extend([
        np.linspace(1.0, 50.0, 12),
        np.geomspace(1.0, 50.0, 12),
        np.array([4.0 * (i + 1) for i in range(12)]),
        np.array([2.0 * (i + 1) ** 1.5 for i in range(12)]),
        np.array([1.0 + 4.0 * i + 0.5 * i * i for i in range(12)]),
    ])
    
    # Shuffle for diversity
    np.random.shuffle(initializations)
    
    for x_init in initializations:
        if time.time() > deadline - 30.0:
            break
        
        # Ensure positive and sorted
        x_init = np.sort(np.abs(x_init) + 0.1)
        
        try:
            res = minimize(
                objective,
                x_init,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(0.01, None)] * 12,
                options={'maxiter': 5000, 'gtol': 1e-10, 'ftol': 1e-10, 'maxfun': 10000}
            )
            
            if np.isfinite(res.fun) and res.fun < best_val:
                # Ensure all roots are positive and distinct
                if np.all(res.x > 0.01) and len(np.unique(np.round(res.x, 10))) == 12:
                    best_val = res.fun
                    best_x = res.x.copy()
        except Exception:
            continue
    
    # Strategy 3: Nelder-Mead polish on best solution
    if best_x is not None and time.time() < deadline - 30.0:
        try:
            res_nm = minimize(
                objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if np.isfinite(res_nm.fun) and res_nm.fun < best_val:
                if np.all(res_nm.x > 0.01) and len(np.unique(np.round(res_nm.x, 10))) == 12:
                    best_val = res_nm.fun
                    best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Strategy 4: If we have time, do more alpha-based searches and polish
    if best_x is not None and time.time() < deadline - 60.0:
        # Fine-tune around best alpha
        if alpha_candidates:
            best_alpha = alpha_candidates[0][0]
            for a in np.linspace(best_alpha - 0.5, best_alpha + 0.5, 30):
                if time.time() > deadline - 30.0:
                    break
                try:
                    c4, roots = _c4_from_alpha(a)
                    if np.isfinite(c4) and c4 < best_val:
                        best_val = c4
                        best_x = roots.copy()
                except Exception:
                    continue
        
        # Polish with L-BFGS
        try:
            res = minimize(
                objective,
                best_x,
                jac=gradient,
                method='L-BFGS-B',
                bounds=[(0.01, None)] * 12,
                options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-12, 'maxfun': 10000}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                if np.all(res.x > 0.01) and len(np.unique(np.round(res.x, 10))) == 12:
                    best_val = res.fun
                    best_x = res.x.copy()
        except Exception:
            pass
    
    # Final validation
    if best_x is None:
        # Fallback: return the seed-like configuration
        best_x = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Ensure sorted and positive
    best_x = np.sort(best_x)
    if np.any(best_x <= 0):
        best_x = np.abs(best_x) + 0.1
        best_x = np.sort(best_x)
    
    return best_x.tolist()

# EVOLVE-BLOCK-END
