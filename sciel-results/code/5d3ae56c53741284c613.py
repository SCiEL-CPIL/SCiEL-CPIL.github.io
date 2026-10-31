"""Uncertainty bound minimization via secular equation + Newton refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, pi

_DEADLINE = 205.0


def _hermite_coeffs(n):
    """Return coefficients of physicists' Hermite polynomial H_n(x) as list (low to high power)."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    h_prev = [1.0]
    h_curr = [0.0, 2.0]
    for m in range(2, n + 1):
        h_new = [0.0] * (len(h_curr) + 1)
        for p, c in enumerate(h_curr):
            if abs(c) > 1e-15:
                h_new[p + 1] += 2.0 * c
        for p, c in enumerate(h_prev):
            if abs(c) > 1e-15:
                h_new[p] -= 2.0 * (m - 1) * c
        h_new = [c for c in h_new]
        h_prev, h_curr = h_curr, h_new
    return h_curr


def _int_x2m(m):
    """∫_{-∞}^{∞} x^{2m} e^{-x^2} dx = sqrt(π) (2m-1)!! / 2^m"""
    if m == 0:
        return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val


# Precompute Hermite coefficient lists
_H0 = _hermite_coeffs(0)
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)

_DEG_MAX = 12
_POLY_LEN = _DEG_MAX + 1

# Convert to numpy arrays
_H0_arr = np.array(_H0)
_H4_arr = np.array(_H4)
_H8_arr = np.array(_H8)
_H12_arr = np.array(_H12)

# H values at zero
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 2520.0
_H12_0 = 27720.0


def _poly_product(p1, p2):
    """Multiply two polynomials given as coefficient arrays (low to high power)."""
    result = np.zeros(len(p1) + len(p2) - 1)
    for i in range(len(p1)):
        if p1[i] != 0.0:
            for j in range(len(p2)):
                if p2[j] != 0.0:
                    result[i + j] += p1[i] * p2[j]
    return result


def _int_poly_sq(p):
    """∫ p(x)^2 e^{-x^2} dx where p is a coefficient array."""
    p2 = _poly_product(p, p)
    result = 0.0
    for k in range(0, len(p2), 2):
        m = k // 2
        result += p2[k] * _int_x2m(m)
    return result


def _int_x2_poly_sq(p):
    """∫ x^2 p(x)^2 e^{-x^2} dx."""
    p2 = _poly_product(p, p)
    result = 0.0
    for k in range(0, len(p2), 2):
        m = (k + 2) // 2
        result += p2[k] * _int_x2m(m)
    return result


def _int_prod(p1, p2):
    """∫ p1(x) p2(x) e^{-x^2} dx."""
    prod = _poly_product(p1, p2)
    result = 0.0
    for k in range(0, len(prod), 2):
        m = k // 2
        result += prod[k] * _int_x2m(m)
    return result


def _diff_minus_x(p):
    """Compute q(x) = p'(x) - x*p(x)."""
    n = len(p)
    q = np.zeros(n + 1)
    for k in range(1, n):
        q[k - 1] += p[k] * k
    for k in range(n):
        q[k + 1] += -p[k]
    return q


def _compute_bound(a0, a4, a8):
    """Compute the uncertainty bound sqrt(M2 * G) / M0."""
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    p = a0 * _H0_arr + a4 * _H4_arr + a8 * _H8_arr + a12 * _H12_arr
    
    M0 = _int_poly_sq(p)
    M2 = _int_x2_poly_sq(p)
    
    q = _diff_minus_x(p)
    G = _int_poly_sq(q)
    
    if M0 <= 1e-15 or G <= 1e-15:
        return np.inf
    
    return sqrt(M2 * G) / M0


def _compute_bound_and_grad(a0, a4, a8):
    """Compute bound and gradient using finite differences."""
    f0 = _compute_bound(a0, a4, a8)
    h = 1e-8
    df0 = (_compute_bound(a0 + h, a4, a8) - _compute_bound(a0 - h, a4, a8)) / (2 * h)
    df4 = (_compute_bound(a0, a4 + h, a8) - _compute_bound(a0, a4 - h, a8)) / (2 * h)
    df8 = (_compute_bound(a0, a4, a8 + h) - _compute_bound(a0, a4, a8 - h)) / (2 * h)
    return f0, np.array([df0, df4, df8])


def _build_matrices():
    """
    Build the quadratic form matrices for the secular equation approach.
    
    We express M0, M2, G as quadratic forms in v = [a0, a4, a8]:
    M0 = v^T A0 v, M2 = v^T A2 v, G = v^T AG v
    
    The constraint g(0) = 0 gives a12 = -(a0 + 12*a4 + 2520*a8)/27720.
    So p = a0*e0 + a4*e4 + a8*e8 where:
    e0 = H0 - (1/27720)*H12
    e4 = H4 - (12/27720)*H12
    e8 = H8 - (2520/27720)*H12
    """
    e0 = _H0_arr - (_H0_0 / _H12_0) * _H12_arr
    e4 = _H4_arr - (_H4_0 / _H12_0) * _H12_arr
    e8 = _H8_arr - (_H8_0 / _H12_0) * _H12_arr
    
    e_list = [e0, e4, e8]
    
    A0 = np.zeros((3, 3))
    A2 = np.zeros((3, 3))
    AG = np.zeros((3, 3))
    
    q_list = [_diff_minus_x(e) for e in e_list]
    
    for i in range(3):
        for j in range(3):
            A0[i, j] = _int_prod(e_list[i], e_list[j])
            A2[i, j] = _int_x2_poly_sq(e_list[i]) if i == j else 0.0
            # For cross terms in M2: ∫ x^2 e_i e_j e^{-x^2}
            # Actually we need ∫ x^2 p_i p_j e^{-x^2}
            # Let me compute this properly
            if i != j:
                # ∫ x^2 e_i(x) e_j(x) e^{-x^2} dx
                # = ∫ (x e_i(x)) (e_j(x)) x e^{-x^2} dx ... no
                # Just compute directly
                prod = _poly_product(e_list[i], e_list[j])
                A2[i, j] = 0.0
                for k in range(0, len(prod), 2):
                    m = (k + 2) // 2
                    A2[i, j] += prod[k] * _int_x2m(m)
            AG[i, j] = _int_prod(q_list[i], q_list[j])
    
    # Symmetrize
    A0 = (A0 + A0.T) / 2
    A2 = (A2 + A2.T) / 2
    AG = (AG + AG.T) / 2
    
    return A0, A2, AG


def _solve_secular_equation(A0, A2, AG):
    """
    Solve the secular equation for the optimal eigenvalue.
    
    We want to minimize r = (v^T A2 v)(v^T AG v) / (v^T A0 v)^2.
    
    At the optimum, the gradient of r with respect to v (on the constraint v^T A0 v = 1)
    gives us a system. Let λ = r. Then:
    A2 v / (v^T A0 v) + AG v / (v^T A0 v) = λ * A0 v / (v^T A0 v)
    Wait, that's not quite right. Let me think more carefully.
    
    Actually, let's use a different approach. The minimum of 
    r = (v^T A2 v)(v^T AG v) / (v^T A0 v)^2
    
    can be found by setting up the Lagrangian. Let's fix v^T A0 v = 1,
    then minimize (v^T A2 v)(v^T AG v).
    
    The stationarity condition is:
    (v^T AG v) A2 v + (v^T A2 v) AG v = 2μ v^T A0 v * A0 v
    where μ is the Lagrange multiplier.
    
    This is a nonlinear eigenvalue problem. Let's use a simpler approach:
    parameterize and use Newton's method on the 3D problem.
    """
    # Use a multi-start approach with the matrices to guide initial guesses
    best_v = None
    best_bound = np.inf
    
    # Try several initial points
    initial_points = [
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.001]),
        np.array([-0.5, 0.2, -0.005]),
        np.array([-1.5, -0.1, 0.002]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([-2.0, 0.5, 0.0]),
        np.array([-0.5, 0.0, 0.0]),
    ]
    
    for v0 in initial_points:
        try:
            bound = _compute_bound(v0[0], v0[1], v0[2])
            if np.isfinite(bound) and bound < best_bound:
                best_bound = bound
                best_v = v0.copy()
        except Exception:
            pass
    
    return best_v, best_bound


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    start_time = time.time()
    
    # Phase 1: Build matrices and find initial point
    try:
        A0, A2, AG = _build_matrices()
        best_v, best_bound = _solve_secular_equation(A0, A2, AG)
    except Exception:
        best_v = np.array([-1.13, 0.04, 0.000309])
        best_bound = _compute_bound(best_v[0], best_v[1], best_v[2])
    
    if best_v is None:
        best_v = np.array([-1.13, 0.04, 0.000309])
        best_bound = _compute_bound(best_v[0], best_v[1], best_v[2])
    
    # Phase 2: Newton's method with line search
    if time.time() - start_time < _DEADLINE - 20:
        try:
            v = best_v.copy()
            for iteration in range(200):
                if time.time() - start_time > _DEADLINE - 15:
                    break
                
                f0, grad = _compute_bound_and_grad(v[0], v[1], v[2])
                
                if np.linalg.norm(grad) < 1e-14:
                    break
                
                # Hessian via finite differences
                h = 1e-6
                H = np.zeros((3, 3))
                for i in range(3):
                    vi_plus = v.copy()
                    vi_minus = v.copy()
                    vi_plus[i] += h
                    vi_minus[i] -= h
                    _, grad_plus = _compute_bound_and_grad(vi_plus[0], vi_plus[1], vi_plus[2])
                    _, grad_minus = _compute_bound_and_grad(vi_minus[0], vi_minus[1], vi_minus[2])
                    H[:, i] = (grad_plus - grad_minus) / (2 * h)
                
                H = (H + H.T) / 2
                
                # Regularize if needed
                eigvals = np.linalg.eigvalsh(H)
                if eigvals[0] < 1e-10:
                    H += (1e-8 - eigvals[0]) * np.eye(3)
                
                try:
                    step = np.linalg.solve(H, -grad)
                except np.linalg.LinAlgError:
                    step = -grad / np.linalg.norm(grad)
                
                # Line search
                alpha = 1.0
                for _ in range(50):
                    v_new = v + alpha * step
                    f_new = _compute_bound(v_new[0], v_new[1], v_new[2])
                    if np.isfinite(f_new) and f_new < f0 - 1e-4 * alpha * np.dot(grad, step):
                        v = v_new
                        break
                    alpha *= 0.5
                
                if np.isfinite(f_new) and f_new < best_bound:
                    best_bound = f_new
                    best_v = v.copy()
                
                if alpha < 1e-10:
                    break
        except Exception:
            pass
    
    # Phase 3: Multi-start local optimization
    if time.time() - start_time < _DEADLINE - 10:
        from scipy.optimize import minimize
        
        # Generate diverse starting points around the best found
        rng = np.random.default_rng(42)
        n_starts = 20
        
        for i in range(n_starts):
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            if i == 0:
                v0 = best_v.copy()
            else:
                scale = 0.5 * (1.0 + i / n_starts)
                v0 = best_v + scale * rng.standard_normal(3) * np.array([1.0, 0.2, 0.01])
            
            try:
                result = minimize(
                    lambda x: _compute_bound(x[0], x[1], x[2]),
                    v0,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                if np.isfinite(result.fun) and result.fun < best_bound:
                    best_bound = result.fun
                    best_v = result.x.copy()
            except Exception:
                pass
    
    # Phase 4: Final Newton polish
    if time.time() - start_time < _DEADLINE - 5:
        try:
            v = best_v.copy()
            for iteration in range(50):
                if time.time() - start_time > _DEADLINE - 2:
                    break
                
                f0, grad = _compute_bound_and_grad(v[0], v[1], v[2])
                
                if np.linalg.norm(grad) < 1e-16:
                    break
                
                h = 1e-8
                H = np.zeros((3, 3))
                for i in range(3):
                    vi_plus = v.copy()
                    vi_minus = v.copy()
                    vi_plus[i] += h
                    vi_minus[i] -= h
                    _, grad_plus = _compute_bound_and_grad(vi_plus[0], vi_plus[1], vi_plus[2])
                    _, grad_minus = _compute_bound_and_grad(vi_minus[0], vi_minus[1], vi_minus[2])
                    H[:, i] = (grad_plus - grad_minus) / (2 * h)
                
                H = (H + H.T) / 2
                eigvals = np.linalg.eigvalsh(H)
                if eigvals[0] < 1e-12:
                    H += (1e-10 - eigvals[0]) * np.eye(3)
                
                try:
                    step = np.linalg.solve(H, -grad)
                except np.linalg.LinAlgError:
                    break
                
                alpha = 1.0
                for _ in range(30):
                    v_new = v + alpha * step
                    f_new = _compute_bound(v_new[0], v_new[1], v_new[2])
                    if np.isfinite(f_new) and f_new < f0:
                        v = v_new
                        break
                    alpha *= 0.5
                
                if np.isfinite(f_new) and f_new < best_bound:
                    best_bound = f_new
                    best_v = v.copy()
                
                if alpha < 1e-12:
                    break
        except Exception:
            pass
    
    return best_v
# EVOLVE-BLOCK-END
