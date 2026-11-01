"""
Optimized search for Hermite-Gaussian coefficients minimizing the uncertainty bound.
Uses a multi-start strategy with Sobol sampling and analytic-gradient L-BFGS-B.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

_T0 = time.time()
_DEADLINE = _T0 + 180.0
_H12_0 = 10395.0  # H12(0), monic Hermite


def _I(n):
    """Integral of x^(2n) e^{-x^2} over R = (2n)!/(4^n n!) * sqrt(pi)."""
    if n < 0:
        return 0.0
    v = np.sqrt(np.pi)
    for k in range(1, n + 1):
        v *= (2.0 * k - 1.0) / (4.0 * k)
    return v


# Precompute integrals up to sufficient degree
# Max degree in P is 12, so P*P has degree 24.
# x^2 * P^2 has degree 26.
# We need integrals of x^(2k) e^{-x^2} for k up to 13.
_I = np.array([_I(n) for n in range(26)])

# Number of monomial terms (x^0, x^2, ..., x^12) -> 7 terms
N_TERMS = 7

# Precompute Moment Matrices for Position (x) side
# M0_x[i, j] = integral x^(2i) x^(2j) e^{-x^2} dx = I[i+j]
# M2_x[i, j] = integral x^(2i+2) x^(2j) e^{-x^2} dx = I[i+j+1]
M0_x = np.zeros((N_TERMS, N_TERMS))
M2_x = np.zeros((N_TERMS, N_TERMS))
for i in range(N_TERMS):
    for j in range(N_TERMS):
        M0_x[i, j] = _I[i + j]
        M2_x[i, j] = _I[i + j + 1]

# For Fourier side, the effective polynomial coefficients are q_k = (-1)^k c_k
# Let D = diag([1, -1, 1, -1, 1, -1, 1])
D = np.diag([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

# The Fourier variance involves |hat{g}|^2.
# The seed applies a factor of 2*pi to the moments n0, n2, n4.
# Let's replicate the seed's logic:
# n0 = 2*pi * (q^T M0_x q)  -- wait, does the kernel change?
# The seed used _moments(q) which computes sum q_i q_j I[i+j].
# This is exactly q^T M0_x q.
# So M0_k = 2*pi * M0_x (if q^T M0_x q is the base integral).
# And M2_k = 2*pi * M2_x.
# Note: The seed's q is derived from p (monomial coeffs of P).
# q_i = (-1)^i p_i.
# So q^T M0_x q = p^T D M0_x D p.
# Since M0_x and M2_x have specific symmetries, D M0_x D is just a signed version.
# But we can just compute q^T M0_x q directly if we pass q.
# However, for gradient computation, it's cleaner to define matrices in terms of p.
# Let's define matrices for the quadratic forms in p.
# Q0_x(p) = p^T M0_x p
# Q2_x(p) = p^T M2_x p
# Q0_k(p) = 2*pi * p^T (D M0_x D) p
# Q2_k(p) = 2*pi * p^T (D M2_x D) p

M0_k_mat = 2.0 * np.pi * (D @ M0_x @ D)
M2_k_mat = 2.0 * np.pi * (D @ M2_x @ D)


def _get_monomial_coeffs(a0, a4, a8):
    """
    Convert Hermite coefficients [a0, a4, a8] to monomial coefficients p.
    Constraint: P(0) = 0 => a12 is fixed.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / _H12_0
    p = np.array([
        a0 + 3.0 * a4 + 105.0 * a8 + _H12_0 * a12,
        -6.0 * a4 - 420.0 * a8 - 7560.0 * a12,
        a4 + 210.0 * a8 + 13860.0 * a12,
        -28.0 * a8 - 6930.0 * a12,
        a8 + 1155.0 * a12,
        -66.0 * a12,
        a12,
    ])
    return p


def _bound_and_grad(x):
    """
    Compute the uncertainty bound and its analytic gradient.
    x = [a0, a4, a8]
    """
    a0, a4, a8 = x[0], x[1], x[2]
    p = _get_monomial_coeffs(a0, a4, a8)
    
    # Quadratic forms
    m0 = p @ M0_x @ p
    m2 = p @ M2_x @ p
    n0 = p @ M0_k_mat @ p
    n2 = p @ M2_k_mat @ p
    
    if not (np.isfinite(m0) and np.isfinite(n0)) or m0 <= 1e-30 or n0 <= 1e-30:
        return 1e18, np.zeros(3)
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0.0 or vF <= 0.0:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vf * vF)
    
    # Gradient of bound w.r.t. a = [a0, a4, a8]
    # bound = sqrt( (m2/m0) * (n2/n0) )
    # Let R = vf * vF. bound = sqrt(R).
    # d bound / da = (1 / (2 * bound)) * dR/da
    # dR/da = vF * d(vf)/da + vf * d(vF)/da
    
    # vf = m2/m0. d(vf)/da = (dm2 * m0 - m2 * dm0) / m0^2
    # dm0/da = 2 * (dp/da)^T M0_x p
    # dm2/da = 2 * (dp/da)^T M2_x p
    
    # We need dp/da. p is linear in a.
    # p = J @ a + b (where b is 0 here since a12 is linear in a and p is linear in a12)
    # Actually p is homogeneous in a.
    # p = A @ a where A is a 7x3 matrix.
    
    A = np.array([
        [1.0 + _H12_0 * (-1.0/_H12_0), 3.0 + _H12_0 * (-3.0/_H12_0), 105.0 + _H12_0 * (-105.0/_H12_0)],
        [0.0 + _H12_0 * (-(-7560.0/_H12_0)), -6.0 + _H12_0 * (-(-22680.0/_H12_0)), -420.0 + _H12_0 * (-(-1587600.0/_H12_0))], # This is getting messy. Let's just differentiate the formula.
    ])
    
    # Better: Calculate Jacobian of p w.r.t a directly.
    # p[0] = a0 + 3a4 + 105a8 + 10395 a12. a12 = -(a0+3a4+105a8)/10395.
    # p[0] = a0 + 3a4 + 105a8 - (a0+3a4+105a8) = 0. 
    # Wait. The constraint P(0)=0 means the constant term of P is 0.
    # P(0) = a0*1 + a4*3 + a8*105 + a12*10395 = 0.
    # So p[0] is ALWAYS 0.
    # Let's check the seed's p[0]:
    # p[0] = a0 + 3a4 + 105a8 + 10395 * (-(a0+3a4+105a8)/10395) = 0.
    # So p[0] is identically zero.
    
    # Let's compute the Jacobian J = dp/da numerically or analytically.
    # Since p is linear in a, J is constant.
    # p_k(a) = sum_j J[k,j] a_j
    
    # Let's compute J by evaluating p at basis vectors.
    J = np.zeros((N_TERMS, 3))
    for j in range(3):
        a_test = np.zeros(3)
        a_test[j] = 1.0
        J[:, j] = _get_monomial_coeffs(*a_test)
        
    # Now compute gradients of m0, m2, n0, n2 w.r.t a.
    # dm0/da = 2 * J^T (M0_x @ p)
    # dm2/da = 2 * J^T (M2_x @ p)
    # dn0/da = 2 * J^T (M0_k_mat @ p)
    # dn2/da = 2 * J^T (M2_k_mat @ p)
    
    M0_x_p = M0_x @ p
    M2_x_p = M2_x @ p
    M0_k_p = M0_k_mat @ p
    M2_k_p = M2_k_mat @ p
    
    dm0 = 2.0 * (J.T @ M0_x_p)
    dm2 = 2.0 * (J.T @ M2_x_p)
    dn0 = 2.0 * (J.T @ M0_k_p)
    dn2 = 2.0 * (J.T @ M2_k_p)
    
    dvm2 = (dm2 * m0 - m2 * dm0) / (m0 * m0)
    dvn2 = (dn2 * n0 - n2 * dn0) / (n0 * n0)
    
    grad = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(x):
    return _bound_and_grad(x)[0]


def construct():
    best_val = np.inf
    best_x = np.array([0.0, 0.0, 0.0])
    
    # 1. Initial Heuristic Starts
    heuristic_starts = [
        np.array([-1.13, 0.04, 1.0 / 3240.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0 / 3240.0]),
        np.array([-1.5, 0.5, 0.0]),
        np.array([-0.5, -0.5, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
    ]
    
    for start in heuristic_starts:
        if time.time() >= _DEADLINE - 10:
            break
        try:
            res = minimize(
                _bound_and_grad,
                start,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    # 2. Sobol Sampling for Global Search
    # Use Sobol sequence to cover the space [-5, 5]^3
    # We will run this if we have enough time.
    # Each local optimization takes ~0.05-0.1s.
    # 180s / 0.1s = 1800 iterations.
    # Let's aim for 200-500 points depending on time.
    
    if time.time() < _DEADLINE - 30:
        # Determine number of points based on remaining time
        # Estimate time per point
        t_start = time.time()
        try:
            test_res = minimize(_bound_and_grad, np.array([0.1, 0.1, 0.1]), method='L-BFGS-B', jac=True, options={'maxiter': 100})
            t_point = time.time() - t_start
            if t_point <= 0: t_point = 0.01
        except:
            t_point = 0.05
            
        # Reserve 20s for final polish
        time_left = _DEADLINE - time.time() - 20
        n_points = int(time_left / t_point)
        n_points = max(10, min(n_points, 1000)) # Cap at 1000 to be safe
        
        # Generate Sobol points
        sampler = qmc.Sobol(d=3, scramble=True, seed=42)
        samples = sampler.random(n_points)
        
        # Scale to [-5, 5]
        samples = samples * 10.0 - 5.0
        
        for i in range(n_points):
            if time.time() >= _DEADLINE - 20:
                break
            
            start = samples[i]
            try:
                res = minimize(
                    _bound_and_grad,
                    start,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except Exception:
                pass

    # 3. Final Polish on the best candidate
    if time.time() < _DEADLINE - 2:
        try:
            # Use Trust-Region Constrained for high precision
            # Or just L-BFGS-B with very tight tolerance
            res = minimize(
                _bound_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    return best_x
