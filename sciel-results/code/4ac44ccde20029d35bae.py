"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import eigh
import time

_DEADLINE = 205.0  # safety margin before 210s kill
_START_TIME = None

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _hermite_poly(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def _compute_bound(coeffs):
    """
    Compute the upper bound on A(f)*A(F) for the test function
    g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2)
    where c12 is fixed so that g(0) = 0.
    
    The bound is C4 * sqrt(Var(x) * Var(p)) where C4 is the constant
    from the uncertainty principle applied to this specific form.
    
    For a wavefunction psi(x) = g(x) / ||g||, we have:
    - <x> = 0 (odd moments vanish for even Hermite polynomials)
    - <p> = 0 (same reason)
    - <x^2> = sum of contributions
    - <p^2> = sum of contributions
    
    The Fourier transform of H_n(x)*exp(-x^2/2) is (-i)^n H_n(x)*exp(-x^2/2).
    
    For the bound on A(f)*A(F), we use:
    A(f) = sqrt(<x^2> - <x>^2) = sqrt(<x^2>)
    A(F) = sqrt(<p^2> - <p>^2) = sqrt(<p^2>)
    
    The product is sqrt(<x^2> * <p^2>).
    
    We need to normalize: let N = <g|g> = integral |g|^2 dx
    Then <x^2> = <g|x^2|g> / N, <p^2> = <g|p^2|g> / N
    
    So the bound = sqrt(<g|x^2|g> * <g|p^2|g>) / N
    """
    c0, c4, c8 = coeffs
    # g(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0) = 1
    # H4(0) = -12
    # H8(0) = 1008
    # H12(0) = -103968
    # So: c0 - 12*c4 + 1008*c8 - 103968*c12 = 0
    # c12 = (c0 - 12*c4 + 1008*c8) / 103968
    c12 = (c0 - 12 * c4 + 1008 * c8) / 103968.0
    
    coeffs_all = np.array([c0, c4, c8, c12])
    degrees = [0, 4, 8, 12]
    
    # Compute integrals using the orthogonality of Hermite polynomials
    # integral H_n(x) H_m(x) exp(-x^2) dx = sqrt(pi) * 2^n * n! * delta_nm
    # But our weight is exp(-x^2/2), so let's change variables.
    # Actually, let's use the standard result:
    # integral H_n(x) H_m(x) exp(-x^2/2) dx is not standard orthogonality.
    
    # Let me reconsider. The standard Hermite polynomials are orthogonal with
    # weight exp(-x^2). Our function is H_n(x) * exp(-x^2/2).
    
    # Let's compute the integrals numerically for accuracy.
    x = np.linspace(-20, 20, 200000)
    dx = x[1] - x[0]
    weight = np.exp(-x**2 / 2)
    
    # Build the function g(x)
    g = np.zeros_like(x)
    for c, n in zip(coeffs_all, degrees):
        g += c * _hermite_poly(n, x) * weight
    
    # Normalization
    N = np.sum(g**2) * dx
    
    # <x^2> integral
    x2_g = np.sum(x**2 * g**2) * dx
    
    # <p^2> integral: p = -i d/dx, so <p^2> = integral |g'(x)|^2 dx
    # g'(x) = sum c_n * [H_n'(x) * weight + H_n(x) * weight']
    # H_n'(x) = 2n H_{n-1}(x)
    # weight' = -x * weight
    # So g'(x) = weight * sum c_n * [2n H_{n-1}(x) - x H_n(x)]
    
    gp = np.zeros_like(x)
    for c, n in zip(coeffs_all, degrees):
        if n > 0:
            gp += c * (2 * n * _hermite_poly(n - 1, x) - x * _hermite_poly(n, x))
        else:
            gp += c * (-x * _hermite_poly(0, x))
    gp *= weight
    
    p2_g = np.sum(gp**2) * dx
    
    # The bound is sqrt(<x^2> * <p^2>) / N where <x^2> = x2_g/N, <p^2> = p2_g/N
    # So bound = sqrt((x2_g/N) * (p2_g/N)) = sqrt(x2_g * p2_g) / N
    
    if N <= 0:
        return np.inf
    
    bound = np.sqrt(x2_g * p2_g) / N
    return bound

def construct():
    """
    Construct the optimal coefficients for the Hermite combination.
    
    We formulate this as a generalized eigenvalue problem. The objective is to minimize
    sqrt(<x^2> * <p^2>) / <1> where <1> = <g|g>.
    
    Let's define the matrices:
    M_xx[i,j] = integral H_{d_i}(x) H_{d_j}(x) x^2 exp(-x^2) dx
    M_pp[i,j] = integral H'_{d_i}(x) H'_{d_j}(x) exp(-x^2) dx  (for p^2)
    M_00[i,j] = integral H_{d_i}(x) H_{d_j}(x) exp(-x^2) dx
    
    Wait, our weight is exp(-x^2/2), not exp(-x^2). Let me redo this.
    
    Actually, let me use the constraint g(0) = 0 to eliminate c12, and then
    optimize over c0, c4, c8.
    
    The objective is: minimize f(c0, c4, c8) = sqrt(<x^2> * <p^2>) / <1>
    
    This is a ratio of quadratic forms. We can use the generalized eigenvalue approach.
    
    Let's define vectors and matrices properly.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Degrees of freedom: c0, c4, c8 (c12 is determined by constraint)
    degrees = [0, 4, 8]
    n_basis = 3
    
    # We need to compute:
    # A = <g|x^2|g> = sum_{i,j} c_i c_j M_xx[i,j]
    # B = <g|p^2|g> = sum_{i,j} c_i c_j M_pp[i,j]  
    # C = <g|g> = sum_{i,j} c_i c_j M_00[i,j]
    # where g = sum c_i H_{d_i}(x) exp(-x^2/2) + c12 H_12(x) exp(-x^2/2)
    # and c12 = (c0 - 12*c4 + 1008*c8) / 103968
    
    # Let's express everything in terms of c0, c4, c8.
    # Let v = [c0, c4, c8]
    # c12 = alpha^T v where alpha = [1/103968, -12/103968, 1008/103968]
    
    # g = sum_{i=0}^{2} v_i H_{d_i}(x) exp(-x^2/2) + (alpha^T v) H_12(x) exp(-x^2/2)
    #   = exp(-x^2/2) * sum_{i=0}^{2} v_i [H_{d_i}(x) + alpha_i H_12(x)]
    
    # Define phi_i(x) = H_{d_i}(x) + alpha_i H_12(x) for i = 0, 1, 2
    # Then g = exp(-x^2/2) * sum v_i phi_i(x)
    
    # M_00[i,j] = integral phi_i(x) phi_j(x) exp(-x^2) dx
    # M_xx[i,j] = integral x^2 phi_i(x) phi_j(x) exp(-x^2) dx
    # M_pp[i,j] = integral phi_i'(x) phi_j'(x) exp(-x^2) dx
    
    # Wait, the weight in the integral for normalization is exp(-x^2/2)^2 = exp(-x^2).
    # So yes, weight is exp(-x^2).
    
    # Let's compute these matrices numerically.
    x = np.linspace(-15, 15, 100000)
    dx = x[1] - x[0]
    w = np.exp(-x**2)
    
    alpha = np.array([1.0/103968.0, -12.0/103968.0, 1008.0/103968.0])
    
    # Compute phi_i(x)
    H12 = _hermite_poly(12, x)
    phi = np.zeros((n_basis, len(x)))
    for i, d in enumerate(degrees):
        phi[i] = _hermite_poly(d, x) + alpha[i] * H12
    
    # Compute phi_i'(x) = H_{d_i}'(x) + alpha_i H_12'(x)
    # H_n'(x) = 2n H_{n-1}(x)
    phi_prime = np.zeros((n_basis, len(x)))
    for i, d in enumerate(degrees):
        if d > 0:
            phi_prime[i] = 2 * d * _hermite_poly(d - 1, x) + alpha[i] * 2 * 12 * _hermite_poly(11, x)
        else:
            phi_prime[i] = 0.0 + alpha[i] * 2 * 12 * _hermite_poly(11, x)
    
    # Build matrices
    M_00 = np.zeros((n_basis, n_basis))
    M_xx = np.zeros((n_basis, n_basis))
    M_pp = np.zeros((n_basis, n_basis))
    
    for i in range(n_basis):
        for j in range(n_basis):
            M_00[i, j] = np.sum(phi[i] * phi[j] * w) * dx
            M_xx[i, j] = np.sum(x**2 * phi[i] * phi[j] * w) * dx
            M_pp[i, j] = np.sum(phi_prime[i] * phi_prime[j] * w) * dx
    
    # Symmetrize
    M_00 = (M_00 + M_00.T) / 2
    M_xx = (M_xx + M_xx.T) / 2
    M_pp = (M_pp + M_pp.T) / 2
    
    # The objective is: minimize sqrt(v^T M_xx v * v^T M_pp v) / (v^T M_00 v)
    
    # This is not a standard generalized eigenvalue problem directly, but we can
    # use iterative optimization. Let's use scipy.optimize.minimize.
    
    from scipy.optimize import minimize
    
    def objective(v):
        v = np.array(v)
        A = v @ M_xx @ v
        B = v @ M_pp @ v
        C = v @ M_00 @ v
        if C <= 0 or A <= 0 or B <= 0:
            return 1e10
        return np.sqrt(A * B) / C
    
    # Initial guess: the prior best
    v0 = np.array([-113/100, 1/25, 1/3240])
    
    # Try multiple initial points
    best_v = v0.copy()
    best_val = objective(v0)
    
    # Try some other initial points
    np.random.seed(42)
    for trial in range(20):
        if _time_left() < 30:
            break
        v_init = np.random.randn(n_basis) * 0.1 + v0
        res = minimize(objective, v_init, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_v = res.x
    
    # Polish with BFGS
    if _time_left() > 10:
        res = minimize(objective, best_v, method='BFGS',
                      options={'maxiter': 1000, 'gtol': 1e-12})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_v = res.x
    
    # Final polish with L-BFGS-B
    if _time_left() > 5:
        res = minimize(objective, best_v, method='L-BFGS-B',
                      options={'maxiter': 5000, 'gtol': 1e-14, 'ftol': 1e-16})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_v = res.x
    
    return np.array(best_v)
# EVOLVE-BLOCK-END
