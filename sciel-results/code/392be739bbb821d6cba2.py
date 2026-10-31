import numpy as np
import time
from scipy.optimize import minimize, root
from scipy.linalg import eigh

_DEADLINE = time.time() + 195

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
# I[0] = sqrt(pi)
# I[n] = I[n-1] * (2n-1)/(2n)
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Coefficients for Hermite polynomials in power basis x^{2k}
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# p_k is the coefficient of x^{2k} in P(x)
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# a12 = -(a0 + 3*a4 + 105*a8) / 945

# Matrix A such that p = A * a, where a = [a0, a4, a8]
# p has 7 components (k=0..6)
# Row k of A corresponds to coefficient of x^{2k}

# H0 coeffs: [1, 0, 0, 0, 0, 0, 0]
# H4 coeffs: [3, -6, 1, 0, 0, 0, 0] (order x^0, x^2, x^4)
# H8 coeffs: [105, -420, 210, -28, 1, 0, 0]
# H12 coeffs: [945, -7560, 13860, -6930, 1155, -66, 1]

H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 is linear in a: a12 = c12_0*a0 + c12_4*a4 + c12_8*a8
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# p = a0*H0_c + a4*H4_c + a8*H8_c + a12*H12_c
# p = a0*(H0_c + c12_0*H12_c) + a4*(H4_c + c12_4*H12_c) + a8*(H8_c + c12_8*H12_c)

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

# For g(x) = P(x) * exp(-x^2/2), the Fourier transform is:
# G(xi) = sqrt(2*pi) * P(-i*xi) * exp(-xi^2/2)
# So |G(xi)|^2 involves P(-i*xi)^2 * exp(-xi^2)
# The moments in xi-space can be expressed in terms of p_k with alternating signs
# This is already captured by q_k = p_k * (-1)^k in the current code

def _get_p(a):
    return A_mat @ a

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
    """
    p = _get_p(a)
    
    # q_k = p_k * (-1)^k
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    # Build moment matrices for efficient computation
    # Mx0 = p^T @ Ix0 @ p, where Ix0[i,j] = I[i+j]
    # Mx1 = p^T @ Ix1 @ p, where Ix1[i,j] = I[i+j+1]
    # Similarly for Mxi with q
    
    # Precompute the moment matrices (they are constant)
    Ix0 = np.zeros((7, 7))
    Ix1 = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            Ix0[i, j] = I[i + j]
            Ix1[i, j] = I[i + j + 1]
    
    Mx0 = p @ Ix0 @ p
    Mx1 = p @ Ix1 @ p
    Mxi0 = q @ Ix0 @ q
    Mxi1 = q @ Ix1 @ q
    
    if Mx0 <= 1e-14 or Mxi0 <= 1e-14:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-14 or vxi <= 1e-14:
        return 1e18, np.zeros(3)
        
    prod = vx * vxi
    if prod <= 0:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(prod)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation using matrix derivatives
    # dMx0/da = 2 * A_mat^T @ Ix0 @ p
    # dMx1/da = 2 * A_mat^T @ Ix1 @ p
    # dMxi0/da = 2 * A_mat^T @ Ix0 @ q
    # dMxi1/da = 2 * A_mat^T @ Ix1 @ q
    
    dMx0 = 2.0 * A_mat.T @ (Ix0 @ p)
    dMx1 = 2.0 * A_mat.T @ (Ix1 @ p)
    dMxi0 = 2.0 * A_mat.T @ (Ix0 @ q)
    dMxi1 = 2.0 * A_mat.T @ (Ix1 @ q)
    
    # dvx = (dMx1*Mx0 - Mx1*dMx0) / Mx0^2
    inv_Mx0_sq = 1.0 / (Mx0 * Mx0)
    inv_Mxi0_sq = 1.0 / (Mxi0 * Mxi0)
    
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) * inv_Mx0_sq
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) * inv_Mxi0_sq
        
    # d(bound) = (1/(2*bound)) * (vxi*dvx + vx*dvxi)
    factor = 0.5 / bound
    grad = factor * (vxi * dvx + vx * dvxi)
    
    if not np.all(np.isfinite(grad)):
        return bound, np.zeros(3)
        
    return bound, grad

def _stationary_equations(a):
    """
    Returns the gradient of F = Vx * Vxi.
    We want to find roots of this system.
    """
    p = _get_p(a)
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
    # Build moment matrices
    Ix0 = np.zeros((7, 7))
    Ix1 = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            Ix0[i, j] = I[i + j]
            Ix1[i, j] = I[i + j + 1]
    
    Mx0 = p @ Ix0 @ p
    Mx1 = p @ Ix1 @ p
    Mxi0 = q @ Ix0 @ q
    Mxi1 = q @ Ix1 @ q
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    dMx0 = 2.0 * A_mat.T @ (Ix0 @ p)
    dMx1 = 2.0 * A_mat.T @ (Ix1 @ p)
    dMxi0 = 2.0 * A_mat.T @ (Ix0 @ q)
    dMxi1 = 2.0 * A_mat.T @ (Ix1 @ q)
    
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) / (Mx0 * Mx0)
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) / (Mxi0 * Mxi0)
        
    # Gradient of F = Vx * Vxi
    # dF = Vxi * dVx + Vx * dVxi
    grad_F = vxi * dvx + vx * dvxi
    
    return grad_F

def _is_local_min(a):
    """
    Check if a is a local minimum by checking Hessian positive definiteness.
    Approximate Hessian using finite differences of the gradient.
    """
    eps = 1e-6
    grad0 = _stationary_equations(a)
    if np.any(np.abs(grad0) > 1e-4):
        return False
        
    hess = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            a_plus = a.copy()
            a_minus = a.copy()
            a_plus[i] += eps
            a_minus[i] -= eps
            if i == j:
                hess[i, j] = (_stationary_equations(a_plus)[i] - _stationary_equations(a_minus)[i]) / (2*eps)
            else:
                # Cross derivatives
                a_pp = a.copy(); a_pm = a.copy(); a_mp = a.copy(); a_mm = a.copy()
                a_pp[i] += eps; a_pp[j] += eps
                a_pm[i] += eps; a_pm[j] -= eps
                a_mp[i] -= eps; a_mp[j] += eps
                a_mm[i] -= eps; a_mm[j] -= eps
                hess[i, j] = (_stationary_equations(a_pp)[i] - _stationary_equations(a_pm)[i] - _stationary_equations(a_mp)[i] + _stationary_equations(a_mm)[i]) / (4*eps*eps)
                
    try:
        eigvals = np.linalg.eigvalsh(hess)
        return np.all(eigvals > -1e-6)
    except:
        return False

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    # Seed value
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 5  # seconds
    
    # Generate initial guesses for root finding
    starts = []
    
    # Grid-based starts - wider range
    a0_range = np.linspace(-3, 2, 8)
    a4_range = np.linspace(-0.5, 0.5, 8)
    a8_range = np.linspace(-0.1, 0.1, 8)
    
    for a0 in a0_range:
        for a4 in a4_range:
            for a8 in a8_range:
                starts.append(np.array([a0, a4, a8]))
                
    # Random starts - more extensive
    rng = np.random.default_rng(42)
    for _ in range(100):
        a0 = rng.uniform(-4, 3)
        a4 = rng.uniform(-1.0, 1.0)
        a8 = rng.uniform(-0.2, 0.2)
        starts.append(np.array([a0, a4, a8]))
        
    # Include seed
    starts.append(seed_params)
    
    # Solve stationary point equations for each start
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            # Use hybr (MINPACK) which is robust for small systems
            res = root(_stationary_equations, start, method='hybr', tol=1e-12)
            
            if res.success:
                a_sol = res.x
                # Verify it's a valid point
                val, _ = _compute_bound_and_grad(a_sol)
                
                if np.isfinite(val) and val < best_val:
                    # Polish with L-BFGS-B to ensure it's a minimum
                    if time.time() < _DEADLINE - deadline_margin:
                        res_min = minimize(_compute_bound_and_grad, a_sol, method='L-BFGS-B', jac=True, options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-12})
                        if res_min.success and res_min.fun < val:
                            val = res_min.fun
                            a_sol = res_min.x
                            
                    if val < best_val:
                        best_val = val
                        best_params = a_sol.copy()
        except Exception:
            pass

    # Final high-precision polish with multiple passes
    if time.time() < _DEADLINE - deadline_margin:
        try:
            # First pass: standard L-BFGS-B
            res_min = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if res_min.success and res_min.fun < best_val:
                best_val = res_min.fun
                best_params = res_min.x
        except Exception:
            pass
        
        # Second pass: even tighter tolerance
        if time.time() < _DEADLINE - deadline_margin:
            try:
                res_min2 = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B', jac=True, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
                if res_min2.success and res_min2.fun < best_val:
                    best_val = res_min2.fun
                    best_params = res_min2.x
            except Exception:
                pass
        
        # Third pass: try from slightly perturbed points to escape numerical plateaus
        if time.time() < _DEADLINE - 2 * deadline_margin:
            rng_polish = np.random.default_rng(123)
            for _ in range(10):
                if time.time() > _DEADLINE - deadline_margin:
                    break
                try:
                    perturbation = rng_polish.normal(0, 1e-7, size=3)
                    start_polish = best_params + perturbation
                    res_polish = minimize(_compute_bound_and_grad, start_polish, method='L-BFGS-B', jac=True, options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
                    if res_polish.success and res_polish.fun < best_val:
                        best_val = res_polish.fun
                        best_params = res_polish.x
                except Exception:
                    pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
