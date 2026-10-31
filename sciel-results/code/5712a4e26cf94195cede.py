import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, root

_DEADLINE = time.time() + 200

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(30)
I[0] = np.sqrt(np.pi)
for n in range(1, 30):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Hermite polynomial coefficients in power basis x^{2k}
H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -(a0 + 3*a4 + 105*a8) / 945
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2))

signs = np.array([1, -1, 1, -1, 1, -1, 1])

def _get_p(a):
    return A_mat @ a

def _compute_bound_and_grad(a):
    """Compute the uncertainty bound and its gradient for given coefficients a = [a0, a4, a8]."""
    p = _get_p(a)
    q = p * signs
    
    # Compute moments
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            Mx0 += pi * p[j] * I[i+j]
            Mx1 += pi * p[j] * I[i+j+1]
            Mxi0 += qi * q[j] * I[i+j]
            Mxi1 += qi * q[j] * I[i+j+1]
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    
    # Compute gradients
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    grad_vx = np.zeros(3)
    grad_vxi = np.zeros(3)
    
    for m in range(3):
        grad_vx[m] = (dMx[1, m]*Mx0 - Mx1*dMx[0, m]) / (Mx0*Mx0)
        grad_vxi[m] = (dMxi[1, m]*Mxi0 - Mxi1*dMxi[0, m]) / (Mxi0*Mxi0)
        
    grad_bound = (vxi * grad_vx + vx * grad_vxi) / (2.0 * bound)
    
    return bound, grad_bound

def _stationary_equations(a):
    """
    Computes the gradient of the squared objective (Vx * Vxi).
    F(a) = Vxi * grad_Vx + Vx * grad_Vxi = 0
    This is the condition for stationary points of the product.
    """
    p = _get_p(a)
    q = p * signs
    
    Mx0 = 0.0
    Mx1 = 0.0
    Mxi0 = 0.0
    Mxi1 = 0.0
    
    for i in range(7):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        for j in range(7):
            Mx0 += pi * p[j] * I[i+j]
            Mx1 += pi * p[j] * I[i+j+1]
            Mxi0 += qi * q[j] * I[i+j]
            Mxi1 += qi * q[j] * I[i+j+1]
            
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    # Compute gradients of Mx0, Mx1, Mxi0, Mxi1
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    grad_vx = np.zeros(3)
    grad_vxi = np.zeros(3)
    
    for m in range(3):
        grad_vx[m] = (dMx[1, m]*Mx0 - Mx1*dMx[0, m]) / (Mx0*Mx0)
        grad_vxi[m] = (dMxi[1, m]*Mxi0 - Mxi1*dMxi[0, m]) / (Mxi0*Mxi0)
        
    # System: Vxi * grad_vx + Vx * grad_vxi = 0
    F = vxi * grad_vx + vx * grad_vxi
    
    return F

def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the uncertainty product.
    Returns array of shape (3,).
    """
    
    seed_params = np.array([-1.13, 0.04, 0.000308642])
    seed_val, _ = _compute_bound_and_grad(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 15  # seconds
    
    # Generate multiple initial guesses
    initial_guesses = [seed_params]
    
    # Add perturbations around seed
    for scale in [0.001, 0.01, 0.05, 0.1]:
        for dx in [-scale, 0, scale]:
            for dy in [-scale, 0, scale]:
                for dz in [-scale*0.01, 0, scale*0.01]:
                    candidate = seed_params + np.array([dx, dy, dz])
                    initial_guesses.append(candidate)
    
    # Add some heuristic guesses
    initial_guesses.extend([
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
        np.array([-1.1, 0.06, 0.0002]),
        np.array([-1.15, 0.035, 0.00035]),
    ])
    
    # Remove duplicates and invalid points
    unique_guesses = []
    seen = set()
    for g in initial_guesses:
        key = tuple(np.round(g, 6))
        if key not in seen and np.all(np.isfinite(g)):
            seen.add(key)
            unique_guesses.append(g)
    
    # Phase 1: Try gradient-based minimization from each starting point
    for start in unique_guesses:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            res = minimize(_compute_bound_and_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.success and np.all(np.isfinite(res.x)):
                val, _ = _compute_bound_and_grad(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Try root-finding on stationary equations from best point and neighbors
    polish_starts = [best_params]
    for scale in [0.001, 0.01]:
        for dx in [-scale, scale]:
            for dy in [-scale, scale]:
                for dz in [-scale*0.01, scale*0.01]:
                    polish_starts.append(best_params + np.array([dx, dy, dz]))
    
    for start in polish_starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        for method in ['hybr', 'lm']:
            if time.time() > _DEADLINE - deadline_margin:
                break
            try:
                res = root(_stationary_equations, start, method=method, tol=1e-14)
                if res.success and np.all(np.isfinite(res.x)):
                    val, _ = _compute_bound_and_grad(res.x)
                    if val < best_val:
                        best_val = val
                        best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish using tighter tolerance
    if time.time() < _DEADLINE - deadline_margin - 5:
        try:
            res = root(_stationary_equations, best_params, method='hybr', tol=1e-16)
            if res.success and np.all(np.isfinite(res.x)):
                val, _ = _compute_bound_and_grad(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
