import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, root

_DEADLINE = time.time() + 200

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
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

def _get_p(a):
    return A_mat @ a

def _compute_bound(a):
    """Compute the uncertainty bound for given coefficients a = [a0, a4, a8]."""
    p = _get_p(a)
    
    # q_k = p_k * (-1)^k
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
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
        return 1e18
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18
        
    return bound

def _stationary_equations(a):
    """
    Computes the gradient of the squared objective (Vx * Vxi).
    F(a) = Vxi * grad_Vx + Vx * grad_Vxi = 0
    This is the condition for stationary points of the product.
    """
    p = _get_p(a)
    q = p * np.array([1, -1, 1, -1, 1, -1, 1])
    
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
            
    # grad Vx = (dMx1 * Mx0 - Mx1 * dMx0) / Mx0^2
    # grad Vxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) / Mxi0^2
    
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
    seed_val = _compute_bound(seed_params)
    
    best_val = seed_val
    best_params = seed_params.copy()
    
    deadline_margin = 10  # seconds
    
    # List of initial guesses to try for the root finder
    # 1. The seed
    # 2. Perturbations of the seed
    # 3. Some heuristic guesses
    initial_guesses = [
        seed_params,
        seed_params + np.array([0.01, 0.01, 0.0001]),
        seed_params - np.array([0.01, 0.01, 0.0001]),
        seed_params + np.array([-0.01, 0.01, 0.0001]),
        seed_params + np.array([0.01, -0.01, 0.0001]),
        np.array([-1.0, 0.05, 0.0003]),
        np.array([-1.2, 0.03, 0.0004]),
    ]
    
    for start in initial_guesses:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        # Try Hybrid method
        try:
            res = root(_stationary_equations, start, method='hybr', tol=1e-12)
            if res.success and np.all(np.isfinite(res.x)):
                val = _compute_bound(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
        
        # Try Levenberg-Marquardt method
        if time.time() > _DEADLINE - deadline_margin:
            break
        try:
            res = root(_stationary_equations, start, method='lm', tol=1e-12)
            if res.success and np.all(np.isfinite(res.x)):
                val = _compute_bound(res.x)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
