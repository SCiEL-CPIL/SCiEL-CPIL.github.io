import numpy as np
import time
from scipy.optimize import root, minimize

_DEADLINE = time.time() + 205  # safety margin before 210s kill

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
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

H0_c = np.array([1.0, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3.0, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105.0, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945.0, -7560, 13860, -6930, 1155, -66, 1])

# a12 is linear in a: a12 = c12_0*a0 + c12_4*a4 + c12_8*a8
c12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

col0 = H0_c + c12[0]*H12_c
col1 = H4_c + c12[1]*H12_c
col2 = H8_c + c12[2]*H12_c

A_mat = np.column_stack((col0, col1, col2)) # Shape (7, 3)

sign_vec = np.array([1.0, -1, 1, -1, 1, -1, 1])

def _get_p(a):
    return A_mat @ a

def _compute_variances(a):
    """
    Compute Vx and Vxi.
    Returns (Vx, Vxi, p, q)
    """
    p = _get_p(a)
    q = p * sign_vec
    
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
            idx0 = i + j
            idx1 = i + j + 1
            I0 = I[idx0]
            I1 = I[idx1]
            
            Mx0 += pi * p[j] * I0
            Mx1 += pi * p[j] * I1
            
            Mxi0 += qi * q[j] * I0
            Mxi1 += qi * q[j] * I1
            
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return 1e18, 1e18, p, q
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, 1e18, p, q
        
    return vx, vxi, p, q

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound J = sqrt(Vx * Vxi) and its gradient.
    """
    vx, vxi, p, q = _compute_variances(a)
    
    if vx >= 1e17 or vxi >= 1e17:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    Mx0_calc = np.dot(p, Sx[0])
    Mx1_calc = np.dot(p, Sx[1])
    Mxi0_calc = np.dot(q, Sxi[0])
    Mxi1_calc = np.dot(q, Sxi[1])
    
    if Mx0_calc <= 1e-15 or Mxi0_calc <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx_calc = Mx1_calc / Mx0_calc
    vxi_calc = Mxi1_calc / Mxi0_calc
    
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx0_calc - Mx1_calc*dMx[0, m]) / (Mx0_calc*Mx0_calc)
        dvxi[m] = (dMxi[1, m]*Mxi0_calc - Mxi1_calc*dMxi[0, m]) / (Mxi0_calc*Mxi0_calc)
        
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    for m in range(3):
        grad[m] = factor * (vxi_calc*dvx[m] + vx_calc*dvxi[m])
        
    return bound, grad

def _objective(a):
    return _compute_bound_and_grad(a)[0]

def _objective_grad(a):
    return _compute_bound_and_grad(a)[1]

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
    
    deadline_margin = 15  # seconds
    
    # Generate diverse starting points
    np.random.seed(42)
    starts = [seed_params.copy()]
    
    # Random starts with various scales
    for scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for _ in range(20):
            start = np.random.uniform(-scale, scale, 3)
            starts.append(start)
    
    # Structured starts around the seed
    for da0 in [-0.5, -0.1, 0.1, 0.5]:
        for da4 in [-0.05, -0.01, 0.01, 0.05]:
            for da8 in [-0.001, -0.0001, 0.0001, 0.001]:
                starts.append(seed_params + np.array([da0, da4, da8]))
    
    # More aggressive structured starts
    for a0 in [-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3]:
        for a4 in [-0.2, -0.1, 0, 0.1, 0.2]:
            for a8 in [-0.01, -0.005, 0, 0.005, 0.01]:
                starts.append(np.array([a0, a4, a8]))
    
    # Sort starts by initial objective value to prioritize promising regions
    def _init_score(s):
        v, _ = _compute_bound_and_grad(s)
        return v
    
    starts.sort(key=_init_score)
    
    # First pass: L-BFGS-B from top candidates
    for start in starts[:50]:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            res_local = minimize(_objective, start, 
                                 jac=_objective_grad, 
                                 method='L-BFGS-B', 
                                 options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-13})
            
            if np.isfinite(res_local.fun) and res_local.fun < best_val:
                best_val = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass
    
    # Second pass: root finding on stationary equations for better convergence
    for start in starts[:30]:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            # Use the current best as a guide
            if np.linalg.norm(start - best_params) > 5.0:
                continue
                
            res = root(_objective_grad, start, method='hybrid', tol=1e-14)
            
            if res.success:
                val, grad = _compute_bound_and_grad(res.x)
                if np.isfinite(val) and val < best_val and np.linalg.norm(grad) < 1e-10:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish with very tight tolerances
    if time.time() < _DEADLINE - deadline_margin and best_val < seed_val:
        try:
            res_polish = minimize(_objective, best_params, 
                                  jac=_objective_grad, 
                                  method='L-BFGS-B', 
                                  options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-15})
            if np.isfinite(res_polish.fun) and res_polish.fun < best_val:
                best_val = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass
        
        # Additional root-finding polish
        try:
            res_root_polish = root(_objective_grad, best_params, method='hybrid', tol=1e-15)
            if res_root_polish.success:
                val, grad = _compute_bound_and_grad(res_root_polish.x)
                if np.isfinite(val) and val < best_val and np.linalg.norm(grad) < 1e-12:
                    best_val = val
                    best_params = res_root_polish.x.copy()
        except Exception:
            pass
            
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
