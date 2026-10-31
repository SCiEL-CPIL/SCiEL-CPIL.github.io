import numpy as np
import time
from scipy.optimize import minimize, root

_DEADLINE = time.time() + 200

# Precompute I[n] = integral_{-inf}^{inf} x^{2n} exp(-x^2) dx
I = np.zeros(25)
I[0] = np.sqrt(np.pi)
for n in range(1, 25):
    I[n] = I[n-1] * (2.0*n - 1.0) / (2.0*n)

# Hermite polynomial coefficients in power basis x^{2k}
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# We express g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
# Constraint g(0) = 0 implies a0*1 + a4*3 + a8*105 + a12*945 = 0
# So a12 = -(a0 + 3*a4 + 105*a8) / 945

# Let P(x) = sum_{k=0}^6 p_k x^{2k}
# p_k = a0 * (H0 coeff) + a4 * (H4 coeff) + a8 * (H8 coeff) + a12 * (H12 coeff)

# Coefficients of H_n in basis x^{2k} for k=0..6
H0_c = np.array([1, 0, 0, 0, 0, 0, 0])
H4_c = np.array([3, -6, 1, 0, 0, 0, 0])
H8_c = np.array([105, -420, 210, -28, 1, 0, 0])
H12_c = np.array([945, -7560, 13860, -6930, 1155, -66, 1])

# a12 = -1/945 * a0 - 3/945 * a4 - 105/945 * a8
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
    
    # q_k = p_k * (-1)^k (for Fourier transform side)
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

def _compute_bound_and_grad(a):
    """
    Compute the uncertainty bound and its analytic gradient.
    a: array of shape (3,)
    Returns: (bound, grad)
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
        return 1e18, np.zeros(3)
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
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
            
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx0 - Mx1*dMx[0, m]) / (Mx0*Mx0)
        dvxi[m] = (dMxi[1, m]*Mxi0 - Mxi1*dMxi[0, m]) / (Mxi0*Mxi0)
        
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    for m in range(3):
        grad[m] = factor * (vxi*dvx[m] + vx*dvxi[m])
        
    return bound, grad

def _stationary_equations(a):
    """
    Returns the gradient of the squared objective (Vx * Vxi).
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
            
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx0 - Mx1*dMx[0, m]) / (Mx0*Mx0)
        dvxi[m] = (dMxi[1, m]*Mxi0 - Mxi1*dMxi[0, m]) / (Mxi0*Mxi0)
        
    # Gradient of Vx*Vxi
    # d/dx (Vx*Vxi) = Vxi * dVx + Vx * dVxi
    grad_sq = vxi * dvx + vx * dvxi
    
    return grad_sq

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
    
    # 1. Try root finding for stationary points from multiple starts
    # The optimal point is a stationary point of the objective.
    # Using scipy.optimize.root (HYBRD or LM) is often more robust for this.
    
    starts = [
        seed_params,
        np.array([-1.0, 0.05, 0.001]),
        np.array([-2.0, 0.02, 0.0005]),
        np.array([-0.5, 0.08, 0.002]),
        np.array([-1.5, 0.01, 0.0001]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.13, 0.04, 0.000308642]), # seed again
        np.array([-1.13, 0.04, 0.000308642]) * 1.1,
        np.array([-1.13, 0.04, 0.000308642]) * 0.9,
    ]
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(10):
        s = rng.normal(0, 1, 3)
        s[0] = s[0] * 1.0 - 0.5 # bias towards negative a0
        s[1] = s[1] * 0.1
        s[2] = s[2] * 0.005
        starts.append(s)
        
    # Also try log-space transformed starts to explore scale
    # a0 ~ 1, a4 ~ 0.01, a8 ~ 0.001
    # Let a0 = exp(u0), a4 = exp(u1), a8 = exp(u2) ? No, signs matter.
    # Just use diverse linear starts.
    
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            # Use HYBRD for robustness
            res = root(_stationary_equations, start, method='hybrd', tol=1e-12)
            if res.success:
                val = _compute_bound(res.x)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            elif hasattr(res, 'x') and res.x is not None:
                # Sometimes root fails but converges to a good point
                val = _compute_bound(res.x)
                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass

    # 2. Local polish with L-BFGS-B from the best point found
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_compute_bound, best_params, method='L-BFGS-B',
                          jac=lambda a: _compute_bound_and_grad(a)[1],
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # 3. Try a few more local starts from the vicinity of the best point
    if time.time() < _DEADLINE - deadline_margin:
        for _ in range(20):
            if time.time() > _DEADLINE - deadline_margin:
                break
            # Perturb the best solution
            noise = rng.normal(0, 0.1, 3)
            noise[2] *= 0.01 # a8 is smaller
            a_start = best_params + noise
            
            try:
                res = minimize(_compute_bound, a_start, method='L-BFGS-B',
                              jac=lambda a: _compute_bound_and_grad(a)[1],
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass

    # Final high-precision polish
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_compute_bound, best_params, method='L-BFGS-B',
                          jac=lambda a: _compute_bound_and_grad(a)[1],
                          options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
