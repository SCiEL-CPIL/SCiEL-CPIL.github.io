import numpy as np
import time
from scipy.optimize import minimize

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
    
    # Compute moments M_x[n] and M_xi[n] for n=0,1
    # M[n] = sum_{i,j} p_i p_j I[i+j+n]
    # We can compute this efficiently using the I array
    
    Mx = np.zeros(2)
    Mxi = np.zeros(2)
    
    for i in range(7):
        pi = p[i]
        if pi == 0: continue
        for j in range(7):
            idx0 = i + j
            idx1 = i + j + 1
            Mx[0] += pi * p[j] * I[idx0]
            Mx[1] += pi * p[j] * I[idx1]
            
            idx0_q = i + j
            idx1_q = i + j + 1
            Mxi[0] += q[i] * q[j] * I[idx0_q]
            Mxi[1] += q[i] * q[j] * I[idx1_q]
            
    if Mx[0] <= 1e-15 or Mxi[0] <= 1e-15:
        return 1e18, np.zeros(3)
        
    vx = Mx[1] / Mx[0]
    vxi = Mxi[1] / Mxi[0]
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return 1e18, np.zeros(3)
        
    bound = np.sqrt(vx * vxi)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
        
    # Gradient computation
    # J = 0.5 * (vx * vxi)^0.5
    # dJ = 0.5 * (vx * vxi)^-0.5 * (d(vx*vxi))
    # d(vx*vxi) = vxi * dvx + vx * dvxi
    # dvx = d(Mx1/Mx0) = (dMx1*Mx0 - Mx1*dMx0) / Mx0^2
    # dvxi = d(Mxi1/Mxi0) = (dMxi1*Mxi0 - Mxi1*dMxi0) / Mxi0^2
    
    # dM[n] = 2 * sum_{i,j} (dp_i * p_j) I[i+j+n]
    # dp_i = sum_m A[i,m] * da_m
    # So dM[n]_m = 2 * sum_i A[i,m] sum_j p_j I[i+j+n]
    
    # Let S[n]_i = sum_j p_j I[i+j+n]
    Sx = np.zeros((2, 7))
    Sxi = np.zeros((2, 7))
    
    for n in range(2):
        for i in range(7):
            Sx[n, i] = np.dot(p, I[i+n:i+n+7])
            Sxi[n, i] = np.dot(q, I[i+n:i+n+7])
            
    # dM[n] is a vector of 3 components (for each a_m)
    # dM[n]_m = 2 * sum_i A[i,m] S[n, i]
    dMx = np.zeros((2, 3))
    dMxi = np.zeros((2, 3))
    
    for m in range(3):
        for n in range(2):
            dMx[n, m] = 2.0 * np.dot(A_mat[:, m], Sx[n, :])
            dMxi[n, m] = 2.0 * np.dot(A_mat[:, m], Sxi[n, :])
            
    # dvx_m = (dMx[1,m]*Mx[0] - Mx[1]*dMx[0,m]) / Mx[0]^2
    dvx = np.zeros(3)
    dvxi = np.zeros(3)
    
    for m in range(3):
        dvx[m] = (dMx[1, m]*Mx[0] - Mx[1]*dMx[0, m]) / (Mx[0]*Mx[0])
        dvxi[m] = (dMxi[1, m]*Mxi[0] - Mxi[1]*dMxi[0, m]) / (Mxi[0]*Mxi[0])
        
    # d(bound) = (1/(2*bound)) * (vxi*dvx + vx*dvxi)
    grad = np.zeros(3)
    factor = 1.0 / (2.0 * bound)
    for m in range(3):
        grad[m] = factor * (vxi*dvx[m] + vx*dvxi[m])
        
    return bound, grad

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
    
    # Generate starting points
    starts = [seed_params.copy()]
    
    # Grid starts
    for a0 in np.linspace(-3, 1, 7):
        for a4 in np.linspace(-0.1, 0.1, 5):
            for a8 in np.linspace(0, 0.01, 5):
                starts.append(np.array([a0, a4, a8]))
                
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(100):
        starts.append(rng.normal(0, 1.0, 3))
    for _ in range(50):
        starts.append(rng.normal(0, 0.5, 3))
        
    # Shuffle
    rng2 = np.random.default_rng(99)
    perm = np.arange(len(starts))
    perm[1:] = rng2.permutation(perm[1:])
    starts = [starts[i] for i in perm]
    
    elapsed = time.time() - _DEADLINE
    deadline_margin = 5 # seconds
    
    # Optimization Loop
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            # Use L-BFGS-B with analytic gradient
            res = minimize(_compute_bound_and_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                # Verify with high precision if close
                val_check, _ = _compute_bound_and_grad(res.x)
                if val_check < best_val:
                    best_val = val_check
                    best_params = res.x.copy()
        except Exception:
            pass

    # Final polish on best found
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(_compute_bound_and_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # One more Nelder-Mead just in case, without gradient (robust)
    if time.time() < _DEADLINE - deadline_margin:
        try:
            res = minimize(lambda x: _compute_bound_and_grad(x)[0], best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
