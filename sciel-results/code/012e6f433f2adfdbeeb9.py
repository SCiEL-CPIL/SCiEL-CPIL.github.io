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
    
    deadline_margin = 5  # seconds
    
    # Log-magnitude and sign parameterization:
    # a_i = sign_i * exp(m_i)
    # We optimize over (m0, m4, m8, s0, s4, s8) where s_i in {-1, 1}
    # For continuous optimization, we use s_i = tanh(t_i) to make it smooth
    
    def _log_sign_obj(ts):
        """
        ts: array of shape (6,) = [m0, m4, m8, t0, t4, t8]
        where a_i = sign(t_i) * exp(m_i), sign(t) = tanh(t)
        """
        m0, m4, m8 = ts[0], ts[1], ts[2]
        t0, t4, t8 = ts[3], ts[4], ts[5]
        
        s0 = np.tanh(t0)
        s4 = np.tanh(t4)
        s8 = np.tanh(t8)
        
        a0 = s0 * np.exp(m0)
        a4 = s4 * np.exp(m4)
        a8 = s8 * np.exp(m8)
        
        a = np.array([a0, a4, a8])
        val, _ = _compute_bound_and_grad(a)
        return val
    
    def _log_sign_grad(ts):
        """
        Gradient of _log_sign_obj with respect to ts.
        Chain rule: d_obj/dm_i = d_obj/da_i * a_i (since da_i/dm_i = a_i)
                     d_obj/dt_i = d_obj/da_i * s_i * exp(m_i) * (1 - s_i^2)
        """
        m0, m4, m8 = ts[0], ts[1], ts[2]
        t0, t4, t8 = ts[3], ts[4], ts[5]
        
        s0 = np.tanh(t0)
        s4 = np.tanh(t4)
        s8 = np.tanh(t8)
        
        a0 = s0 * np.exp(m0)
        a4 = s4 * np.exp(m4)
        a8 = s8 * np.exp(m8)
        
        a = np.array([a0, a4, a8])
        val, grad_a = _compute_bound_and_grad(a)
        
        grad_ts = np.zeros(6)
        # d_obj/dm_i = grad_a[i] * a_i
        grad_ts[0] = grad_a[0] * a0
        grad_ts[1] = grad_a[1] * a4
        grad_ts[2] = grad_a[2] * a8
        
        # d_obj/dt_i = grad_a[i] * s_i * exp(m_i) * (1 - s_i^2)
        ds0 = 1.0 - s0 * s0
        ds4 = 1.0 - s4 * s4
        ds8 = 1.0 - s8 * s8
        
        grad_ts[3] = grad_a[0] * s0 * np.exp(m0) * ds0
        grad_ts[4] = grad_a[1] * s4 * np.exp(m4) * ds4
        grad_ts[5] = grad_a[2] * s8 * np.exp(m8) * ds8
        
        return val, grad_ts
    
    # Generate starting points in log-sign space
    # Convert seed to log-sign space
    def _to_log_sign(a):
        ts = np.zeros(6)
        for i, ai in enumerate(a):
            if abs(ai) < 1e-15:
                ts[i] = -10.0  # log(magnitude) ~ -10 means magnitude ~ 4.5e-5
                ts[i+3] = 0.0  # sign ~ 0
            else:
                ts[i] = np.log(abs(ai))
                ts[i+3] = np.arctanh(np.sign(ai))
        return ts
    
    seed_ts = _to_log_sign(seed_params)
    
    starts = [seed_ts.copy()]
    
    # Grid starts in log-sign space
    # a0 ranges from about -3 to 1, so log|a0| from log(0.01) to log(3)
    # a4 ranges from about -0.1 to 0.1
    # a8 ranges from about 0 to 0.01
    
    rng = np.random.default_rng(42)
    
    # Random starts in log-sign space
    for _ in range(150):
        ts = np.zeros(6)
        ts[0] = rng.uniform(-8, 2)  # log|a0|
        ts[1] = rng.uniform(-8, 0)  # log|a4|
        ts[2] = rng.uniform(-8, -2)  # log|a8|
        ts[3] = rng.uniform(-3, 3)   # t0 (sign parameter)
        ts[4] = rng.uniform(-3, 3)   # t4
        ts[5] = rng.uniform(-3, 3)   # t8
        starts.append(ts)
    
    # Some structured starts
    for sign0 in [-1, 1]:
        for sign4 in [-1, 1]:
            for sign8 in [-1, 1]:
                ts = np.array([
                    np.log(0.5),  # log|a0|
                    np.log(0.01), # log|a4|
                    np.log(0.001),# log|a8|
                    np.arctanh(sign0 * 0.9),
                    np.arctanh(sign4 * 0.9),
                    np.arctanh(sign8 * 0.9)
                ])
                starts.append(ts)
    
    # Shuffle
    rng2 = np.random.default_rng(99)
    perm = np.arange(len(starts))
    perm[1:] = rng2.permutation(perm[1:])
    starts = [starts[i] for i in perm]
    
    # Optimization Loop in log-sign space
    for start in starts:
        if time.time() > _DEADLINE - deadline_margin:
            break
            
        try:
            res = minimize(_log_sign_obj, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if np.isfinite(res.fun) and res.fun < best_val:
                # Convert back to a-space
                m0, m4, m8 = res.x[0], res.x[1], res.x[2]
                t0, t4, t8 = res.x[3], res.x[4], res.x[5]
                
                s0 = np.tanh(t0)
                s4 = np.tanh(t4)
                s8 = np.tanh(t8)
                
                a0 = s0 * np.exp(m0)
                a4 = s4 * np.exp(m4)
                a8 = s8 * np.exp(m8)
                
                candidate_a = np.array([a0, a4, a8])
                val_check, _ = _compute_bound_and_grad(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass

    # Final polish in log-sign space on best found
    if time.time() < _DEADLINE - deadline_margin:
        try:
            best_ts = _to_log_sign(best_params)
            res = minimize(_log_sign_obj, best_ts, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                m0, m4, m8 = res.x[0], res.x[1], res.x[2]
                t0, t4, t8 = res.x[3], res.x[4], res.x[5]
                
                s0 = np.tanh(t0)
                s4 = np.tanh(t4)
                s8 = np.tanh(t8)
                
                a0 = s0 * np.exp(m0)
                a4 = s4 * np.exp(m4)
                a8 = s8 * np.exp(m8)
                
                candidate_a = np.array([a0, a4, a8])
                val_check, _ = _compute_bound_and_grad(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass

    # One more Nelder-Mead in log-sign space
    if time.time() < _DEADLINE - deadline_margin:
        try:
            best_ts = _to_log_sign(best_params)
            res = minimize(lambda ts: _log_sign_obj(ts), best_ts, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                m0, m4, m8 = res.x[0], res.x[1], res.x[2]
                t0, t4, t8 = res.x[3], res.x[4], res.x[5]
                
                s0 = np.tanh(t0)
                s4 = np.tanh(t4)
                s8 = np.tanh(t8)
                
                a0 = s0 * np.exp(m0)
                a4 = s4 * np.exp(m4)
                a8 = s8 * np.exp(m8)
                
                candidate_a = np.array([a0, a4, a8])
                val_check, _ = _compute_bound_and_grad(candidate_a)
                if val_check < best_val:
                    best_val = val_check
                    best_params = candidate_a.copy()
        except Exception:
            pass

    if not np.all(np.isfinite(best_params)):
        return seed_params

    return np.array(best_params)
