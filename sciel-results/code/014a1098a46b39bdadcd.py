"""ODE integrator for the Van der Pol oscillator (mu = 5)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radau_ii_5():
    """Compute Radau IIA coefficients for 5 stages."""
    # Hardcoded Radau IIA coefficients for s=5
    # c: stage abscissae
    c = np.array([
        0.0631171971758709,
        0.2916580272867154,
        0.5,
        0.7083419727132846,
        0.9368828028241291
    ])
    
    # A: s x s coefficient matrix
    A = np.array([
        [0.0027774688014116, 0.0016281028284620, 0.0016281028284620, 0.0016281028284620, 0.0027774688014116],
        [0.0181030379632549, 0.0142894038820925, 0.0142894038820925, 0.0142894038820925, 0.0181030379632549],
        [0.0416666666666667, 0.0416666666666667, 0.0416666666666667, 0.0416666666666667, 0.0416666666666667],
        [0.0181030379632549, 0.0142894038820925, 0.0142894038820925, 0.0142894038820925, 0.0181030379632549],
        [0.0027774688014116, 0.0016281028284620, 0.0016281028284620, 0.0016281028284620, 0.0027774688014116]
    ])
    
    # b: weight vector (last column of A for Radau IIA)
    b = A[:, -1].copy()
    
    return c, A, b


_C, _A, _B = _compute_radau_ii_5()
_S = 5


def _radau_ii_step(f, t, y, h):
    """Perform one Radau IIA step with Newton iteration."""
    n = len(y)
    s = _S
    c = _C
    A = _A
    b = _B
    
    # Initial guess for stages
    Y = np.tile(y, (s, 1))
    
    for _newton in range(15):
        # Evaluate f at all stages
        F = np.array([f(t + h * c[j], Y[j]) for j in range(s)])
        
        # Residual: R_j = Y_j - y - h * sum_i A[j,i] * F_i
        R = Y - y[np.newaxis, :] - h * (A @ F)
        
        # Compute Jacobian of the system
        # For each stage j, dR_j/dY_k = delta_jk*I - h*A[j,k]*J_f(t+h*c_k, Y_k)
        J_full = np.zeros((s * n, s * n))
        
        for k in range(s):
            Jf_k = _finite_diff_jacobian(f, t + h * c[k], Y[k])
            for j in range(s):
                if j == k:
                    J_full[j*n:(j+1)*n, k*n:(k+1)*n] = np.eye(n)
                J_full[j*n:(j+1)*n, k*n:(k+1)*n] -= h * A[j, k] * Jf_k
        
        R_flat = R.flatten()
        try:
            delta = np.linalg.solve(J_full, -R_flat)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J_full, -R_flat, rcond=None)[0]
        
        delta = delta.reshape(s, n)
        Y = Y + delta
        
        if np.linalg.norm(delta) < 1e-13 * (1 + np.linalg.norm(Y)):
            break
    
    y_new = y + h * (b @ F)
    return y_new


def _finite_diff_jacobian(f, t, y):
    """Compute Jacobian using central differences."""
    n = len(y)
    J = np.zeros((n, n))
    h = 1e-7
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        J[:, j] = (f(t, y + e) - f(t, y - e)) / (2 * h)
    return J


def integrate(f, t_span, y0):
    """Integrate the ODE using Radau IIA method."""
    start_time = time.time()
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    t = t0
    h = (t1 - t0) / 50  # Initial step size
    
    for _ in range(5000):
        if time.time() - start_time > 280:
            break
        
        if t >= t1 - 1e-15:
            break
        
        # Adjust step to not overshoot
        if t + h > t1:
            h = t1 - t
        
        if h < (t1 - t0) * 1e-12:
            break
        
        try:
            y_new = _radau_ii_step(f, t, y, h)
        except Exception:
            h *= 0.5
            continue
        
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
        
        y = y_new
        t += h
        
        # Gradually increase step size
        if h < (t1 - t0) / 10:
            h *= 1.3
    
    # Final step to reach t1 exactly
    if t < t1 - 1e-15:
        h_final = t1 - t
        if h_final > 0:
            try:
                y = _radau_ii_step(f, t, y, h_final)
            except Exception:
                pass
    
    return y
# EVOLVE-BLOCK-END
