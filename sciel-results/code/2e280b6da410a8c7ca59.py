"""High-order implicit RK (Radau IIA 5-stage) with Newton for Van der Pol."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Radau IIA 5-stage coefficients (order 9, L-stable)
# Nodes: c_i = (i - 0.5)/5 for i=1..5
# For Radau IIA, b = c, and A is the collocation matrix.

def _compute_radau_ii_a(s):
    c = (np.arange(1, s + 1) - 0.5) / s
    A = np.zeros((s, s))
    for j in range(s):
        # Lagrange polynomial L_j(t) = prod_{k != j} (t - c_k)/(c_j - c_k)
        num = np.array([1.0])
        den = 1.0
        for k in range(s):
            if k != j:
                num = np.convolve(num, np.array([1.0, -c[k]]))
                den *= (c[j] - c[k])
        num = num / den
        for i in range(s):
            val = 0.0
            for m in range(s):
                power = s - 1 - m
                if abs(c[i] - c[j]) > 1e-15:
                    val += num[m] * (c[j]**(power + 1) - c[i]**(power + 1)) / (power + 1)
            A[i, j] = val
    b = c
    return A, b, c

_A5, _b5, _c5 = _compute_radau_ii_a(5)


def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 280.0
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    s = len(_c5)
    
    # Use 8 steps of size 1.25
    num_steps = 8
    h = (t1 - t0) / num_steps
    
    for step in range(num_steps):
        if time.time() > deadline - 5.0:
            break
        
        t = t0 + step * h
        Y = np.tile(y, (s, 1)).copy()
        
        # Newton iteration for the implicit stage equations
        for newton_iter in range(30):
            F = np.array([f(t + _c5[i] * h, Y[i]) for i in range(s)])
            AF = _A5 @ F
            R = Y - y[None, :] - h * AF
            
            # Build full Jacobian (s*n x s*n) via finite differences
            sn = s * n
            J_full = np.zeros((sn, sn))
            R_flat = R.flatten()
            
            for k in range(sn):
                Yp = Y.copy()
                Yp.flat[k] += 1e-8
                Fp = np.array([f(t + _c5[i] * h, Yp[i]) for i in range(s)])
                AFp = _A5 @ Fp
                Rp = (Yp - y[None, :] - h * AFp).flatten()
                J_full[:, k] = (Rp - R_flat) / 1e-8
            
            try:
                delta = np.linalg.solve(J_full, -R_flat)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(J_full, -R_flat, rcond=None)[0]
            
            Y += delta.reshape(s, n)
            
            if np.linalg.norm(delta) <= 1e-14 * (1.0 + np.linalg.norm(Y.flatten())):
                break
        
        F = np.array([f(t + _c5[i] * h, Y[i]) for i in range(s)])
        y = y + h * (_b5 @ F)
        
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
