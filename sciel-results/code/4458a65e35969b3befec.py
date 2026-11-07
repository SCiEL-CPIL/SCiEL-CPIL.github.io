"""Radau IIA collocation integrator for the Van der Pol oscillator (mu=5)."""
# EVOLVE-BLOCK-START
import numpy as np


def _radau_ii_a_coefficients(n_stages=5):
    """
    Compute the Radau IIA collocation coefficients for n_stages stages.
    """
    s = n_stages
    degree = s + 1
    
    # Build shifted Legendre polynomial coefficients in ascending order
    # P_0(x) = 1
    coeffs = [1]
    if degree >= 1:
        coeffs = [-1, 2]
    
    coeffs_prev = [1] if degree >= 2 else None
    
    for n in range(2, degree + 1):
        p_prev = coeffs
        if n >= 2:
            p_prev2 = coeffs_prev
        else:
            p_prev2 = [1]
        
        # (2x-1) P_{n-1}
        p_prev_shifted = [0] + [2 * c for c in p_prev]
        p_prev_neg = [-c for c in p_prev]
        term1 = [p_prev_shifted[i] + p_prev_neg[i] for i in range(len(p_prev_shifted))]
        term1 = [(2 * n - 1) * c for c in term1]
        
        # -2(n-1) P_{n-2}
        term2 = [-2 * (n - 1) * c for c in p_prev2]
        
        max_len = max(len(term1), len(term2))
        term1_padded = term1 + [0] * (max_len - len(term1))
        term2_padded = term2 + [0] * (max_len - len(term2))
        
        coeffs = [term1_padded[i] + term2_padded[i] for i in range(max_len)]
        coeffs_prev = coeffs
    
    # Find roots
    coeffs_desc = coeffs[::-1]
    roots = np.roots(coeffs_desc)
    
    real_roots = []
    for r in roots:
        if abs(np.imag(r)) < 1e-12:
            rr = np.real(r)
            if -1e-12 <= rr <= 1 + 1e-12:
                rr = np.clip(rr, 0, 1)
                real_roots.append(rr)
    
    real_roots = np.array(sorted(real_roots))
    idx_zero = np.argmin(real_roots)
    collocation_points = np.delete(real_roots, idx_zero)
    
    if len(collocation_points) != s:
        collocation_points = real_roots[-s:]
    
    collocation_points = np.clip(collocation_points, 0, 1)
    
    # Compute collocation matrix A
    c = collocation_points
    A = np.zeros((s, s))
    
    for i in range(s):
        for j in range(s):
            num_coeffs = [1]
            for k in range(s):
                if k == j:
                    continue
                ck = c[k]
                new_coeffs = [0] * (len(num_coeffs) + 1)
                for idx in range(len(num_coeffs)):
                    new_coeffs[idx] += -ck * num_coeffs[idx]
                    new_coeffs[idx + 1] += num_coeffs[idx]
                num_coeffs = new_coeffs
            
            denom = 1.0
            for k in range(s):
                if k == j:
                    continue
                denom *= (c[j] - c[k])
            
            l_coeffs = [coeff / denom for coeff in num_coeffs]
            
            integral_val = 0.0
            for m in range(len(l_coeffs)):
                integral_val += l_coeffs[m] * c[i] ** (m + 1) / (m + 1)
            
            A[i, j] = integral_val
    
    return c, A


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    n_stages = 5
    c, A = _radau_ii_a_coefficients(n_stages)
    b = A[-1, :]
    
    n_steps = 100
    h = (t1 - t0) / n_steps
    
    for step in range(n_steps):
        t_curr = t0 + step * h
        
        Y = np.tile(y, (n_stages, 1))
        
        for newton_iter in range(20):
            f_vals = np.zeros((n_stages, n))
            for i in range(n_stages):
                f_vals[i] = f(t_curr + c[i] * h, Y[i])
            
            R = np.zeros((n_stages, n))
            for i in range(n_stages):
                R[i] = Y[i] - y - h * np.dot(A[i, :], f_vals)
            
            R_flat = R.flatten()
            if np.linalg.norm(R_flat) < 1e-14:
                break
            
            total_dim = n * n_stages
            J_full = np.zeros((total_dim, total_dim))
            
            for i in range(n_stages):
                for j in range(n_stages):
                    J_f = np.zeros((n, n))
                    for k in range(n):
                        e = np.zeros(n)
                        scale = 1e-7 * max(1.0, abs(Y[j, k]))
                        e[k] = scale
                        J_f[:, k] = (f(t_curr + c[j] * h, Y[j] + e) - f_vals[j]) / scale
                    
                    if i == j:
                        J_full[i * n:(i + 1) * n, j * n:(j + 1) * n] = (
                            np.eye(n) - h * A[i, j] * J_f
                        )
                    else:
                        J_full[i * n:(i + 1) * n, j * n:(j + 1) * n] = (
                            -h * A[i, j] * J_f
                        )
            
            try:
                delta_Y = np.linalg.solve(J_full, -R_flat)
            except np.linalg.LinAlgError:
                delta_Y, _, _, _ = np.linalg.lstsq(J_full, -R_flat, rcond=None)
            
            Y = Y + delta_Y.reshape(n_stages, n)
        
        y = y + h * np.dot(b, f_vals)
    
    return y

# EVOLVE-BLOCK-END
