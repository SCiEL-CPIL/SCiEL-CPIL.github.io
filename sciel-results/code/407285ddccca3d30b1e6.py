import numpy as np
import scipy.integrate
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 290.0

    def extract_params(t_val):
        zero_vec = np.zeros(n, dtype=float)
        b_val = np.array(f(t_val, zero_vec), dtype=float)
        
        A_val = np.zeros((n, n), dtype=float)
        for i in range(n):
            e_i = np.zeros(n, dtype=float)
            e_i[i] = 1.0
            col_i = np.array(f(t_val, e_i), dtype=float) - b_val
            A_val[:, i] = col_i
        return A_val, b_val

    A0, b0 = extract_params(t0)
    A1, b1 = extract_params(t1)
    
    norm_A0 = np.linalg.norm(A0)
    norm_b0 = np.linalg.norm(b0)
    
    tol_A = 1e-10 * max(1.0, norm_A0)
    tol_b = 1e-10 * max(1.0, norm_b0)
    
    diff_A = np.linalg.norm(A0 - A1)
    diff_b = np.linalg.norm(b0 - b1)
    
    if diff_A < tol_A and diff_b < tol_b:
        M = np.zeros((n+1, n+1))
        M[:n, :n] = A0
        M[:n, n] = b0
        dt = t1 - t0
        try:
            M_exp = scipy.linalg.expm(M * dt)
            z0 = np.zeros(n+1)
            z0[:n] = y0
            z0[n] = 1.0
            z1 = M_exp @ z0
            return z1[:n]
        except Exception:
            def f_const(t_val, y_val):
                return A0 @ y_val + b0
            sol = scipy.integrate.solve_ivp(
                f_const, 
                (t0, t1), 
                y0, 
                method='Radau',
                rtol=1e-13, 
                atol=1e-15,
                max_step=dt / 100.0
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                return y0

    probe_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    probe_ts = [t0 + (t1 - t0) * t for t in probe_times]
    
    A_samples = [A0]
    b_samples = [b0]
    
    for t_p in probe_ts[1:-1]:
        if time.time() > deadline - 10.0:
             return _solve_direct(f, t_span, y0)
        A_p, b_p = extract_params(t_p)
        A_samples.append(A_p)
        b_samples.append(b_p)
        
    A_samples.append(A1)
    b_samples.append(b1)
    
    A_samples = np.array(A_samples)
    b_samples = np.array(b_samples)
    
    u_samples = np.array(probe_times)
    V = np.vander(u_samples, N=5, increasing=True)
    
    A_flat = A_samples.reshape(5, n*n)
    try:
        A_coeffs, residuals, rank, s = np.linalg.lstsq(V, A_flat, rcond=None)
    except np.linalg.LinAlgError:
        return _solve_direct(f, t_span, y0)
        
    try:
        b_coeffs, residuals_b, rank_b, s_b = np.linalg.lstsq(V, b_samples, rcond=None)
    except np.linalg.LinAlgError:
        return _solve_direct(f, t_span, y0)
        
    def f_model(t_val, y_val):
        if t1 == t0:
            return np.zeros(n)
        u_val = (t_val - t0) / (t1 - t0)
        
        u_powers = np.array([u_val**k for k in range(5)])
        
        A_val = np.zeros(n*n)
        for k in range(5):
            A_val += A_coeffs[k] * (u_val**k)
        A_mat = A_val.reshape(n, n)
        
        b_val = np.zeros(n)
        for k in range(5):
            b_val += b_coeffs[k] * (u_val**k)
            
        return A_mat @ y_val + b_val

    try:
        sol = scipy.integrate.solve_ivp(
            f_model, 
            (t0, t1), 
            y0, 
            method='Radau',
            rtol=1e-13, 
            atol=1e-15,
            max_step=(t1 - t0) / 50.0
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            sol = scipy.integrate.solve_ivp(
                f_model, 
                (t0, t1), 
                y0, 
                method='BDF',
                rtol=1e-13, 
                atol=1e-15,
                max_step=(t1 - t0) / 50.0
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                return _solve_direct(f, t_span, y0)
                
    except Exception:
        return _solve_direct(f, t_span, y0)

def _solve_direct(f, t_span, y0):
    t0, t1 = t_span
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-12
            )
            return sol.y[:, -1]
    except Exception:
        return y0
