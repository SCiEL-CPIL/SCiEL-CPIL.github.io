import numpy as np
import scipy.linalg as sla

def integrate(f, t_span, y0):
    t0, t1 = t_span
    T = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Get b at t=0
    y_test_0 = np.zeros(N)
    b_0 = f(t0, y_test_0)
    
    # Check if b is constant by checking at t_mid
    t_mid = 0.5 * (t0 + t1)
    b_mid = f(t_mid, y_test_0)
    
    is_affine = np.linalg.norm(b_0) > 1e-10
    is_time_independent = np.linalg.norm(b_0 - b_mid) < 1e-8 * (1 + np.linalg.norm(b_0))
    
    # Extract A columns
    # A[:, i] = f(t0, e_i) - b_0
    A = np.empty((N, N), dtype=float)
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0
        v_i = f(t0, e_i)
        A[:, i] = v_i - b_0
        
    # Verify linearity at t0
    # Check f(t0, y_random) == A @ y_random + b_0
    y_rand = np.random.randn(N)
    f_rand = f(t0, y_rand)
    pred_rand = A @ y_rand + b_0
    linearity_ok = np.linalg.norm(f_rand - pred_rand) < 1e-6 * (1 + np.linalg.norm(f_rand))
    
    if is_time_independent and linearity_ok:
        # Solve exactly using matrix exponential via eigendecomposition
        if is_affine:
            # Augmented system: z' = M z, z = [y; 1]
            # M = [[A, b_0], [0, 0]]
            M = np.zeros((N + 1, N + 1), dtype=float)
            M[:N, :N] = A
            M[:N, N] = b_0
            z0 = np.zeros(N + 1)
            z0[:N] = y0
            z0[N] = 1.0
            n_dim = N + 1
        else:
            M = A
            z0 = y0
            n_dim = N
            
        # Eigendecomposition
        try:
            vals, vecs = np.linalg.eig(M)
            # Compute exp(lambda * T)
            exp_vals = np.exp(vals * T)
            
            # z_end = V * diag(exp_vals) * V^-1 * z0
            # Let w = V^-1 * z0
            # w = solve(V, z0)
            w = np.linalg.solve(vecs, z0)
            # w_scaled = exp_vals * w
            w_scaled = exp_vals * w
            # z_end = V * w_scaled
            z_end = vecs @ w_scaled
            
            if is_affine:
                z_end = z_end[:N]
            
            # Check for numerical stability: if imaginary parts are small, take real
            if np.max(np.abs(z_end.imag)) < 1e-10 * (1 + np.max(np.abs(z_end.real))):
                z_end = z_end.real
            else:
                # If significant imaginary part, something might be wrong, or the system is genuinely complex?
                # But f returns float. So result should be real.
                # If imag is large, fallback.
                raise np.linalg.LinAlgError("Large imaginary part in result")
                
            return z_end
                
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            # Fallback if eig fails or is unstable
            pass
    else:
        # Fallback to robust adaptive solver
        # Use LSODA which switches between Adams and BDF, good for stiff/non-stiff
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='LSODA', rtol=1e-10, atol=1e-12, max_step=T/100)
        if sol.success:
            return sol.y[:, -1]
        else:
            # Last resort: RK4 with many steps
            steps = 10000
            h = T / steps
            t, y = t0, y0.copy()
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h/2 * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
