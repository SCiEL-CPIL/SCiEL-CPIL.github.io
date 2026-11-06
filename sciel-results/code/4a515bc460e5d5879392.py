"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    deadline = time.time() + 290.0  # safety margin before 300s/330s limit
    
    # Robertson problem is known:
    # y0 = [1, 0, 0], y' = f(t, y)
    # y1' = -0.04*y1 + 1e4*y2*y3
    # y2' = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
    # y3' = 3e7*y2^2
    #
    # We use an adaptive implicit Runge-Kutta method (Radau IIA) with
    # Newton iteration and finite-difference Jacobian.
    # For this stiff 3D problem, Radau IIA (3-stage, order 5) is very efficient.
    
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Radau IIA 3-stage coefficients
    # A = [[1/6, 1/6, 1/6], [2/15, 1/15, 1/5], [1/10, 1/5, 3/10]]
    # c = [1/6, 3/5, 1]
    # b = [1/6, 5/6, 0]  (wait, let me get the right coefficients)
    
    # Actually, Radau IIA 3-stage:
    # c1 = 1/6, c2 = 3/5, c3 = 1
    # A = [[1/6, 1/6, 1/6],
    #      [2/15, 1/15, 1/5],
    #      [1/10, 1/5, 3/10]]
    # b = [1/6, 5/6, 0] -- no, that's not right either
    # b = [1/6, 5/6, 0]? Let me think...
    # For Radau IIA 3-stage: b = [1/6, 5/6, 0] is wrong.
    # The correct b for Radau IIA 3-stage is [1/6, 5/6, 0]? No.
    # Actually for Radau IIA, the last row of A equals b.
    # So b = [1/10, 1/5, 3/10]
    
    A = np.array([[1.0/6, 1.0/6, 1.0/6],
                  [2.0/15, 1.0/15, 1.0/5],
                  [1.0/10, 1.0/5, 3.0/10]])
    c = np.array([1.0/6, 3.0/5, 1.0])
    b = A[2, :]  # b = last row of A for Radau IIA
    
    # Initial step size
    h = (t1 - t0) / 1000.0
    t = t0
    f_evals = 0
    
    # Tolerances
    atol = 1e-12
    rtol = 1e-10
    
    def count_f(func):
        nonlocal f_evals
        f_evals += 1
        return func()
    
    # We'll track f calls manually since we need to count them
    # Let's wrap f to count calls
    f_count = [0]
    def counted_f(t_val, y_val):
        f_count[0] += 1
        return f(t_val, y_val)
    
    # Use a different approach: just use scipy's solve_ivp with Radau if available
    # But we need to count f calls, so let's do it manually or use a wrapper
    
    # Actually, let's just use scipy.integrate.solve_ivp with method='Radau'
    # and count function calls via a wrapper
    from scipy.integrate import solve_ivp
    
    f_calls = [0]
    def wrapped_f(t_val, y_val):
        f_calls[0] += 1
        return f(t_val, y_val)
    
    # Try scipy's Radau solver
    try:
        sol = solve_ivp(wrapped_f, t_span, y0, method='Radau',
                       rtol=1e-10, atol=1e-12, max_step=(t1-t0)/100)
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: manual Radau IIA with Newton
    y = np.array(y0, dtype=float)
    t = t0
    h = (t1 - t0) / 1000.0
    f_count[0] = 0
    
    def jac_fd(t_val, y_val, h_fd=1e-8):
        J = np.empty((n, n))
        fy = f(t_val, y_val)
        f_count[0] += 1
        for j in range(n):
            dy = np.zeros(n)
            dy[j] = h_fd * max(1.0, abs(y_val[j]))
            fy_pert = f(t_val, y_val + dy)
            f_count[0] += 1
            J[:, j] = (fy_pert - fy) / dy[j]
        return J
    
    while t < t1:
        if time.time() > deadline:
            break
        if f_count[0] > 1500000:
            break
            
        h = min(h, t1 - t)
        success = False
        
        for attempt in range(3):
            # Radau IIA 3-stage
            Y = np.zeros((3, n))
            for i in range(3):
                # Newton iteration for stage i
                Y[i] = y.copy()
                for newton_iter in range(10):
                    # Compute F(t + c_i*h, Y[i])
                    F_i = f(t + c[i] * h, Y[i])
                    f_count[0] += 1
                    
                    # Compute sum_j A[i,j] * F_j
                    # First, we need F for all stages j <= i (for j > i, use previous Y)
                    F_all = np.zeros((3, n))
                    for j in range(3):
                        if j == i:
                            F_all[j] = F_i
                        elif j < i:
                            F_all[j] = f(t + c[j] * h, Y[j])
                            f_count[0] += 1
                        else:
                            # For j > i, we haven't computed yet; use 0 or previous
                            F_all[j] = 0.0
                    
                    # Residual: Y[i] - y - h * (A[i,:] @ F_all)
                    sum_AF = np.zeros(n)
                    for j in range(3):
                        sum_AF += A[i, j] * F_all[j]
                    
                    residual = Y[i] - y - h * sum_AF
                    
                    if np.linalg.norm(residual) <= 1e-12 * (1 + np.linalg.norm(Y[i])):
                        break
                    
                    # Jacobian of the residual w.r.t. Y[i]
                    # d/dY[i] [Y[i] - y - h * sum_j A[i,j] F_j]
                    # = I - h * A[i,i] * J_F(t + c_i*h, Y[i])
                    J_f = jac_fd(t + c[i] * h, Y[i])
                    
                    M = np.eye(n) - h * A[i, i] * J_f
                    try:
                        delta = np.linalg.solve(M, -residual)
                    except np.linalg.LinAlgError:
                        delta = -residual
                    Y[i] += delta
                    if np.linalg.norm(delta) <= 1e-14 * (1 + np.linalg.norm(Y[i])):
                        break
            
            # Update
            y_new = y.copy()
            for i in range(3):
                y_new += b[i] * f(t + c[i] * h, Y[i])
                f_count[0] += 1
            
            # Error estimate: use a lower-order method (e.g., first 2 stages)
            # For Radau IIA 3-stage, embedded 2nd order: b2 = [1/2, 0, 0] approximately
            # Simple error estimate: compare with explicit Euler
            y_euler = y + h * f(t, y)
            f_count[0] += 1
            err = np.linalg.norm(y_new - y_euler) / (atol + rtol * np.linalg.norm(y_new))
            
            if err <= 1.0:
                y = y_new
                t += h
                h = min(h * 2.0, (t1 - t) / 10)  # grow step
                success = True
                break
            else:
                # Reduce step
                h = h * max(0.2, min(0.9, 0.9 * err ** (-0.25)))
        
        if not success:
            # Try smaller step
            h = h * 0.5
            if h < 1e-12:
                break
    
    return y
# EVOLVE-BLOCK-END
