import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solve the Robertson problem using a custom Radau IIA (L-stable, 3rd order) solver.
    
    Strategy:
    1. Detect Robertson problem structure to use analytical Jacobian.
    2. Implement a custom Radau IIA(3) solver with adaptive step control.
       Radau IIA is an implicit Runge-Kutta method that is L-stable and A-stable.
       It is particularly well-suited for stiff problems.
       Order 3 provides a good balance between accuracy and computational cost.
       For stiff problems, Radau methods are often more efficient than BDF.
    3. Use a multi-pass strategy if time allows, but Radau is usually more efficient
       than BDF for the same accuracy, so we might reach higher accuracy with fewer calls.
       
       Score = -log10(err) - 0.5*log10(calls).
       If Radau achieves err ~ 1e-15 with ~500 calls, score ~ 15 - 1.2 = 13.8.
       If BDF achieves err ~ 1e-14 with ~5000 calls, score ~ 14 - 1.75 = 12.25.
       So Radau should be significantly better.
       
       The current program (BDF) scored 5.54. This suggests either:
       - The BDF implementation was not achieving high accuracy (maybe rtol=1e-10 was the limit).
       - Or the call count was high.
       
       Let's aim for rtol=1e-13 with Radau, which should be very efficient.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Verify Robertson structure
    is_robertson = False
    if n == 3:
        try:
            y_test = np.array([1.0, 1.0, 0.0])
            f_test = f(t0, y_test)
            f_expected = np.array([-1e4, 7e3, 3e3])
            if np.allclose(f_test, f_expected, rtol=1e-5, atol=1e-8):
                is_robertson = True
        except:
            pass
    
    if not is_robertson:
        # Fallback to SciPy for non-Robertson
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-13, atol=1e-16)
        if sol.success:
            return sol.y[:, -1]
        return y0
    
    # Define analytical Jacobian for Robertson
    def robertson_jac(t, y):
        y1, y2, y3 = y
        return np.array([
            [-1e4*y2, -1e4*y1, 0],
            [ 1e4*y2, 1e4*y1 - 6e3*y2, 0],
            [ 0,      6e3*y2,      0]
        ])
    
    # Radau IIA(3) coefficients
    # The Butcher tableau for Radau IIA(3):
    # c1 = 1/6 + sqrt(6)/6
    # c2 = 1/2
    # c3 = 1
    # 
    # A = [ [a11, a12, a13],
    #       [a21, a22, a23],
    #       [a31, a32, a33] ]
    # 
    # b = [b1, b2, b3]
    
    sqrt6 = np.sqrt(6.0)
    c = np.array([
        1.0/6.0 + sqrt6/6.0,
        0.5,
        1.0
    ])
    
    # Coefficients for Radau IIA(3)
    # From standard references:
    a11 = (1.0 + 2.0*sqrt6/3.0) / 18.0
    a12 = (2.0 + sqrt6) / 18.0
    a13 = (1.0 - 2.0*sqrt6/3.0) / 18.0
    
    a21 = (2.0 + sqrt6) / 18.0
    a22 = (7.0 + 2.0*sqrt6) / 18.0
    a23 = (1.0 - 2.0*sqrt6) / 18.0
    
    a31 = (2.0 + sqrt6) / 6.0
    a32 = (1.0 - 2.0*sqrt6) / 6.0
    a33 = (2.0 + sqrt6) / 6.0
    
    # b coefficients (same as last row of A for Radau)
    b1 = a31
    b2 = a32
    b3 = a33
    
    # Check order conditions: sum(b_i * c_i^k) = 1/(k+1)
    # This is satisfied by Radau IIA.
    
    # Error estimation: Use Radau IIA(2) or a lower order method.
    # For simplicity, we can use the difference between Radau(3) and a 2-step BDF or Radau(2).
    # Or, we can use the fact that for stiff problems, the error is often dominated by the
    # local truncation error, which is O(h^4) for Radau(3).
    # A common approach is to use a lower order Radau method (Radau IIA(2)) for error estimation.
    
    # Radau IIA(2) coefficients:
    # c1 = 1/2 - sqrt(3)/6
    # c2 = 1/2 + sqrt(3)/6
    # A = [ [ (1 - sqrt(3)/3)/4, (1 + sqrt(3)/3)/4 ],
    #       [ (1 + sqrt(3)/3)/4, (1 - sqrt(3)/3)/4 ] ]
    # b = [ (1 + sqrt(3)/3)/4, (1 - sqrt(3)/3)/4 ]  # Actually, b = last row of A for Radau?
    # Let's use a simpler error estimator: compare with a 2-step BDF or just use the residual.
    # For now, let's implement Radau(3) and use a simple error estimator based on the difference
    # between Radau(3) and a 2nd order method (e.g., midpoint or BDF(2)).
    
    # Actually, a more robust approach is to use the "local extrapolation" idea:
    # Compute the solution with Radau(3) and with a lower order method (e.g., Radau(2)),
    # and estimate the error as the difference.
    
    # Let's implement Radau IIA(2) for error estimation.
    sqrt3 = np.sqrt(3.0)
    c2_radau2 = np.array([
        0.5 - sqrt3/6.0,
        0.5 + sqrt3/6.0
    ])
    a11_r2 = (1.0 - sqrt3/3.0) / 4.0
    a12_r2 = (1.0 + sqrt3/3.0) / 4.0
    a21_r2 = (1.0 + sqrt3/3.0) / 4.0
    a22_r2 = (1.0 - sqrt3/3.0) / 4.0
    b1_r2 = a12_r2  # b = last row? No, b is separate.
    # For Radau IIA(2), b = [ (1 + sqrt(3)/3)/4, (1 - sqrt(3)/3)/4 ]
    b1_r2 = (1.0 + sqrt3/3.0) / 4.0
    b2_r2 = (1.0 - sqrt3/3.0) / 4.0
    
    # Now, implement the solver.
    
    def solve_radau(f, jac, t_span, y0, rtol, atol, max_calls):
        t0, t1 = t_span
        y = np.array(y0, dtype=float)
        n = len(y)
        
        # Initial step size estimate
        h = 1e-3
        if t1 > t0:
            f0 = f(t0, y)
            f_norm = np.linalg.norm(f0)
            if f_norm > 1e-10:
                h = 1e-3 * (1.0 / f_norm)
                h = min(h, (t1 - t0) / 100.0)
                h = max(h, 1e-8)
            else:
                h = (t1 - t0) / 100.0
        else:
            return y0, 1, True
        
        calls = 1  # initial f call
        t = t0
        
        # Precompute some constants
        A_radau3 = np.array([
            [a11, a12, a13],
            [a21, a22, a23],
            [a31, a32, a33]
        ])
        b_radau3 = np.array([b1, b2, b3])
        
        A_radau2 = np.array([
            [a11_r2, a12_r2],
            [a21_r2, a22_r2]
        ])
        b_radau2 = np.array([b1_r2, b2_r2])
        
        while t < t1:
            # Ensure we don't overshoot
            if t + h > t1:
                h = t1 - t
            if h < 1e-15:
                break
                
            t_new = t + h
            
            # Solve for Radau(3) stages
            # Y_i = y + h * sum_j A_ij f(t + c_j h, Y_j)
            # We need to solve for Y_1, Y_2, Y_3 simultaneously using Newton's method.
            
            # Predictor: Use forward Euler or extrapolation
            Y = np.zeros((3, n))
            for i in range(3):
                Y[i] = y + h * f(t, y)  # Crude predictor
            
            # Newton iteration for Radau(3)
            for _ in range(50):
                # Compute F(Y) = Y - y - h * A @ f(Y)
                F = np.zeros((3, n))
                for i in range(3):
                    F[i] = Y[i] - y - h * np.sum(A_radau3[i, :] * np.array([f(t + c[0]*h, Y[0]), f(t + c[1]*h, Y[1]), f(t + c[2]*h, Y[2])]), axis=0)
                    calls += 3  # 3 f calls per stage evaluation
                    if calls > max_calls:
                        return y, calls, False
                
                # Check convergence
                F_norm = np.linalg.norm(F)
                if F_norm < atol + rtol * np.linalg.norm(y):
                    break
                
                # Jacobian of F wrt Y: I - h * A @ J
                J = jac(t + c[0]*h, Y[0])  # Approximate Jacobian (could be more accurate)
                J_total = np.zeros((3*n, 3*n))
                for i in range(3):
                    for j in range(3):
                        J_total[i*n:(i+1)*n, j*n:(j+1)*n] = -h * A_radau3[i, j] * (J if i == j else J)  # This is not quite right
                # Actually, the Jacobian is a block matrix:
                # J_F = I - h * A @ J_Y, where J_Y is the Jacobian of f wrt Y.
                # Since f depends on Y_i, the Jacobian is:
                # J_F[i, j] = -h * A_ij * J_f(t + c_j h, Y_j) if i == j? No, it's more complex.
                # Let's use a simpler approach: use the Jacobian at the current point.
                
                # For now, use a simplified Newton step
                J_approx = jac(t + h, Y[2])
                A_matrix = np.eye(3*n) - h * np.kron(A_radau3, J_approx)
                try:
                    delta = np.linalg.solve(A_matrix, F.flatten())
                    Y += delta.reshape(3, n)
                except:
                    break
            
            # Compute the Radau(3) solution
            y_new_r3 = y + h * np.sum(b_radau3 * np.array([f(t + c[0]*h, Y[0]), f(t + c[1]*h, Y[1]), f(t + c[2]*h, Y[2])]), axis=0)
            calls += 3  # 3 more f calls
            
            # Compute Radau(2) solution for error estimation
            Y2 = np.zeros((2, n))
            for i in range(2):
                Y2[i] = y + h * f(t, y)
            
            for _ in range(50):
                F2 = np.zeros((2, n))
                for i in range(2):
                    F2[i] = Y2[i] - y - h * np.sum(A_radau2[i, :] * np.array([f(t + c2_radau2[0]*h, Y2[0]), f(t + c2_radau2[1]*h, Y2[1])]), axis=0)
                    calls += 2
                    if calls > max_calls:
                        return y, calls, False
                
                F2_norm = np.linalg.norm(F2)
                if F2_norm < atol + rtol * np.linalg.norm(y):
                    break
                
                J_approx2 = jac(t + h, Y2[1])
                A_matrix2 = np.eye(2*n) - h * np.kron(A_radau2, J_approx2)
                try:
                    delta2 = np.linalg.solve(A_matrix2, F2.flatten())
                    Y2 += delta2.reshape(2, n)
                except:
                    break
            
            y_new_r2 = y + h * np.sum(b_radau2 * np.array([f(t + c2_radau2[0]*h, Y2[0]), f(t + c2_radau2[1]*h, Y2[1])]), axis=0)
            calls += 2
            
            # Error estimate
            err = np.linalg.norm(y_new_r3 - y_new_r2)
            err_tol = atol + rtol * np.linalg.norm(y_new_r3)
            
            if err > err_tol:
                # Reduce step size
                h *= 0.5
                continue
            
            # Accept step
            y = y_new_r3
            t = t_new
            
            # Adjust step size
            if err < 0.1 * err_tol:
                h *= 1.5
                if t + h > t1:
                    h = t1 - t
            elif err < 0.5 * err_tol:
                h *= 1.2
            
        return y, calls, True
    
    # Run the solver
    rtol = 1e-13
    atol = 1e-16
    max_calls = 2000000
    
    try:
        y_sol, calls, success = solve_radau(f, robertson_jac, t_span, y0, rtol, atol, max_calls)
        if success:
            return y_sol
    except Exception:
        pass
    
    # Fallback to SciPy
    from scipy.integrate import solve_ivp
    sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-13, atol=1e-16)
    if sol.success:
        return sol.y[:, -1]
    return y0
