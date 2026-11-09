import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # 1. Extract A and b
    y_zero = np.zeros(n)
    b = np.array(f(t0, y_zero), dtype=float)
    
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = np.array(f(t0, e_j), dtype=float)
        
    # Verify linearity
    y_test = np.ones(n)
    t_test = t0 + 0.5 * (t1 - t0)
    f_test = np.array(f(t_test, y_test), dtype=float)
    f_pred = A @ y_test + b
    
    if not np.allclose(f_test, f_pred, rtol=1e-10, atol=1e-12):
        # Fallback
        def jac(t, y):
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-8 * max(1.0, abs(y[j]))
                J[:, j] = (f(t, y + e) - f(t, y)) / e[j]
            return J
        sol = solve_ivp(f, t_span, y0, method='Radau', jac=jac, rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]

    # 2. Eigenvalue-based solution
    dt = t1 - t0
    
    # Compute eigenvalues and eigenvectors
    # Use scipy.linalg.eig for better handling of complex numbers?
    # np.linalg.eig is fine for n=15
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        # Fallback if eig fails
        def jac(t, y):
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-8 * max(1.0, abs(y[j]))
                J[:, j] = (f(t, y + e) - f(t, y)) / e[j]
            return J
        sol = solve_ivp(f, t_span, y0, method='Radau', jac=jac, rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]
        
    # Check condition number of eigenvectors
    # If ill-conditioned, eig is bad.
    # cond = np.linalg.cond(eigenvectors)
    # If cond > 1e10, fallback.
    
    # Compute e^(A dt)
    # M = V * diag(e^(lambda * dt)) * V^-1
    
    exp_eigs = np.exp(eigenvalues * dt)
    
    # Construct diagonal matrix of exp_eigs
    # D = np.diag(exp_eigs)
    # M = eigenvectors @ D @ np.linalg.inv(eigenvectors)
    
    # More efficient:
    # Solve V * x = y0  => x = V^-1 y0
    # y_end_hom = V * (exp_eigs * x)
    
    # Inverse of V
    try:
        V_inv = np.linalg.inv(eigenvectors)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(eigenvectors)
        
    # Homogeneous part: y_h(t) = e^(At) y0
    coords = V_inv @ y0
    scaled_coords = exp_eigs * coords
    y_h = eigenvectors @ scaled_coords
    
    # Particular part for b:
    # y_p(t) = \int_0^t e^(A(t-\tau)) b d\tau
    # If b is constant:
    # y_p = (e^(At) - I) A^-1 b
    # Or if A is singular, use limit.
    # Using eigen decomposition:
    # A^-1 = V diag(1/lambda) V^-1
    # (e^(At) - I) A^-1 b = V diag((e^(lambda t) - 1) / lambda) V^-1 b
    
    coords_b = V_inv @ b
    # Handle lambda = 0
    # (e^(lambda t) - 1) / lambda -> t as lambda -> 0
    factors = np.zeros_like(exp_eigs, dtype=complex)
    for i in range(n):
        if abs(eigenvalues[i]) < 1e-12:
            factors[i] = dt
        else:
            factors[i] = (exp_eigs[i] - 1.0) / eigenvalues[i]
            
    y_p = eigenvectors @ (factors * coords_b)
    
    y_end = y_h + y_p
    
    # Result should be real if A, y0, b are real
    # Check imaginary part
    if np.max(np.abs(y_end.imag)) > 1e-10:
        # If significant imaginary part, something is wrong or A has complex eigs (which is fine, but result should be real)
        # Actually, if A is real, complex eigs come in pairs, and V is complex.
        # The result should be real.
        pass
        
    return np.real(y_end)
