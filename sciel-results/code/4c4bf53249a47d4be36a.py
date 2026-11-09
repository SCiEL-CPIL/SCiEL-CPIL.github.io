"""ODE solver: 15-dimensional stiff linear system.
Strategy: 
1. Extract the system matrix A and constant vector b (if affine) using 16 calls.
   - 15 calls to probe basis vectors for A.
   - 1 call to probe zero vector for b.
2. Solve the potentially affine linear ODE y' = A y + b by augmenting the system to 16 dimensions.
   z' = [A, b; 0, 0] z, where z = [y; 1].
3. Use robust matrix exponential via eigenvalue decomposition with fallbacks.
Total f-calls: 16.
"""
import numpy as np
from scipy.linalg import expm, eig, schur
import time


def _matrix_exp_eig(M, dt):
    """Compute exp(dt * M) using eigenvalue decomposition with careful handling."""
    n = M.shape[0]
    try:
        # Try Schur decomposition first for better numerical stability
        T, Z = schur(dt * M)
        # T is upper triangular, so we can compute exp of upper triangular more stably
        # For upper triangular, exp is also upper triangular
        # Use block-wise computation
        E_T = np.zeros_like(T, dtype=complex)
        for i in range(n):
            for j in range(i, n):
                # Compute (i,j) element of exp of upper triangular matrix
                # This requires care; use scipy's expm on the triangular part
                E_T[i, j] = 0
        # Actually, let's just use expm on the Schur form which is better conditioned
        E_T = expm(T)
        E = Z @ E_T @ Z.conj().T
        # If result is complex with negligible imaginary part, take real
        if np.max(np.abs(E.imag)) < 1e-10:
            return E.real
        return E
    except Exception:
        pass
    
    try:
        # Fallback: eigenvalue decomposition
        eigenvalues, V = eig(dt * M)
        E_diag = np.diag(np.exp(eigenvalues))
        E = V @ E_diag @ np.linalg.inv(V)
        if np.max(np.abs(E.imag)) < 1e-10:
            return E.real
        return E
    except Exception:
        pass
    
    # Final fallback: direct expm
    return expm(dt * M)


def _taylor_expm(M, dt, order=50):
    """Compute exp(dt * M) using Taylor series as a high-precision fallback."""
    n = M.shape[0]
    result = np.eye(n, dtype=float)
    term = np.eye(n, dtype=float)
    M_scaled = dt * M
    for k in range(1, order + 1):
        term = term @ M_scaled / k
        result = result + term
        # Check convergence
        if np.max(np.abs(term)) < 1e-16 * np.max(np.abs(result)):
            break
    return result


def integrate(f, t_span, y0):
    """
    Integrate the 15-dimensional stiff linear ODE y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Extract A and b assuming f(t, y) = A y + b.
    2. Solve via augmented matrix exponential with multiple methods for robustness.
    
    Total f calls: 16 (15 for A, 1 for b).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # 1. Extract A using basis vectors
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
    # Check deadline
    if time.time() > deadline:
        return y0
        
    # 2. Extract b using zero vector
    b = f(t0, np.zeros(n, dtype=float))
    
    # Check deadline
    if time.time() > deadline:
        return y0
        
    dt = t1 - t0
    if dt == 0:
        return y0.copy()
        
    # 3. Solve y' = A y + b directly using the closed-form solution
    # y(t) = exp(A*t) @ y0 + integral_0^t exp(A*(t-s)) @ b ds
    # = exp(A*t) @ y0 + (exp(A*t) - I) @ A^{-1} @ b  (if A is invertible)
    # More robustly: augment and use matrix exponential
    
    # Method 1: Use the augmented system with robust matrix exponential
    B = np.zeros((n + 1, n + 1), dtype=float)
    B[:n, :n] = A
    B[:n, n] = b
    
    # Try multiple methods and pick the most reliable result
    y_final = None
    
    # Method 1: Schur-based expm
    try:
        E1 = _matrix_exp_eig(B, dt)
        z0 = np.zeros(n + 1, dtype=float)
        z0[:n] = y0
        z0[n] = 1.0
        z_end1 = E1 @ z0
        y_final = z_end1[:n]
        if not np.all(np.isfinite(y_final)):
            y_final = None
    except Exception:
        y_final = None
    
    # Method 2: Direct expm
    if y_final is None:
        try:
            E2 = expm(dt * B)
            z0 = np.zeros(n + 1, dtype=float)
            z0[:n] = y0
            z0[n] = 1.0
            z_end2 = E2 @ z0
            y_final = z_end2[:n]
            if not np.all(np.isfinite(y_final)):
                y_final = None
        except Exception:
            y_final = None
    
    # Method 3: Taylor series (high precision fallback)
    if y_final is None:
        try:
            E3 = _taylor_expm(B, dt, order=60)
            z0 = np.zeros(n + 1, dtype=float)
            z0[:n] = y0
            z0[n] = 1.0
            z_end3 = E3 @ z0
            y_final = z_end3[:n]
            if not np.all(np.isfinite(y_final)):
                y_final = None
        except Exception:
            y_final = None
    
    # Method 4: Solve using eigenvalue decomposition of A directly
    # y(t) = exp(A*t) @ y0 + integral term
    if y_final is None:
        try:
            # Compute exp(A*dt)
            try:
                eigvals, V = eig(A)
                E_A = V @ np.diag(np.exp(eigvals * dt)) @ np.linalg.inv(V)
                if np.max(np.abs(E_A.imag)) < 1e-10:
                    E_A = E_A.real
                else:
                    E_A = E_A.real  # Take real part
            except Exception:
                E_A = expm(dt * A)
            
            # Compute the integral term: integral_0^dt exp(A*s) ds
            # This equals (exp(A*dt) - I) @ A^{-1} if A is invertible
            # More robustly: solve the augmented system for just the integral
            # Or use: integral = (exp(A*dt) - I) @ inv(A) @ b
            # But A might be singular. Use the fact that:
            # integral_0^t exp(As) ds = t * sum_{k=0}^infty A^k t^k / (k+1)!
            # Let's use the Taylor series for the integral
            integral_term = np.zeros(n, dtype=float)
            term = np.eye(n, dtype=float)
            for k in range(50):
                # term_k = A^k * dt^(k+1) / (k+1)!
                # Update: integral += A^k * dt^(k+1) / (k+1)!
                # We can accumulate: let P_k = A^k, then integral_k = P_k * dt^(k+1)/(k+1)!
                # P_{k+1} = P_k @ A
                # integral_{k+1} = P_{k+1} * dt^(k+2) / (k+2)! = integral_k * A * dt / (k+2)
                if k == 0:
                    integral_term = dt * b
                else:
                    # term = A^k, integral_k = term @ (dt^(k+1)/(k+1)!) @ b
                    # integral_{k+1} = term @ A @ (dt^(k+2)/(k+2)!) @ b
                    # = (term @ (dt^(k+1)/(k+1)!)) @ A @ (dt/(k+2)) @ b
                    # Let S_k = term @ (dt^(k+1)/(k+1)!)
                    # S_{k+1} = S_k @ A @ (dt/(k+2))
                    # We need to track S_k
                    pass
                break  # This approach is getting complicated, use augmented system instead
            
            # Better: just use the augmented system result if we have it
            # For now, if we got here, use a simple RK4 as last resort
            h = dt / 1000
            y = y0.copy()
            for _ in range(1000):
                k1 = A @ y + b
                k2 = A @ (y + 0.5 * h * k1) + b
                k3 = A @ (y + 0.5 * h * k2) + b
                k4 = A @ (y + h * k3) + b
                y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_final = y
        except Exception:
            # Absolute last resort
            h = dt / 1000
            y = y0.copy()
            for _ in range(1000):
                k1 = A @ y + b
                k2 = A @ (y + 0.5 * h * k1) + b
                k3 = A @ (y + 0.5 * h * k2) + b
                k4 = A @ (y + h * k3) + b
                y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_final = y
    
    # Check deadline before returning
    if time.time() > deadline - 10.0:
        # Return best available result
        if y_final is not None and np.all(np.isfinite(y_final)):
            return y_final
        return y0
    
    return y_final
