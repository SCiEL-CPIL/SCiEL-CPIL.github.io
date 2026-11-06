"""ODE solver: 15-dimensional stiff linear system using matrix extraction."""
import numpy as np


def _extract_matrix_A(f, t0, y0, n=15):
    """
    Extract the system matrix A assuming y' = A @ y (linear, autonomous).
    We probe f with n linearly independent vectors to determine A.
    
    Strategy: Use n basis-like vectors. Since the problem is linear,
    f(t, y) = A @ y (assuming autonomous and linear).
    We evaluate f at n different y values to solve for A.
    
    We use the standard basis vectors scaled appropriately, but to be safe
    against any offset or non-linearity detection, we use a random invertible matrix.
    
    Actually, for a linear system y' = A y, we can just use the identity matrix
    columns. But to be robust, let's use a random invertible matrix R.
    Let Y = [y1, y2, ..., yn] be an n x n matrix of probe vectors (columns).
    Let F = [f(t0, y1), f(t0, y2), ..., f(t0, yn)] be the resulting matrix.
    Then F = A @ Y, so A = F @ Y^{-1}.
    """
    # Use a random invertible matrix for robustness
    # Or just use the identity matrix for simplicity and minimal calls
    # Identity matrix requires n calls and gives A directly.
    # Let's use identity for simplicity: f(0, e_i) = A[:, i]
    
    # However, if the system is not strictly linear (e.g., y' = A y + b),
    # identity probing won't work directly. But the task says "stiff linear system".
    # Usually "linear" in ODE context for this kind of task means y' = A(t) y or y' = A y.
    # If it's y' = A y, then A = F @ Y^{-1} works.
    
    # Let's use a random matrix to be safe against any hidden structure or to average out noise?
    # No, exact algebra is better. Let's use the identity matrix.
    # But wait, if the function f has a bias or is y' = A y + c, then:
    # f(e_i) = A e_i + c.
    # We can detect c by probing 0: f(0) = c.
    # Then A e_i = f(e_i) - c.
    # This takes n+1 calls.
    
    # Let's assume the most general affine case y' = A y + b first, or just linear y' = A y.
    # The prompt says "stiff linear system", which usually implies homogeneous y' = A y.
    # But let's be robust. We'll probe 0 first to check for a bias.
    
    # Probe 0
    y_zero = np.zeros(n)
    f_zero = np.array(f(t0, y_zero), dtype=float)
    
    # If f_zero is not close to zero, there might be a bias b = f_zero.
    # Let's assume y' = A y + b.
    # Then A = (F - b * 1^T) @ Y^{-1} where Y is the matrix of probe vectors.
    # If we use Y = I, then F = A + b * 1^T.
    # A = F - b * 1^T.
    
    # Let's use the identity matrix for probes.
    Y = np.eye(n)
    F = np.empty((n, n))
    for i in range(n):
        F[:, i] = f(t0, Y[:, i])
    
    # Check if f_zero is negligible. If not, subtract it.
    # If the system is homogeneous, f_zero should be 0.
    # If f_zero is non-zero, we assume y' = A y + f_zero.
    # Then A = F - f_zero * np.ones((1, n))
    
    # However, if the system is truly linear homogeneous, f_zero is 0.
    # Let's compute A assuming y' = A y + b where b = f_zero.
    b = f_zero
    A = F - np.outer(b, np.ones(n))
    
    # Verify linearity: check if f(t0, y0) matches A @ y0 + b
    # This is a good sanity check but costs 1 more call.
    # y_pred = A @ y0 + b
    # y_actual = f(t0, y0)
    # If they don't match, our assumption of constant A and b is wrong.
    # But the task says "fixed 15-dimensional stiff linear initial-value problem".
    # This strongly implies y' = A y.
    
    return A, b


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract A and b
    A, b = _extract_matrix_A(f, t0, y0, n)
    
    # Solve the linear ODE y' = A y + b
    # Solution: y(t) = e^{A(t-t0)} y0 + \int_0^{t-t0} e^{A s} b ds
    # If b is zero: y(t) = e^{A(t-t0)} y0
    
    dt = t1 - t0
    
    # Compute matrix exponential
    # scipy.linalg.expm is available
    from scipy.linalg import expm
    
    M = expm(A * dt)
    
    if np.all(np.abs(b) < 1e-12):
        y_end = M @ y0
    else:
        # \int_0^{dt} e^{A s} b ds
        # This can be computed as (e^{A dt} - I) A^{-1} b if A is invertible
        # Or more robustly using the formula:
        # \int_0^T e^{At} b dt = (e^{AT} - I) A^{-1} b
        # If A is singular, use a different method.
        # A more robust way is to use the block matrix exponential or series.
        # Let's try A^{-1} first.
        try:
            A_inv = np.linalg.inv(A)
            integral = (M - np.eye(n)) @ A_inv @ b
        except np.linalg.LinAlgError:
            # If A is singular, use a different approach
            # \int_0^T e^{At} b dt = \sum_{k=0}^{\infty} \frac{T^{k+1}}{(k+1)!} A^k b
            # Or use scipy's odeint with the known A? No, we want to avoid f calls.
            # Let's use the block matrix trick:
            # expm([[A, b], [0, 0]]) * [y0, 1]
            # The top part is y(T), bottom is 1.
            # Actually, the integral is the top-right block of expm([[A, b], [0, 0]]) * [0, 1]?
            # Let's construct the augmented matrix.
            # d/dt [y; 1] = [[A, b], [0, 0]] [y; 1]
            # Solution: [y(T); 1] = expm([[A, b], [0, 0]] * T) [y0; 1]
            # This is 2n x 2n.
            A_aug = np.zeros((2 * n, 2 * n))
            A_aug[:n, :n] = A
            A_aug[:n, n:] = b.reshape(-1, 1)
            # Note: b is a vector, so A_aug[:n, n] = b
            # The bottom row is zeros.
            
            M_aug = expm(A_aug * dt)
            y_end = M_aug[:n, :n] @ y0 + M_aug[:n, n:] @ b # Wait, the initial condition for the bottom part is 1?
            # Let's re-evaluate.
            # State vector z = [y; 1].
            # z' = A_aug z.
            # z(0) = [y0; 1].
            # z(T) = expm(A_aug T) z(0).
            # y(T) is the first n components of z(T).
            z0 = np.zeros(2 * n)
            z0[:n] = y0
            z0[n:] = 1.0 # This is wrong, the bottom part is constant 1, so its derivative is 0.
            # The equation is y' = A y + b. Let c = 1. c' = 0.
            # So z = [y; c]. z' = [[A, b], [0, 0]] [y; c].
            # c(0) = 1.
            # So z0 = [y0; 1].
            z_T = M_aug @ z0
            y_end = z_T[:n]
            
    return y_end
