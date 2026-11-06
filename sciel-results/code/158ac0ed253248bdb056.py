"""ODE solver for Heat2D 10x10 using eigen-decomposition of the Laplacian."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_heat_laplacian(n=10):
    """Build the 2D heat equation Laplacian matrix (discrete, 5-point stencil).
    
    For a 10x10 grid with Dirichlet BCs, the system is y' = L @ y
    where L has the standard 5-point Laplacian structure.
    We build it assuming the standard ordering and standard heat equation
    discretization with dx=dy=1 (or whatever the reference uses).
    
    Actually, we need to figure out the matrix from the function f.
    Since f(t, y) should be L @ y for the heat equation, we can extract L
    by probing f with basis vectors.
    """
    # We'll extract the matrix by calling f with identity columns
    # But that's n calls. Let's think of a better way.
    # Actually, for the heat equation, L is a fixed matrix. We can probe it.
    # But the task says f is given. Let's just probe with a few vectors
    # to reconstruct L, or better: assume it's the standard heat Laplacian.
    
    # For a 10x10 grid with the standard 5-point stencil:
    # L[i,j] where the grid is row-major: index = i*10 + j
    # L has -4 on diagonal, 1 on neighbors (up, down, left, right)
    # with Dirichlet BCs (boundary is zero, so interior points only)
    
    # Wait, 10x10 could mean 100 interior points, or 10x10 total grid.
    # The task says "10x10 two-dimensional heat equation" - likely 100 unknowns.
    # Let's build the standard 10x10 Laplacian (100x100 matrix).
    
    N = n
    size = N * N
    L = np.zeros((size, size))
    
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            L[idx, idx] = -4.0
            if i > 0:
                L[idx, (i-1)*N + j] = 1.0
            if i < N-1:
                L[idx, (i+1)*N + j] = 1.0
            if j > 0:
                L[idx, i*N + (j-1)] = 1.0
            if j < N-1:
                L[idx, i*N + (j+1)] = 1.0
    
    return L


def _extract_matrix_from_f(f, t, n):
    """Extract the matrix A from f(t, y) = A @ y by probing with basis vectors.
    
    This requires n calls to f. We use this to identify the system matrix.
    """
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t, e)
    return A


def _eigendecompose_solve(A, t_span, y0):
    """Solve y' = A@y using eigen-decomposition (exact for linear systems).
    
    y(t) = V @ diag(exp(lambda_i * (t - t0))) @ V^{-1} @ y0
    
    For symmetric A, V is orthogonal, so V^{-1} = V.T
    """
    t0, t1 = t_span
    dt = t1 - t0
    
    # Check if A is symmetric (within tolerance)
    if np.allclose(A, A.T, atol=1e-12):
        # Use eigendecomposition for symmetric matrix
        eigenvalues, V = np.linalg.eigh(A)
        # y(t1) = V @ (exp(eigenvalues * dt) * (V.T @ y0))
        y0_proj = V.T @ y0
        y_final = V @ (np.exp(eigenvalues * dt) * y0_proj)
    else:
        # Use general eigendecomposition
        eigenvalues, V = np.linalg.eig(A)
        V_inv = np.linalg.inv(V)
        y0_proj = V_inv @ y0
        y_final = V @ (np.exp(eigenvalues * dt) * y0_proj)
    
    return y_final


def _rk4(f, t0, t1, y0, steps):
    """Standard RK4 integration."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


def integrate(f, t_span, y0):
    """Solve the ODE y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy:
    1. Try to detect if the system is linear and time-independent (y' = A@y).
    2. If so, use eigen-decomposition for an exact solution (1 call for verification + n calls for matrix extraction).
    3. Fall back to RK4 with adaptive step size if linear detection fails.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin
    
    # Try to detect linear time-invariant system
    # First, check if f(t, y) depends on t at all
    # Probe: f(t0, y0) vs f(t1, y0)
    try:
        fy_t0 = f(t0, y0)
        fy_t1 = f(t1, y0)
        
        if np.allclose(fy_t0, fy_t1, rtol=1e-10, atol=1e-12):
            # Time-independent. Now check if it's linear: f(t, y) = A @ y
            # Probe with zero vector: if f(t, 0) = 0, it's linear (no constant term)
            zero_vec = np.zeros(n)
            f_zero = f(t0, zero_vec)
            
            if np.allclose(f_zero, np.zeros(n), atol=1e-12):
                # Appears to be linear and time-independent. Extract A.
                # Extract matrix by probing with basis vectors
                A = np.zeros((n, n))
                for j in range(n):
                    e = np.zeros(n)
                    e[j] = 1.0
                    A[:, j] = f(t0, e)
                
                # Verify linearity with a random test
                test_y = np.random.randn(n)
                test_f = f(t0, test_y)
                test_Ay = A @ test_y
                if np.allclose(test_f, test_Ay, rtol=1e-8, atol=1e-10):
                    # Confirmed linear. Use eigen-decomposition.
                    y_final = _eigendecompose_solve(A, t_span, y0)
                    if np.all(np.isfinite(y_final)):
                        return y_final
                # If verification fails, fall through to RK4
            # If not linear (has constant term), fall through to RK4
        # If time-dependent, fall through to RK4
    except Exception:
        pass
    
    # Fallback: RK4 with a reasonable number of steps
    # The heat equation is stiff, so we need many steps for explicit methods
    # But 2 million calls budget allows ~500k steps for RK4 (4 calls/step)
    # Let's use a moderate number of steps first, then refine if needed
    steps = 5000
    y = _rk4(f, t0, t1, y0, steps)
    
    # Check if we're still within deadline and can do a refinement
    elapsed = time.time() - start_time
    if elapsed < deadline - 50 and np.all(np.isfinite(y)):
        # Do a second pass with more steps to check convergence
        steps2 = steps * 2
        y2 = _rk4(f, t0, t1, y0, steps2)
        if np.all(np.isfinite(y2)):
            # Use Richardson extrapolation or just take the finer one
            y = y2
    
    return y
# EVOLVE-BLOCK-END
