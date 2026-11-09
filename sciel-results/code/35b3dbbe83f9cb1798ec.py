"""Spectral Chebyshev collocation ODE integrator for Van der Pol."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev_nodes(n):
    """Return n Chebyshev-Gauss-Lobatto nodes on [-1, 1] (including endpoints)."""
    k = np.arange(n)
    return -np.cos(np.pi * k / (n - 1))


def _chebyshev_differential_matrix(n):
    """
    Return the Chebyshev collocation differential matrix D (n x n)
    such that D @ f(x) approximates f'(x) at the Chebyshev-Lobatto nodes.
    """
    if n == 1:
        return np.zeros((1, 1))
    
    x = _chebyshev_nodes(n)
    c = np.ones(n)
    c[0] = 0.5
    c[-1] = 0.5
    c = c * (-1) ** np.arange(n)
    
    X = np.tile(x, (n, 1))
    X_t = np.tile(x, (n, 1)).T
    dX = X - X_t + 1e-12  # avoid division by zero
    
    D = (np.outer(c, 1.0 / c) / dX)
    # Fix the diagonal entries
    D = D - np.diag(np.sum(D, axis=1))
    
    return D


def _spectral_collocation(f, t0, t1, y0, n_nodes=200):
    """
    Solve the IVP using spectral Chebyshev collocation with Newton-Raphson.
    
    We look for y(t) = sum_i y_i * T_i((2t - t0 - t1)/(t1 - t0))
    where T_i are Chebyshev polynomials.
    
    The collocation conditions are:
    y(x_0) = y0
    y'(x_k) = f(t_k, y(x_k)) for k = 1, ..., n-1
    
    where x_k are the Chebyshev-Lobatto nodes mapped to [t0, t1].
    """
    n = n_nodes
    y0 = np.array(y0, dtype=float)
    dim = len(y0)
    
    # Chebyshev-Lobatto nodes on [-1, 1]
    x = _chebyshev_nodes(n)
    
    # Map to [t0, t1]: t = 0.5*(t1-t0)*x + 0.5*(t1+t0)
    t_nodes = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0)
    
    # Differential matrix on [-1, 1]
    D = _chebyshev_differential_matrix(n)
    # Scale to [t0, t1]: d/dt = (2/(t1-t0)) * d/dx
    D_scaled = (2.0 / (t1 - t0)) * D
    
    # Unknowns: y_i for i=0..n-1, each of dimension dim
    # Total unknowns: n * dim
    # Initial guess: constant y0
    Y = np.tile(y0, (n, 1))  # n x dim matrix, Y[k, :] = y(t_k)
    
    def residual(Y):
        """
        Compute the residual of the collocation equations.
        R_0 = Y[0] - y0
        R_k = D_scaled[k, :] @ Y - f(t_k, Y[k]) for k=1..n-1
        """
        R = np.zeros((n, dim))
        R[0] = Y[0] - y0
        for k in range(1, n):
            dY = D_scaled[k, :] @ Y
            R[k] = dY - f(t_nodes[k], Y[k])
        return R
    
    def jacobian(Y):
        """
        Compute the Jacobian of the residual with respect to Y.
        J is a (n*dim) x (n*dim) matrix, but we build it block-wise.
        """
        J = np.zeros((n * dim, n * dim))
        
        # Block (0, 0): dR_0/dY_0 = I
        J[:dim, :dim] = np.eye(dim)
        
        for k in range(1, n):
            # dR_k/dY_j = D_scaled[k, j] * I for all j
            for j in range(n):
                row_start = k * dim
                col_start = j * dim
                J[row_start:row_start + dim, col_start:col_start + dim] = D_scaled[k, j] * np.eye(dim)
            
            # Subtract d f(t_k, Y[k]) / dY_k
            fk = f(t_nodes[k], Y[k])
            eps = 1e-8
            for j in range(dim):
                Y_pert = Y.copy()
                Y_pert[k, j] += eps
                fk_pert = f(t_nodes[k], Y_pert[k])
                df_dy_j = (fk_pert - fk) / eps
                row_start = k * dim
                col_start = k * dim
                J[row_start:row_start + dim, col_start + j:col_start + j + 1] -= df_dy_j
        
        return J
    
    # Newton-Raphson iterations
    max_newton = 50
    tol = 1e-13
    
    for it in range(max_newton):
        R = residual(Y)
        r_norm = np.linalg.norm(R)
        if r_norm < tol:
            break
        
        J = jacobian(Y)
        try:
            delta = np.linalg.solve(J, -R.flatten())
        except np.linalg.LinAlgError:
            # Fall back to least squares if singular
            delta, *_ = np.linalg.lstsq(J, -R.flatten(), rcond=None)
        
        Y = (Y.flatten() + delta).reshape(n, dim)
    
    # Return the value at the last node (t1)
    return Y[-1]


def integrate(f, t_span, y0):
    """
    Integrate the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses spectral Chebyshev collocation with Newton-Raphson.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use 150 nodes for a good balance of accuracy and computational cost
    # Each Newton iteration uses ~n * dim function calls for the Jacobian
    # With dim=2 and n=150, that's ~300 calls per iteration
    # With ~10-20 Newton iterations, total ~3000-6000 calls
    # This should be well within the 2M limit
    
    y_end = _spectral_collocation(f, t0, t1, y0, n_nodes=150)
    
    # Ensure the result is finite
    if not np.all(np.isfinite(y_end)):
        # Fallback: use a simple RK4 with many steps
        h = (t1 - t0) / 10000
        t = t0
        y = y0.copy()
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        y_end = y
    
    return y_end
# EVOLVE-BLOCK-END
