"""ODE solver: Convection-diffusion 100-point using solve_ivp with Radau and sparse Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def _build_sparse_jacobian(n=100, dx=1.0 / (n - 1)):
    """
    Build a sparse tridiagonal matrix representing the Jacobian of the
    convection-diffusion operator for a 100-point 1D problem.
    
    The standard discretization for -u_xx + u_x (or similar) gives a tridiagonal
    matrix. We construct a generic tridiagonal that matches the typical
    convection-diffusion stencil.
    """
    # For a standard convection-diffusion: 
    #   u_t = D * u_xx - c * u_x
    # The Jacobian of the RHS with respect to u is tridiagonal.
    # 
    # Let's use a reasonable tridiagonal structure.
    # Main diagonal: 2D/dx^2 (from diffusion)
    # Off-diagonals: -D/dx^2 +/- c/(2*dx) (from diffusion + convection)
    #
    # We'll use D=1, c=1 as a reasonable default, but the actual f will determine
    # the true Jacobian. However, providing a good approximation helps.
    
    D = 1.0
    c = 1.0
    
    main_diag = np.full(n, 2 * D / (dx * dx))
    upper_diag = np.full(n - 1, -D / (dx * dx) + c / (2 * dx))
    lower_diag = np.full(n - 1, -D / (dx * dx) - c / (2 * dx))
    
    J = diags([lower_diag, main_diag, upper_diag], offsets=[-1, 0, 1], format='csr')
    return J


def _counting_wrapper(f, counter):
    """Wrap f to count calls."""
    def wrapped(t, y):
        counter[0] += 1
        return f(t, y)
    return wrapped


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Set up deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 290s margin before 330s kill
    
    # Build sparse Jacobian approximation
    dx = 1.0 / (n - 1)
    J_sparse = _build_sparse_jacobian(n, dx)
    
    # Define the Jacobian function that returns the sparse matrix
    # We can update it if needed, but for now use a static approximation
    def jac(t, y):
        # Return a copy of the sparse Jacobian
        # The actual Jacobian might depend on y, but for convection-diffusion
        # with constant coefficients, it's constant
        return J_sparse.copy()
    
    # Use solve_ivp with Radau method and tight tolerances
    # Radau is an implicit Runge-Kutta method suitable for stiff problems
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            jac=jac,
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 100,  # Limit step size for accuracy
        )
        
        if sol.success:
            return sol.y[:, -1]
        
        # If Radau failed, try with looser tolerances
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            jac=jac,
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 50,
        )
        
        if sol.success:
            return sol.y[:, -1]
        
    except Exception:
        pass
    
    # Fallback: use LSODA if available
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='LSODA',
            rtol=1e-11,
            atol=1e-13,
            max_step=(t1 - t0) / 100,
        )
        
        if sol.success:
            return sol.y[:, -1]
        
    except Exception:
        pass
    
    # Final fallback: RK45 with many steps
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 1000,
        )
        
        if sol.success:
            return sol.y[:, -1]
        
    except Exception:
        pass
    
    # Last resort: simple RK4 with fixed steps
    h = (t1 - t0) / 5000
    t, y = t0, y0.copy()
    for _ in range(5000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
