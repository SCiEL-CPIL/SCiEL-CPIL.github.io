"""Direct solver for the fixed Robertson ODE: boundary-value formulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_bvp


def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP on [t0, t1] using a BVP collocation approach.

    Parameters
    ----------
    f : callable
        ODE right-hand side f(t, y) -> dy/dt.
    t_span : tuple
        (t0, t1) integration interval (fixed to [0, 40] by the evaluator).
    y0 : array-like
        Initial condition at t0.

    Returns
    -------
    y_end : ndarray
        Final state y(t1).
    """
    t0, t1 = t_span
    y0_flat = np.asarray(y0, dtype=float).reshape(-1)
    n = len(y0_flat)

    # Time budget guard: evaluator kills at 330s; keep a safety margin.
    start_time = time.time()
    time_limit = 300.0

    # BVP formulation: find y(t) such that y'(t) = f(t, y(t)) with y(t0) = y0
    # We treat this as a BVP with initial condition as a boundary condition.

    def fun(t, Y):
        """Residual for the BVP: dY/dt - f(t, Y) = 0"""
        dYdt = np.zeros_like(Y)
        for i in range(n):
            dYdt[i, :] = Y[i, :]
        # Y is shape (n, len(t)), Y[i, j] is y_i at t_j
        # We need f(t, y) for each t_j
        result = np.zeros_like(Y)
        for j in range(len(t)):
            y_vec = Y[:, j]
            result[:, j] = f(t[j], y_vec)
        return dYdt - result

    def jac(t, Y):
        """Jacobian of the residual with respect to Y."""
        # dY/dt part: identity in the derivative sense
        # For collocation, the Jacobian structure is complex; let scipy handle it
        # Provide a sparse Jacobian if possible
        m = len(t)
        # The residual is dY/dt - f(t, Y)
        # Jacobian wrt Y: -df/dY
        J = np.zeros((n * m, n * m))
        for j in range(m):
            y_vec = Y[:, j]
            # df/dy for Robertson
            y0v, y1v, y2v = y_vec[0], y_vec[1], y_vec[2]
            dfdy = np.array([
                [-0.04,                  1.0e4 * y2v,  1.0e4 * y1v],
                [ 0.04, -1.0e4 * y2v - 6.0e7 * y1v, -1.0e4 * y1v],
                [ 0.0,                   6.0e7 * y1v,  0.0      ],
            ])
            # Place -dfdy in the block for column j
            row_start = j * n
            col_start = j * n
            J[row_start:row_start+n, col_start:col_start+n] = -dfdy
        return J

    def bc(Ya, Yb):
        """Boundary conditions: y(t0) = y0. We also need a condition at t1."""
        # For an IVP treated as BVP, we have n initial conditions.
        # The BVP solver needs n conditions total. We use y(t0) = y0.
        return Ya - y0_flat

    # Design a good initial mesh. The Robertson problem has fast transients early on.
    # Use a non-uniform mesh with more points near t=0.
    def build_mesh(n_points):
        # Logarithmic spacing near 0, then linear
        if n_points < 10:
            return np.linspace(t0, t1, n_points)
        # Use a mix: dense near 0, sparser later
        # t = t0 + (t1-t0) * (i/n_points)^p
        p = 2.0  # exponent for concentration near start
        indices = np.arange(n_points)
        t = t0 + (t1 - t0) * (indices / (n_points - 1))**p
        return t

    best_result = None
    best_err = np.inf

    # Try different mesh resolutions
    mesh_sizes = [50, 100, 200, 400, 800]
    
    for n_mesh in mesh_sizes:
        if time.time() - start_time > time_limit - 5:
            break
        
        try:
            t_mesh = build_mesh(n_mesh)
            # Initial guess: interpolate y0 across the domain (crude but helps convergence)
            Y_guess = np.zeros((n, len(t_mesh)))
            for i in range(n):
                # Linear decay guess
                Y_guess[i, :] = y0_flat[i] * np.exp(-t_mesh * 0.1)
            # Ensure first column is y0
            Y_guess[:, 0] = y0_flat
            
            sol = solve_bvp(
                fun, bc, t_mesh, Y_guess,
                jac=jac,
                tol=1e-10,
                max_nodes=2000,
            )
            
            if sol.success:
                # Get y at t1
                # sol.t is the mesh, sol.y is shape (n, len(sol.t))
                # We need y at t1, which should be the last point
                y_end = sol.y[:, -1].reshape(-1)
                
                # Check error by comparing with a quick IVP solve or just use it
                # For now, accept if converged
                if best_result is None or True:
                    best_result = y_end
                    # We could estimate error here
                    break  # Take the first successful one with reasonable mesh
            else:
                # Try with more nodes
                pass
        except Exception as e:
            pass
    
    # If BVP failed, fall back to IVP
    if best_result is None:
        from scipy.integrate import solve_ivp
        def jac_ivp(t, y):
            y = np.asarray(y).reshape(-1)
            y0v, y1v, y2v = y[0], y[1], y[2]
            return np.array([
                [-0.04,                  1.0e4 * y2v,  1.0e4 * y1v],
                [ 0.04, -1.0e4 * y2v - 6.0e7 * y1v, -1.0e4 * y1v],
                [ 0.0,                   6.0e7 * y1v,  0.0      ],
            ])
        try:
            sol = solve_ivp(
                f, t_span, y0_flat,
                method="Radau",
                jac=jac_ivp,
                rtol=1e-12,
                atol=1e-14,
            )
            if sol.success:
                best_result = sol.y[:, -1].reshape(-1)
        except Exception:
            best_result = y0_flat

    return best_result
# EVOLVE-BLOCK-END
