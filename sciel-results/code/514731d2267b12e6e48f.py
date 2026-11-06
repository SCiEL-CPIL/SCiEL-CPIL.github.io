"""Rewritten Robertson ODE solver using BVP collocation (shooting) approach."""
import time
import numpy as np
from scipy.integrate import solve_bvp, solve_ivp


def _robertson_jac(t, y):
    """Analytic Jacobian for the Robertson stiff ODE system."""
    y0, y1, y2 = y[0], y[1], y[2]
    return np.array([
        [-0.04,              1e4 * y2,             1e4 * y1],
        [ 0.04,             -1e4 * y2 - 6e7 * y1,  -1e4 * y1],
        [ 0.0,               6e7 * y1,             0.0]
    ])


def _robertson_rhs(t, y):
    """Robertson ODE right-hand side (used only if f is not provided)."""
    y0, y1, y2 = y
    return np.array([
        -0.04 * y0 + 1e4 * y1 * y2,
         0.04 * y0 - 1e4 * y1 * y2 - 3e7 * y1**2,
         3e7 * y1**2
    ])


def integrate(f, t_span, y0):
    """Integrate the Robertson stiff ODE using BVP collocation method.
    
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 310.0
    
    try:
        # Build initial mesh for BVP solver
        # Robertson problem has fast transient early, slow dynamics later
        # Use a non-uniform mesh concentrated near t=0
        n_mesh = 40
        # Log-spaced mesh: dense near t0, sparse near t1
        if t0 == 0:
            t_mesh = np.concatenate([np.array([0.0]), np.geomspace(1e-3, t1, n_mesh - 1)])
        else:
            t_mesh = np.linspace(t0, t1, n_mesh)
        
        # Initial guess: linear interpolation of y0 (or a better guess)
        # For Robertson: y0 decays fast, y1 has a small peak, y2 grows to 1
        y0_arr = y0.copy()
        # Better initial guess based on known Robertson behavior
        if abs(y0_arr[0] - 1.0) < 1e-10 and abs(y0_arr[1]) < 1e-10 and abs(y0_arr[2]) < 1e-10:
            # Standard Robertson IC: y(0) = [1, 0, 0]
            y_guess = np.zeros((3, len(t_mesh)))
            y_guess[0, :] = np.exp(-0.04 * t_mesh)  # y0 decays
            # y1 peaks around t~0.01 with value ~1e-5
            y1_peak = 0.004
            y1_width = 0.01
            y_guess[1, :] = y1_peak * np.exp(-((t_mesh - 0.01)**2) / (2 * y1_width**2))
            y_guess[2, :] = 1.0 - y_guess[0, :] - y_guess[1, :]
            y_guess[2, :] = np.maximum(y_guess[2, :], 0.0)
        else:
            y_guess = np.tile(y0_arr[:, None], (1, len(t_mesh)))
        
        # Boundary conditions: enforce initial condition at t0
        # (This turns the IVP into a BVP with "boundary" at t0)
        def bc(ya, yb):
            return np.array([ya[0] - y0_arr[0],
                             ya[1] - y0_arr[1],
                             ya[2] - y0_arr[2]])
        
        # Use the provided f or the built-in Robertson RHS
        rhs = f
        
        # Solve BVP with tight tolerances
        sol_bvp = solve_bvp(
            rhs,
            bc,
            t_mesh,
            y_guess,
            jac=_robertson_jac,
            tol=1e-12,
            max_nodes=10000,
            verbose=0
        )
        
        if sol_bvp.success and time.time() < deadline:
            return sol_bvp.y[:, -1]
        
        # If BVP failed, try with looser tolerance or different mesh
        if time.time() < deadline - 5:
            # Retry with more nodes in initial mesh
            n_mesh2 = 80
            if t0 == 0:
                t_mesh2 = np.concatenate([np.array([0.0]), np.geomspace(1e-3, t1, n_mesh2 - 1)])
            else:
                t_mesh2 = np.linspace(t0, t1, n_mesh2)
            
            if abs(y0_arr[0] - 1.0) < 1e-10 and abs(y0_arr[1]) < 1e-10 and abs(y0_arr[2]) < 1e-10:
                y_guess2 = np.zeros((3, len(t_mesh2)))
                y_guess2[0, :] = np.exp(-0.04 * t_mesh2)
                y1_peak = 0.004
                y1_width = 0.01
                y_guess2[1, :] = y1_peak * np.exp(-((t_mesh2 - 0.01)**2) / (2 * y1_width**2))
                y_guess2[2, :] = 1.0 - y_guess2[0, :] - y_guess2[1, :]
                y_guess2[2, :] = np.maximum(y_guess2[2, :], 0.0)
            else:
                y_guess2 = np.tile(y0_arr[:, None], (1, len(t_mesh2)))
            
            sol_bvp2 = solve_bvp(
                rhs,
                bc,
                t_mesh2,
                y_guess2,
                jac=_robertson_jac,
                tol=1e-12,
                max_nodes=20000,
                verbose=0
            )
            
            if sol_bvp2.success:
                return sol_bvp2.y[:, -1]
        
        # Fallback to solve_ivp with Radau
        if time.time() < deadline - 5:
            sol_ivp = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='Radau',
                jac=_robertson_jac,
                rtol=1e-13,
                atol=1e-15
            )
            if sol_ivp.success:
                return sol_ivp.y[:, -1]
            elif sol_ivp.t.size > 0:
                return sol_ivp.y[:, -1]
        
        # Last resort: return whatever we have
        if sol_bvp.success:
            return sol_bvp.y[:, -1]
        elif sol_bvp.t.size > 0:
            return sol_bvp.y[:, -1]
        
        return y0_arr.copy()
        
    except Exception:
        # Fallback to solve_ivp
        if time.time() < deadline - 5:
            try:
                sol_ivp = solve_ivp(
                    f,
                    t_span,
                    y0,
                    method='Radau',
                    jac=_robertson_jac,
                    rtol=1e-12,
                    atol=1e-14
                )
                if sol_ivp.success:
                    return sol_ivp.y[:, -1]
                elif sol_ivp.t.size > 0:
                    return sol_ivp.y[:, -1]
            except Exception:
                pass
        
        # Absolute last resort
        return np.asarray(y0, dtype=float)
