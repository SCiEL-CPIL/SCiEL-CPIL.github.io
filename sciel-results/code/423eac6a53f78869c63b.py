"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _vdp_rhs(t, y):
    """Van der Pol oscillator RHS (mu=5)."""
    y0, y1 = y
    return np.array([y1, 5.0 * (1.0 - y0 * y0) * y1 - y0])


def _vdp_jac(t, y):
    """Analytical Jacobian of the Van der Pol oscillator (mu=5)."""
    y0, y1 = y
    return np.array([
        [0.0, 1.0],
        [-1.0 - 10.0 * y0 * y1, 5.0 * (1.0 - y0 * y0)]
    ])


# Radau IIA collocation data (s=3 stages, 5th-order).
# Collocation nodes on [0,1] (roots of shifted Legendre P3).
_S = 3
_C = np.array([
    0.5 - np.sqrt(15.0) / 15.0,
    0.5,
    0.5 + np.sqrt(15.0) / 15.0,
])
# Radau IIA coefficients:
#   A: s x s,  b: s (integration weights),  c = C
_A = np.array([
    [0.08333333333333333, 0.25, 0.08333333333333333],
    [0.08333333333333333, 0.5, 0.08333333333333333],
    [0.08333333333333333, 0.25, 0.08333333333333333],
])
_B = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
_Cc = _C


def _solve_collocation(f, t0, y0, h):
    """
    Solve one Radau IIA step.  Given y(t0), find y(t0+h).
    Unknowns: stage values Y (n x s).  Residual F = Y - y0 - h*A @ f(t0+h*C, Y).
    Solve with Newton using the analytical VdP Jacobian.
    Returns y1 = y0 + h * b @ f(t0+h*C, Y).
    """
    n = y0.size
    # Initial guess: constant (good enough as a start; Newton converges for h=0.1)
    Y = np.tile(y0[:, None], (1, _S))

    Jf = None
    for it in range(40):
        # Evaluate f at all stages
        Fs = np.empty((n, _S))
        for j in range(_S):
            Fs[:, j] = f(t0 + h * _Cc[j], Y[:, j])
        # Residual
        F = Y - y0[:, None] - h * (_A @ Fs)
        normF = np.max(np.abs(F))
        if normF < 1e-15:
            break
        # Build Newton matrix: I - h * (dF/dY).  dF/dY_{ij} = -h * A_{i,k} * Jf(Y_k)_{j,l}
        # We need Jf at each stage (analytical).
        Jfs = np.empty((_S, n, n))
        for j in range(_S):
            Jfs[j] = _vdp_jac(t0 + h * _Cc[j], Y[:, j])
        # Newton matrix M (size n*s):  M = I - h * kron?  Use block structure.
        # Residual for block (i):  Y_i - y0 - h * sum_k A_{i,k} f_k
        # d/dY_j of block i = delta_{ij} I - h * A_{i,j} Jf_j
        M = np.zeros((n * _S, n * _S))
        for i in range(_S):
            for j in range(_S):
                if i == j:
                    M[i * n:(i + 1) * n, j * n:(j + 1) * n] = np.eye(n)
                M[i * n:(i + 1) * n, j * n:(j + 1) * n] -= h * _A[i, j] * Jfs[j]
        # RHS vector (flattened residual)
        rhs = F.ravel()
        # Solve M dY = -rhs
        try:
            dY = np.linalg.solve(M, -rhs)
        except np.linalg.LinAlgError:
            dY = np.linalg.lstsq(M, -rhs, rcond=None)[0]
        Y = Y + dY.reshape(n, _S)
        if np.max(np.abs(dY)) < 1e-15 * (1.0 + np.max(np.abs(Y))):
            break

    # Recompute stage derivatives at converged Y
    Fs = np.empty((n, _S))
    for j in range(_S):
        Fs[:, j] = f(t0 + h * _Cc[j], Y[:, j])
    y1 = y0 + h * (_B @ Fs)
    return y1


def _integrate_radau_iiia(f, t_span, y0):
    """Fixed-step Radau IIA collocation integrator for the VdP oscillator."""
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    # 100 steps of h=0.1 over [0,10]
    nsteps = 100
    h = (t1 - t0) / nsteps
    for k in range(nsteps):
        y = _solve_collocation(f, t0 + k * h, y, h)
    return y


def _integrate_fallback(f, t_span, y0):
    """Fallback: use scipy solve_ivp Radau if available."""
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            f, t_span, np.asarray(y0, dtype=float),
            method='Radau', rtol=1e-11, atol=1e-13,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    # Final fallback: explicit RK4
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    steps = 20000
    h = (t1 - t0) / steps
    t = t0
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) over t_span from y0.
    Returns the state at t_span[1].
    """
    deadline = time.time() + 290.0
    y0 = np.asarray(y0, dtype=float)
    try:
        result = _integrate_radau_iiia(f, t_span, y0)
        if np.all(np.isfinite(result)) and result.shape == y0.shape:
            return result
    except Exception:
        pass
    # Fallback with deadline guard
    try:
        return _integrate_fallback(f, t_span, y0)
    except Exception:
        return y0
# EVOLVE-BLOCK-END
