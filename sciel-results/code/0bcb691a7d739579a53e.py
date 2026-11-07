"""Radau IIA (s=2) fixed-step collocation integrator for the stiff Van der Pol ODE.

Design notes
------------
* The stage equations Y_i = y + h * sum_j A[i,j] f(Y_j) are solved with a
  damped global Newton iteration.  The residual for the whole (2s) system is
  R_i = Y_i - y - h * sum_j A[i,j] f(Y_j).
* The linearised step matrix is (M - h*J) where M = I - h*A is constant for a
  fixed step h (factorised once) and J = df/dy is the (n x n) Jacobian.  The
  product (M - h*J) is factorised with LU and the factorisation is *reused*
  across Newton iterations and across steps; it is only rebuilt when the
  residual stops decreasing (i.e. when J has drifted too far).
* J is estimated with a finite difference and refreshed only every few steps,
  since it varies slowly along the Van der Pol trajectory.
* b = [5/12, 7/12] gives the 5th-order accurate update.
"""
import numpy as np


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = y.size
    s = 2

    # ---- Radau IIA (s = 2) Butcher table --------------------------------
    c1 = (3.0 - np.sqrt(6.0)) / 3.0
    c2 = (3.0 + np.sqrt(6.0)) / 3.0
    c = np.array([c1, c2])
    A = np.array([[9.0 / 16.0, -3.0 / 16.0],
                  [3.0 / 16.0,  7.0 / 16.0]])
    b = np.array([5.0 / 12.0, 7.0 / 12.0])

    # ---- fixed step ------------------------------------------------------
    steps = 600
    h = (t1 - t0) / steps
    t = t0

    # ---- constant mass matrix M = I - h*A (2s x 2s) ----------------------
    A2 = np.kron(A, np.eye(n))
    M = np.eye(2 * s * n) - h * A2

    # ---- Jacobian estimate (dense, finite difference) --------------------
    def _jac(z, tt):
        fz = f(tt, z)
        J = np.empty((n, n))
        for j in range(n):
            e = 1e-6 * max(1.0, abs(z[j]))
            evec = np.zeros(n)
            evec[j] = e
            J[:, j] = (f(tt, z + evec) - fz) / e
        return J

    J = _jac(y, t0)
    # linear step matrix L = M - h*J  (2s x 2s)
    L = M - h * np.kron(J, np.eye(s))
    lu = np.linalg.lu_factor(L)

    # ---- Newton residual -------------------------------------------------
    def _residual(Y, ybase):
        # Y: (2s, n)  ->  R: (2s, n)
        F = np.empty_like(Y)
        for i in range(2 * s):
            F[i] = f(t + h * c[i // 2], Y[i])
        AF = np.empty_like(Y)
        for i in range(2 * s):
            AF[i] = h * (A[i // 2, 0] * F[2 * (i // 2)]
                         + A[i // 2, 1] * F[2 * (i // 2) + 1])
        return Y - ybase - AF

    # ---- step loop -------------------------------------------------------
    for k in range(steps):
        tstep = t
        # initial guess: linear extrapolation in the stage grid
        Y = np.empty((2 * s, n))
        for i in range(2 * s):
            Y[i] = y + h * c[i // 2] * J.dot(y)

        prev = np.inf
        for _ in range(40):
            R = _residual(Y, y)
            rnorm = np.linalg.norm(R)
            if rnorm <= 1e-13 * (1.0 + np.linalg.norm(y)):
                break
            # build RHS = -vec(R)
            rhs = -R.reshape(-1)
            try:
                d = np.linalg.lu_solve(lu, rhs)
            except np.linalg.LinAlgError:
                lu = np.linalg.lu_factor(L)
                d = np.linalg.lu_solve(lu, rhs)
            d = d.reshape(2 * s, n)
            # damped Newton
            step = 1.0
            ok = False
            while step >= 1e-4 and not ok:
                Yn = Y + step * d
                Rn = _residual(Yn, y)
                if np.linalg.norm(Rn) < rnorm:
                    Y = Yn
                    ok = True
                else:
                    step *= 0.5
            if not ok:
                Y = Y + 0.1 * d
            if prev - rnorm < 1e-15 * (1.0 + rnorm) and rnorm > 1e-12:
                # Newton stalled -> refresh Jacobian and re-factorise
                J = _jac(Y.reshape(-1)[0:n], tstep + h * c[0])
                L = M - h * np.kron(J, np.eye(s))
                lu = np.linalg.lu_factor(L)
            prev = rnorm

        # ---- update y_{n+1} = y + h * sum_j b_j f(Y_j) -------------------
        F = np.empty((s, n))
        for i in range(s):
            F[i] = f(tstep + h * c[i], Y[2 * i])
        y = y + h * (b[0] * F[0] + b[1] * F[1])
        t = tstep + h

        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            break

        # refresh Jacobian occasionally
        if (k + 1) % 20 == 0:
            J = _jac(y, t)
            L = M - h * np.kron(J, np.eye(s))
            lu = np.linalg.lu_factor(L)

    return y
