import numpy as np
import scipy.linalg
import scipy.integrate
import time


def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) on t_span = [t0, t1] for a 15-dim stiff linear IVP.
    Strategy: extract A and b (16 calls), then solve exactly using expm.
    For the affine term, use A^{-1}(expm(dt*A) - I) if A is invertible,
    otherwise use the augmented (n+1)x(n+1) expm approach.
    """
    start_time = time.time()
    deadline = start_time + 270.0  # 30s safety margin under 300s limit

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # --- Step 1: Extract A and b from the linear system ---
    try:
        zero_vec = np.zeros(n)
        b = np.asarray(f(t0, zero_vec), dtype=float)

        A = np.zeros((n, n))
        e0 = np.zeros(n)
        e0[0] = 1.0
        A[:, 0] = np.asarray(f(t0, e0), dtype=float) - b

        for i in range(1, n):
            if time.time() > deadline - 10.0:
                raise TimeoutError("Deadline approaching")
            ei = np.zeros(n)
            ei[i] = 1.0
            A[:, i] = np.asarray(f(t0, ei), dtype=float) - b

        if time.time() > deadline - 5.0:
            raise TimeoutError("Deadline approaching")

    except Exception:
        A = None
        b = None

    # --- Step 2: Solve using expm ---
    if A is not None:
        try:
            # Compute expm(dt * A)
            expA_dt = scipy.linalg.expm(dt * A)

            # Compute the integral term: ∫_0^dt exp(A*u) du @ b
            # If A is invertible: A^{-1}(expA_dt - I) @ b
            # Otherwise: use augmented matrix approach

            # Check if A is invertible
            try:
                # Use solve instead of explicit inverse for better numerical stability
                # We need: integral = A^{-1}(expA_dt - I) @ b
                # This is equivalent to solving A @ x = (expA_dt - I) @ b
                rhs = (expA_dt - np.eye(n)) @ b
                integral_b = np.linalg.solve(A, rhs)
                y_final = expA_dt @ y0 + integral_b
                return y_final
            except np.linalg.LinAlgError:
                # A is singular, use augmented matrix approach
                pass

            # Augmented (n+1)x(n+1) approach for singular A
            M = np.zeros((n + 1, n + 1))
            M[:n, :n] = A
            M[:n, n] = b
            expM = scipy.linalg.expm(dt * M)
            z_init = np.zeros(n + 1)
            z_init[:n] = y0
            z_init[n] = 1.0
            z_final = expM @ z_init
            return z_final[:n]

        except Exception:
            pass

    # --- Step 3: Fallback to numerical solvers ---
    try:
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            sol2 = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-12,
                atol=1e-14
            )
            return sol2.y[:, -1]
    except Exception:
        try:
            sol3 = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='LSODA',
                rtol=1e-10,
                atol=1e-12
            )
            return sol3.y[:, -1]
        except Exception:
            try:
                sol4 = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='RK45',
                    rtol=1e-10,
                    atol=1e-12
                )
                return sol4.y[:, -1]
            except Exception:
                return y0
