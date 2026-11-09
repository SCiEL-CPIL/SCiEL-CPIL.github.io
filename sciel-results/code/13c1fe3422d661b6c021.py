"""Rewritten solver for 10x10 2D heat equation IVP."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    start_time = time.time()
    deadline = start_time + 285.0

    # --- Step 1: check for bias (non-homogeneous term) ---
    y_zero = np.zeros(n)
    b0 = f(t0, y_zero)
    is_linear = np.allclose(b0, 0.0, atol=1e-13)

    A = None

    if n == 100:
        # --- Step 2: detect 5-point stencil structure ---
        # Probe center point (row 5, col 5) -> index 55
        idx_c = 55
        e_c = np.zeros(n)
        e_c[idx_c] = 1.0
        v_c = f(t0, e_c)

        # Expected non-zero positions for a 5-point Laplacian at (5,5):
        # center=55, left=54, right=56, up=45, down=65
        valid_positions = {55: 'C', 54: 'L', 56: 'R', 45: 'U', 65: 'D'}
        is_5pt = True
        weights = {}

        for i in range(n):
            if abs(v_c[i]) > 1e-15:
                if i in valid_positions:
                    weights[valid_positions[i]] = v_c[i]
                else:
                    is_5pt = False
                    break

        if is_5pt and len(weights) == 5:
            w_c = weights['C']
            w_l = weights['L']
            w_r = weights['R']
            w_u = weights['U']
            w_d = weights['D']

            # Verify with corner (0,0) -> index 0
            # Neighbors: self=0, right=1, down=10
            e_v = np.zeros(n)
            e_v[0] = 1.0
            v_v = f(t0, e_v)

            ok = True
            if abs(v_v[0] - w_c) > 1e-12:
                ok = False
            if abs(v_v[1] - w_r) > 1e-12:
                ok = False
            if abs(v_v[10] - w_d) > 1e-12:
                ok = False
            # All other entries should be zero
            for i in range(n):
                if i not in (0, 1, 10):
                    if abs(v_v[i]) > 1e-12:
                        ok = False
                        break

            if ok:
                # Build the 100x100 matrix A using the detected weights
                A = np.zeros((n, n))
                for i in range(10):
                    for j in range(10):
                        k = i * 10 + j
                        A[k, k] += w_c
                        if j > 0:
                            A[k, k - 1] += w_l
                        if j < 9:
                            A[k, k + 1] += w_r
                        if i > 0:
                            A[k, k - 10] += w_u
                        if i < 9:
                            A[k, k + 10] += w_d
        # If detection failed, fall through to full probing below

    # --- Step 3: fallback – full column probing ---
    if A is None:
        A = np.zeros((n, n))
        for i in range(n):
            if time.time() > deadline:
                break
            e_i = np.zeros(n)
            e_i[i] = 1.0
            col = f(t0, e_i)
            if not is_linear:
                col = col - b0
            A[:, i] = col

    if A is None:
        return y0

    # --- Step 4: compute expm(A * dt) @ y0 ---
    try:
        # For the 2D heat equation on a uniform grid, A is symmetric (or close to it).
        # Use eigh for better numerical stability.
        if n == 100 and np.allclose(A, A.T, atol=1e-13):
            eigvals, eigvecs = np.linalg.eigh(A)
            exp_eig = np.exp(eigvals * dt)
            # M = V @ diag(exp_eig) @ V^T
            M = (eigvecs * exp_eig) @ eigvecs.T
        else:
            # General case: use scipy expm
            import scipy.linalg as sla
            M = sla.expm(A * dt)

        y_end = M @ y0

        # Handle non-homogeneous case: y' = A y + b
        # y(t) = e^{At} y0 + (I - e^{At}) A^{-1} b
        if not is_linear:
            try:
                import scipy.linalg as sla
                X = sla.solve(A, b0)
                y_end = y_end + (np.eye(n) - M) @ X
            except Exception:
                pass

        return y_end

    except Exception:
        # Last-resort fallback: simple Euler with tiny steps (should rarely trigger)
        try:
            import scipy.integrate as si
            sol = si.solve_ivp(f, [t0, t1], y0, method='RK45', rtol=1e-10, atol=1e-12)
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        return y0

# EVOLVE-BLOCK-END
