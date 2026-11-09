import numpy as np
import time
import sys

def coefficients(stages):
    """
    Compute 12-stage product formula coefficients for Heisenberg chain time evolution.
    Returns (a, b) where each is a list of 12 floats summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Build the Hamiltonians
    n = 6
    dim = 2 ** n
    delta = 0.8
    h = 0.3

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    Jz = 1.0
    Jxy = delta * Jz

    def add_bond(H, i, j):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        ops[j] = sz
        H += Jz * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sx
        ops[j] = sx
        H += (Jxy / 2) * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sy
        ops[j] = sy
        H += (Jxy / 2) * kron_all(ops)

    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        add_bond(A, i, j)

    for (i, j) in [(1, 2), (3, 4)]:
        add_bond(B, i, j)

    for i in range(n):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term

    # Precompute eigendecompositions for A, B, and A+B
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    vals_AB, vecs_AB = np.linalg.eigh(A + B)

    times = [0.25, 0.5, 1.0]

    def expm_A(t):
        """exp(-i * t * A)"""
        return vecs_A @ np.diag(np.exp(-1j * t * vals_A)) @ vecs_A.conj().T

    def expm_B(t):
        """exp(-i * t * B)"""
        return vecs_B @ np.diag(np.exp(-1j * t * vals_B)) @ vecs_B.conj().T

    def expm_AB(t):
        """exp(-i * t * (A+B))"""
        return vecs_AB @ np.diag(np.exp(-1j * t * vals_AB)) @ vecs_AB.conj().T

    # Precompute exact propagators
    U_exact = {}
    for t in times:
        U_exact[t] = expm_AB(t)

    def compute_error(a, b, t):
        """Compute spectral norm of U(t) - exp(-i t (A+B))"""
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U = expm_B(b[k] * t) @ expm_A(a[k] * t) @ U
        diff = U - U_exact[t]
        s = np.linalg.svd(diff, compute_uv=False)
        return s[0]

    def objective(params):
        """
        params: 24-element array. First 12 are a, last 12 are b.
        Returns the mean spectral norm error over the three times.
        """
        if time.time() > deadline:
            return 1e10

        a_full = params[:12].copy()
        b_full = params[12:24].copy()

        # Normalize to sum to 1
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        if abs(a_sum) < 1e-15:
            a_full = np.ones(12) / 12
        else:
            a_full = a_full / a_sum
        if abs(b_sum) < 1e-15:
            b_full = np.ones(12) / 12
        else:
            b_full = b_full / b_sum

        # Check bounds
        if np.any(a_full < -50) or np.any(a_full > 50):
            return 1e10
        if np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10

        # Compute errors at each time
        errs = []
        for t in times:
            errs.append(compute_error(a_full, b_full, t))
        errs = np.array(errs)

        return np.mean(errs)

    def objective_smooth(params):
        """
        Smooth objective using log-sum-exp over times for better gradient info.
        """
        if time.time() > deadline:
            return 1e10

        a_full = params[:12].copy()
        b_full = params[12:24].copy()

        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        if abs(a_sum) < 1e-15:
            a_full = np.ones(12) / 12
        else:
            a_full = a_full / a_sum
        if abs(b_sum) < 1e-15:
            b_full = np.ones(12) / 12
        else:
            b_full = b_full / b_sum

        if np.any(a_full < -50) or np.any(a_full > 50):
            return 1e10
        if np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10

        errs = []
        for t in times:
            errs.append(compute_error(a_full, b_full, t))
        errs = np.array(errs)

        # Log-sum-exp scaled to approximate the mean but smoother
        max_e = np.max(errs)
        if max_e < 1e-15:
            return 0.0
        lse = max_e + np.log(np.sum(np.exp(errs - max_e)))
        smooth_val = lse - np.log(len(times))
        return smooth_val

    # Nelder-Mead implementation
    def nelder_mead(func, x0, max_iter=300, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        n_dim = len(x0)
        simplex = [x0.copy() for _ in range(n_dim + 1)]
        simplex[0] = x0
        for i in range(1, n_dim + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += 0.015

        fvals = [func(s) for s in simplex]

        for it in range(max_iter):
            if time.time() > deadline - 10:
                break

            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            if abs(fvals[0] - fvals[-1]) < tol:
                break

            centroid = np.mean(simplex[:n_dim], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)

            if fr < fvals[0]:
                # Extension
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    xc = centroid + rho * (centroid - simplex[-1])
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                fc = func(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n_dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])

        return simplex[0], fvals[0]

    # Initialize with uniform weights
    x0 = np.ones(24) / 12.0

    best_x = x0.copy()
    best_obj = objective(x0)

    np.random.seed(42)

    # Phase 1: Coarse optimization with smooth objective
    try:
        res_x, res_f = nelder_mead(objective_smooth, x0, max_iter=250, tol=1e-8)
        if res_f < best_obj:
            best_obj = res_f
            best_x = res_x.copy()
    except Exception:
        pass

    # Phase 2: Refine with raw objective
    try:
        res_x, res_f = nelder_mead(objective, best_x, max_iter=200, tol=1e-10)
        if res_f < best_obj:
            best_obj = res_f
            best_x = res_x.copy()
    except Exception:
        pass

    # Phase 3: Random restarts around best solution
    max_restarts = 6
    for _ in range(max_restarts):
        if time.time() > deadline - 35:
            break
        x_rand = best_x + 0.008 * np.random.randn(24)
        try:
            res_x, res_f = nelder_mead(objective, x_rand, max_iter=120, tol=1e-10)
            if res_f < best_obj:
                best_obj = res_f
                best_x = res_x.copy()
        except Exception:
            pass

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 8:
        try:
            res_x, res_f = nelder_mead(objective, best_x, max_iter=400, tol=1e-12, alpha=0.9, gamma=1.8, rho=0.4, sigma=0.4)
            if res_f < best_obj:
                best_obj = res_f
                best_x = res_x.copy()
        except Exception:
            pass

    # Extract final coefficients
    a_full = best_x[:12]
    b_full = best_x[12:24]

    # Clip and renormalize
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum

    # Final verification and tiny adjustment if needed
    final_err = objective(best_x)
    if final_err > best_obj * 1.01 and time.time() < deadline - 5:
        # Try one more local nudge
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            x_try = best_x + 0.002 * np.random.randn(24)
            e_try = objective(x_try)
            if e_try < final_err:
                best_x = x_try
                final_err = e_try
                a_full = np.clip(best_x[:12], -50, 50)
                b_full = np.clip(best_x[12:24], -50, 50)
                a_sum = np.sum(a_full)
                b_sum = np.sum(b_full)
                if abs(a_sum) > 1e-12:
                    a_full = a_full / a_sum
                if abs(b_sum) > 1e-12:
                    b_full = b_full / b_sum

    return [float(x) for x in a_full], [float(x) for x in b_full]
