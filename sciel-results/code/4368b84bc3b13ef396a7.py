import numpy as np
import time
import sys

def coefficients(stages):
    """
    Compute 12-stage product formula coefficients for Heisenberg chain time evolution.
    Returns (a, b) where each is a list of 12 floats summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

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

    # Precompute eigendecompositions
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    vals_AB, vecs_AB = np.linalg.eigh(A + B)

    # Precompute: vecs_A * exp(-i*t*vals_A) for faster matrix construction
    # We'll compute expm on the fly but use precomputed vecs
    times = [0.25, 0.5, 1.0]

    def expm_A(t):
        """exp(-i * t * A)"""
        return vecs_A @ (np.exp(-1j * t * vals_A)[:, None] * vecs_A.conj().T)

    def expm_B(t):
        """exp(-i * t * B)"""
        return vecs_B @ (np.exp(-1j * t * vals_B)[:, None] * vecs_B.conj().T)

    def expm_AB(t):
        """exp(-i * t * (A+B))"""
        return vecs_AB @ (np.exp(-1j * t * vals_AB)[:, None] * vecs_AB.conj().T)

    # Precompute exact propagators
    U_exact = [expm_AB(t) for t in times]

    def compute_error(a, b, t_idx):
        """Compute spectral norm of U(t) - exp(-i t (A+B)) for times[t_idx]"""
        t = times[t_idx]
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            U = expm_B(b[k] * t) @ expm_A(a[k] * t) @ U
        diff = U - U_exact[t_idx]
        s = np.linalg.svd(diff, compute_uv=False)
        return s[0]

    def objective(params):
        """
        params: 24-element array. First 12 are a, last 12 are b.
        """
        if time.time() > deadline:
            return 1e10

        a_full = params[:12]
        b_full = params[12:24]

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

        total_err = 0.0
        for t_idx in range(3):
            total_err += compute_error(a_full, b_full, t_idx)
        return total_err / 3.0

    # Nelder-Mead implementation with better parameters
    def nelder_mead(func, x0, max_iter=500, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        n = len(x0)
        simplex = [x0.copy() for _ in range(n + 1)]
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.copy()
            simplex[i][i - 1] += 0.01  # Smaller initial step for 24-dim

        fvals = [func(s) for s in simplex]

        for it in range(max_iter):
            if time.time() > deadline - 3:
                break

            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]

            if abs(fvals[0] - fvals[-1]) < tol:
                break

            centroid = np.mean(simplex[:n], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)

            if fr < fvals[0]:
                # Try expansion
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
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])

        return simplex[0], fvals[0]

    # Initial guess: use 4th-order Suzuki-Trotter coefficients as starting point
    # For 12 stages, we can use a symmetric 4th-order scheme
    # Standard 4th-order coefficients (Suzuki):
    # c0 = 1/(4*(1-2**(1/3)))  ~ -0.174760
    # c1 = -1/(2*(1-2**(1/3))) ~ 0.349521
    # c2 = 1/(2*(1-2**(1/3)))  ~ 0.674760
    # For 12 stages with symmetry, we can replicate a 4-step pattern
    # Or use uniform as fallback

    # Let's try a few different initializations
    inits = []

    # Uniform
    inits.append(np.ones(24) / 12)

    # 4th-order Suzuki pattern replicated
    c0 = 1.0 / (4.0 * (1.0 - 2.0 ** (1.0/3.0)))
    c1 = -1.0 / (2.0 * (1.0 - 2.0 ** (1.0/3.0)))
    c2 = 1.0 / (2.0 * (1.0 - 2.0 ** (1.0/3.0)))
    # Pattern: c0, c1, c2, c1, c0, c0, c1, c2, c1, c0, c0, c1, c2, c1, c0... 
    # For 12 stages, use: [c0, c1, c2, c1, c0, c0, c1, c2, c1, c0, c1, c2]
    # Actually let's just use a simple symmetric pattern
    a_suzuki = np.array([c0, c1, c2, c1, c0, c0, c1, c2, c1, c0, c1, c2])
    a_suzuki = a_suzuki / np.sum(a_suzuki)
    b_suzuki = a_suzuki.copy()
    inits.append(np.concatenate([a_suzuki, b_suzuki]))

    # Another pattern: 2nd order with refinement
    a_2nd = np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5])
    b_2nd = np.array([0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0])
    a_2nd = a_2nd / np.sum(a_2nd)
    b_2nd = b_2nd / np.sum(b_2nd)
    inits.append(np.concatenate([a_2nd, b_2nd]))

    # Random perturbations around uniform
    np.random.seed(42)
    for _ in range(3):
        x_rand = np.ones(24) / 12 + 0.05 * np.random.randn(24)
        inits.append(x_rand)

    # Evaluate all initializations and pick the best
    best_x = None
    best_obj = np.inf
    for x_init in inits:
        obj_val = objective(x_init)
        if obj_val < best_obj:
            best_obj = obj_val
            best_x = x_init.copy()

    # Run Nelder-Mead from best initialization
    try:
        res_x, res_f = nelder_mead(objective, best_x, max_iter=400, tol=1e-12)
        if res_f < best_obj:
            best_obj = res_f
            best_x = res_x
    except Exception:
        pass

    # Random restarts with more iterations
    np.random.seed(123)
    max_restarts = 8
    for r in range(max_restarts):
        if time.time() > deadline - 30:
            break
        # Perturb from current best
        x_rand = best_x + 0.02 * np.random.randn(24)
        try:
            res_x, res_f = nelder_mead(objective, x_rand, max_iter=200, tol=1e-12)
            if res_f < best_obj:
                best_obj = res_f
                best_x = res_x
        except Exception:
            pass

    # Final polish: very tight tolerance, small steps
    if time.time() < deadline - 15:
        try:
            # Create a tighter version with smaller initial simplex
            x_polish = best_x.copy()
            n_params = len(x_polish)
            simplex = [x_polish.copy() for _ in range(n_params + 1)]
            for i in range(1, n_params + 1):
                simplex[i] = x_polish.copy()
                simplex[i][i - 1] += 0.001

            fvals = [objective(s) for s in simplex]

            for it in range(100):
                if time.time() > deadline - 5:
                    break

                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]

                if abs(fvals[0] - fvals[-1]) < 1e-14:
                    break

                centroid = np.mean(simplex[:n_params], axis=0)
                xr = centroid + (centroid - simplex[-1])
                fr = objective(xr)

                if fr < fvals[0]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    xc = centroid + 0.5 * (centroid - simplex[-1])
                    fc = objective(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, n_params + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = objective(simplex[i])

            if fvals[0] < best_obj:
                best_obj = fvals[0]
                best_x = simplex[0]
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

    return [float(x) for x in a_full], [float(x) for x in b_full]
