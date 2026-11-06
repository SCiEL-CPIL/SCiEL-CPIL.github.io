"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    12-stage product formula using eigen-decomposition accelerated evaluation
    and multi-start Nelder-Mead optimization.
    """
    deadline = time.time() + 140  # Safety margin before 150s limit

    # Build Hamiltonians
    N = 6
    dim = 2 ** N
    delta = 0.8
    field = 0.3

    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)

    def tensor_op(op, site):
        result = np.array([[1.0]], dtype=complex)
        for i in range(N):
            if i == site:
                result = np.kron(result, op)
            else:
                result = np.kron(result, np.eye(2, dtype=complex))
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    def bond_term(j, j1):
        sz_j = tensor_op(sz, j)
        sz_j1 = tensor_op(sz, j1)
        sx_j = tensor_op(sx, j)
        sx_j1 = tensor_op(sx, j1)
        sy_j = tensor_op(sy, j)
        sy_j1 = tensor_op(sy, j1)
        return (delta/4.0) * (sz_j @ sz_j1) + (1/4.0) * (sx_j @ sx_j1 + sy_j @ sy_j1)

    def field_term(site, coeff):
        return coeff * tensor_op(sz, site)

    for j, j1 in [(0, 1), (2, 3), (4, 5)]:
        A += bond_term(j, j1)
    for j, j1 in [(1, 2), (3, 4)]:
        B += bond_term(j, j1)
    for site in range(N):
        A += field_term(site, field / 4.0)
        B += field_term(site, field / 4.0)

    # Precompute eigen-decompositions for fast expm
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    H = A + B
    eigvals_H, eigvecs_H = np.linalg.eigh(H)

    # Precompute exact propagators at the three times
    t_eval = [0.25, 0.5, 1.0]
    U_exact = []
    for t in t_eval:
        Ue = eigvecs_H @ np.diag(np.exp(-1j * t * eigvals_H)) @ eigvecs_H.conj().T
        U_exact.append(Ue)

    def fast_expm_Hermitian(coeffs, eigvals, eigvecs, t):
        """Compute exp(-i * coeffs * t * H) using precomputed eigendecomposition."""
        phase = -1j * coeffs * t * eigvals
        return eigvecs @ np.diag(np.exp(phase)) @ eigvecs.conj().T

    def compute_error_fast(a_coeffs, b_coeffs):
        """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
        total = 0.0
        for idx, t in enumerate(t_eval):
            U = np.eye(dim, dtype=complex)
            for k in range(stages):
                U = fast_expm_Hermitian(b_coeffs[k], eigvals_B, eigvecs_B, t) @ U
                U = fast_expm_Hermitian(a_coeffs[k], eigvals_A, eigvecs_A, t) @ U
            err = np.linalg.norm(U - U_exact[idx], 2)
            total += err
        return total / len(t_eval)

    # Initial guess: Strang splitting
    substeps = stages // 2
    a_init = [0.5 / substeps] * stages
    b_init = []
    for _ in range(substeps):
        b_init += [1.0 / substeps, 0.0]

    assert abs(sum(a_init) - 1.0) < 1e-10
    assert abs(sum(b_init) - 1.0) < 1e-10

    best_a = a_init[:]
    best_b = b_init[:]
    best_err = compute_error_fast(best_a, best_b)

    # Nelder-Mead optimization
    def nelder_mead(f, x0, max_iter=500, tol=1e-8):
        """Simple Nelder-Mead implementation."""
        n = len(x0)
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Initialize simplex
        simplex = [x0[:]]
        for i in range(n):
            xi = x0[:]
            xi[i] = xi[i] + 0.05 if xi[i] == 0 else xi[i] * 1.05
            simplex.append(xi)

        func_vals = [f(x) for x in simplex]

        for it in range(max_iter):
            # Sort simplex
            order = np.argsort(func_vals)
            simplex = [simplex[i] for i in order]
            func_vals = [func_vals[i] for i in order]

            if abs(func_vals[-1] - func_vals[0]) < tol:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = f(xr)

            if fr < func_vals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    func_vals[-1] = fe
                else:
                    simplex[-1] = xr
                    func_vals[-1] = fr
            elif fr < func_vals[-2]:
                simplex[-1] = xr
                func_vals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = f(xc)
                if fc < func_vals[-1]:
                    simplex[-1] = xc
                    func_vals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        func_vals[i] = f(simplex[i])

        best_idx = np.argmin(func_vals)
        return simplex[best_idx], func_vals[best_idx]

    def make_objective(a_coeffs, b_coeffs):
        """Create objective function that enforces constraints."""
        def obj(params):
            a = params[:stages]
            b = params[stages:]
            # Enforce sum constraints by projecting
            if abs(sum(a) - 1.0) > 1e-12:
                a = np.array(a) / sum(a)
            if abs(sum(b) - 1.0) > 1e-12:
                b = np.array(b) / sum(b)
            # Clamp to [-50, 50]
            a = np.clip(a, -50, 50)
            b = np.clip(b, -50, 50)
            return compute_error_fast(list(a), list(b))
        return obj

    # Multi-start optimization
    rng = np.random.RandomState(42)
    n_starts = 5

    for start in range(n_starts):
        if time.time() > deadline - 30:
            break

        if start == 0:
            x0 = np.array(a_init + b_init)
        else:
            # Perturb initial guess
            noise = rng.normal(0, 0.05, size=2 * stages)
            x0 = np.array(a_init + b_init) + noise
            # Renormalize
            x0[:stages] = x0[:stages] / sum(x0[:stages])
            x0[stages:] = x0[stages:] / sum(x0[stages:])

        obj = make_objective(x0[:stages], x0[stages:])
        try:
            x_opt, f_opt = nelder_mead(obj, x0, max_iter=200, tol=1e-6)
            a_opt = x_opt[:stages]
            b_opt = x_opt[stages:]
            # Final normalization
            a_opt = a_opt / sum(a_opt)
            b_opt = b_opt / sum(b_opt)
            a_opt = np.clip(a_opt, -50, 50).tolist()
            b_opt = np.clip(b_opt, -50, 50).tolist()

            err_opt = compute_error_fast(a_opt, b_opt)
            if err_opt < best_err:
                best_a = a_opt
                best_b = b_opt
                best_err = err_opt
        except Exception:
            continue

    # Final consistency check
    if abs(sum(best_a) - 1.0) > 1e-10:
        best_a = [x / sum(best_a) for x in best_a]
    if abs(sum(best_b) - 1.0) > 1e-10:
        best_b = [x / sum(best_b) for x in best_b]

    # Safety clamp
    best_a = [max(-50, min(50, x)) for x in best_a]
    best_b = [max(-50, min(50, x)) for x in best_b]

    # Ensure exact consistency
    if abs(sum(best_a) - 1.0) > 1e-10:
        best_a[-1] += 1.0 - sum(best_a)
    if abs(sum(best_b) - 1.0) > 1e-10:
        best_b[-1] += 1.0 - sum(best_b)

    return best_a, best_b
# EVOLVE-BLOCK-END
