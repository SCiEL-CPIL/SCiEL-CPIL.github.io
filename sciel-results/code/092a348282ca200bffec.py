import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s kill


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    n = 6
    dim = 2 ** n
    d = 0.8  # anisotropy
    h = 0.3  # field

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    ident = np.eye(2, dtype=complex)

    def op_on_site(op, site):
        ops = [ident] * n
        ops[site] = op
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result

    def bond_op(i, j):
        sx_i = op_on_site(sx, i)
        sx_j = op_on_site(sx, j)
        sy_i = op_on_site(sy, i)
        sy_j = op_on_site(sy, j)
        sz_i = op_on_site(sz, i)
        sz_j = op_on_site(sz, j)
        return sx_i @ sx_j + sy_i @ sy_j + d * sz_i @ sz_j

    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    even_bonds = [(1, 2), (3, 4)]

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    for i, j in odd_bonds:
        A += bond_op(i, j)
    for i, j in even_bonds:
        B += bond_op(i, j)

    half_field = 0.5 * h
    for site in range(n):
        A += half_field * op_on_site(sz, site)
        B += half_field * op_on_site(sz, site)

    H = A + B
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    eigvals_H, eigvecs_H = np.linalg.eigh(H)

    return A, B, (eigvals_A, eigvecs_A), (eigvals_B, eigvecs_B), (eigvals_H, eigvecs_H)


def _expm_hermitian(eigvals, eigvecs, coeff):
    """Compute exp(-i * coeff * H) given H = eigvecs @ diag(eigvals) @ eigvecs^H."""
    if abs(coeff) < 1e-15:
        dim = len(eigvals)
        return np.eye(dim, dtype=complex)
    diag_exp = np.exp(-1j * coeff * eigvals)
    return (eigvecs * diag_exp) @ eigvecs.conj().T


def _compute_error(a, b, eig_A, eig_B, eig_H, times):
    """Compute mean spectral norm error."""
    dim = eig_A[1].shape[0]
    I = np.eye(dim, dtype=complex)

    errors = []
    for t in times:
        U_exact = _expm_hermitian(eig_H[0], eig_H[1], t)
        U_approx = I.copy()
        for k in range(len(a)):
            if abs(b[k]) > 1e-15:
                exp_B = _expm_hermitian(eig_B[0], eig_B[1], b[k] * t)
                U_approx = exp_B @ U_approx
            if abs(a[k]) > 1e-15:
                exp_A = _expm_hermitian(eig_A[0], eig_A[1], a[k] * t)
                U_approx = exp_A @ U_approx
        err = np.linalg.norm(U_approx - U_exact, 2)
        errors.append(err)
    return np.mean(errors)


def _differential_evolution(objective, x0, bounds, pop_size=40, max_iter=80,
                           F=0.8, CR=0.9, tol=1e-10):
    """Simple differential evolution optimizer."""
    n = len(x0)
    lo, hi = bounds

    # Initialize population
    pop = np.zeros((pop_size, n))
    pop[0] = x0
    for i in range(1, pop_size):
        pop[i] = np.random.uniform(lo, hi, n)
        # Clamp
        pop[i] = np.clip(pop[i], lo, hi)

    fvals = np.array([objective(pop[i]) for i in range(pop_size)])

    for gen in range(max_iter):
        # Sort
        order = np.argsort(fvals)
        pop = pop[order]
        fvals = fvals[order]

        if fvals[0] < tol:
            break

        # Check stagnation
        if gen > 10 and abs(fvals[0] - fvals[-1]) < 1e-14:
            break

        # Generate new candidates
        for i in range(pop_size):
            # Select three distinct random indices different from i
            r1, r2, r3 = i
            while r1 == i:
                r1 = np.random.randint(pop_size)
            while r2 == i or r2 == r1:
                r2 = np.random.randint(pop_size)
            while r3 == i or r3 == r1 or r3 == r2:
                r3 = np.random.randint(pop_size)

            # Mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, lo, hi)

            # Crossover
            trial = pop[i].copy()
            j_rand = np.random.randint(n)
            for j in range(n):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)

            # Selection
            ft = objective(trial)
            if ft < fvals[i]:
                pop[i] = trial
                fvals[i] = ft

    return pop[0], fvals[0]


def coefficients(stages):
    """Optimize 12-stage product formula coefficients using differential evolution."""
    t_start = time.time()

    A, B, eig_A, eig_B, eig_H = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]

    n = stages  # 12
    half = n // 2  # 6

    # Precompute exact propagators
    U_exact = {}
    for t in times:
        U_exact[t] = _expm_hermitian(eig_H[0], eig_H[1], t)

    def objective(params):
        """params: 12 values (6 for a, 6 for b), palindromic expansion."""
        if time.time() > _DEADLINE - 5.0:
            return 1e10

        a_local = np.zeros(n)
        b_local = np.zeros(n)
        for k in range(half):
            a_local[k] = params[k]
            a_local[n - 1 - k] = params[k]
            b_local[k] = params[half + k]
            b_local[n - 1 - k] = params[half + k]

        # Sum constraints: sum(a)=1, sum(b)=1
        # With palindromic: 2*sum(params[0:half]) = 1 -> sum = 0.5
        # We'll handle this by normalizing
        sum_a = np.sum(a_local)
        sum_b = np.sum(b_local)

        if abs(sum_a) < 1e-12 or abs(sum_b) < 1e-12:
            return 1e10

        # Normalize to satisfy sum=1
        a_local = a_local / sum_a
        b_local = b_local / sum_b

        # Bounds check
        if np.any(np.abs(a_local) > 50) or np.any(np.abs(b_local) > 50):
            return 1e10

        return _compute_error(a_local, b_local, eig_A, eig_B, eig_H, times)

    # Define search bounds for the 12 free parameters
    lo = np.full(12, -5.0)
    hi = np.full(12, 5.0)

    # Multiple diverse initial points
    np.random.seed(42)

    # Strategy 1: Uniform (baseline)
    x1 = np.ones(12) * (1.0 / n)

    # Strategy 2: Strang-like pattern
    # a: [0.5/s, 0, 0.5/s, 0.5/s, 0, 0.5/s, 0.5/s, 0, 0.5/s, 0.5/s, 0, 0.5/s]
    # For palindromic half: a_half = [0.5/4, 0, 0.5/4, 0.5/4, 0, 0.5/4]
    # b: [0, 1/4, 0, 0, 1/4, 0] -> b_half = [0, 0.25, 0, 0, 0.25, 0]
    x2 = np.array([0.125, 0.0, 0.125, 0.125, 0.0, 0.125,
                   0.0, 0.25, 0.0, 0.0, 0.25, 0.0])

    # Strategy 3: Asymmetric - emphasize middle stages
    x3 = np.array([0.05, 0.10, 0.15, 0.20, 0.20, 0.15,
                   0.05, 0.10, 0.15, 0.20, 0.20, 0.15])

    # Strategy 4: Random perturbation of uniform
    x4 = np.ones(12) * (1.0 / n) + np.random.randn(12) * 0.1
    x4 = np.clip(x4, -2.0, 2.0)

    # Strategy 5: Another random
    x5 = np.random.uniform(-1.0, 2.0, 12)

    # Strategy 6: Negative coefficients at edges
    x6 = np.array([-0.1, 0.15, 0.2, 0.25, 0.25, 0.2,
                   -0.1, 0.15, 0.2, 0.25, 0.25, 0.2])

    initial_points = [x1, x2, x3, x4, x5, x6]

    best_x = None
    best_f = np.inf

    # Phase 1: Coarse search with smaller population
    for init_idx, x0 in enumerate(initial_points):
        if time.time() > _DEADLINE - 40.0:
            break

        try:
            x_opt, f_opt = _differential_evolution(
                objective, x0, (lo, hi),
                pop_size=25, max_iter=40, F=0.7, CR=0.85
            )
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
        except Exception:
            pass

    if best_x is None:
        # Fallback
        a = [1.0 / n] * n
        b = [1.0 / n] * n
        return a, b

    # Phase 2: Fine search around best
    if time.time() < _DEADLINE - 30.0:
        # Tighter bounds around best
        lo2 = best_x - 1.0
        hi2 = best_x + 1.0
        lo2 = np.maximum(lo2, lo)
        hi2 = np.minimum(hi2, hi)

        try:
            x_opt2, f_opt2 = _differential_evolution(
                objective, best_x, (lo2, hi2),
                pop_size=30, max_iter=50, F=0.5, CR=0.8
            )
            if f_opt2 < best_f:
                best_f = f_opt2
                best_x = x_opt2.copy()
        except Exception:
            pass

    # Phase 3: Very fine local refinement
    if time.time() < _DEADLINE - 15.0:
        lo3 = best_x - 0.3
        hi3 = best_x + 0.3
        lo3 = np.maximum(lo3, lo)
        hi3 = np.minimum(hi3, hi)

        try:
            x_opt3, f_opt3 = _differential_evolution(
                objective, best_x, (lo3, hi3),
                pop_size=20, max_iter=40, F=0.3, CR=0.7
            )
            if f_opt3 < best_f:
                best_f = f_opt3
                best_x = x_opt3.copy()
        except Exception:
            pass

    # Expand to full a, b arrays
    a_opt = np.zeros(n)
    b_opt = np.zeros(n)
    for k in range(half):
        a_opt[k] = best_x[k]
        a_opt[n - 1 - k] = best_x[k]
        b_opt[k] = best_x[half + k]
        b_opt[n - 1 - k] = best_x[half + k]

    # Normalize
    sum_a = np.sum(a_opt)
    sum_b = np.sum(b_opt)
    if abs(sum_a) > 1e-12:
        a_opt = a_opt / sum_a
    if abs(sum_b) > 1e-12:
        b_opt = b_opt / sum_b

    # Clamp to [-50, 50]
    a_opt = np.clip(a_opt, -50.0, 50.0)
    b_opt = np.clip(b_opt, -50.0, 50.0)

    # Re-normalize after clamping
    sum_a = np.sum(a_opt)
    sum_b = np.sum(b_opt)
    if abs(sum_a) > 1e-12:
        a_opt = a_opt / sum_a
    if abs(sum_b) > 1e-12:
        b_opt = b_opt / sum_b

    # Final validation
    a_list = a_opt.tolist()
    b_list = b_opt.tolist()

    # Ensure all finite
    if not all(np.isfinite(x) for x in a_list + b_list):
        a_list = [1.0 / n] * n
        b_list = [1.0 / n] * n

    return a_list, b_list
