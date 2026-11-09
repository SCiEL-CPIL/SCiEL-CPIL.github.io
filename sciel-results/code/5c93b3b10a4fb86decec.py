"""Optimized 12-stage product formula for 6-site Heisenberg chain evolution."""
import numpy as np
import time

# Cache for built Hamiltonians
_HAM_CACHE = None
_EXACT_CACHE = {}

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    global _HAM_CACHE
    if _HAM_CACHE is not None:
        return _HAM_CACHE

    N = 6
    gamma = 0.8
    h = 0.3

    # Single-site operators
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    dim = 2 ** N
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    # Bonds: site i and i+1 for i = 0..N-2
    # Odd bonds: i = 0, 2, 4 (bond between sites 0-1, 2-3, 4-5)
    # Even bonds: i = 1, 3 (bond between sites 1-2, 3-4)
    # Actually let's define: bond i connects site i and site i+1
    # Odd-indexed bonds (0-based): bonds 0, 2, 4 -> sites (0,1), (2,3), (4,5)
    # Even-indexed bonds (0-based): bonds 1, 3 -> sites (1,2), (3,4)

    for i in range(N - 1):
        ops = [I] * N
        # XXZ interaction: gamma/2*(Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) + Sz_i Sz_{i+1}
        # Using spin-1/2: Sx = sx/2, Sy = sy/2, Sz = sz/2
        # So: gamma/4*(sx_i sx_{i+1} + sy_i sy_{i+1}) + 1/4 * sz_i sz_{i+1}
        # Standard Heisenberg: J * (Sx Sx + Sy Sy + Sz Sz) with anisotropy
        # XXZ: J * (Sx Sx + Sy Sy + gamma * Sz Sz)
        # With J=1: Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + gamma * Sz_i Sz_{i+1}
        # = 1/4*(sx_i sx_{i+1} + sy_i sy_{i+1}) + gamma/4 * sz_i sz_{i+1}

        ops[i] = sx
        ops[i+1] = sx
        term_xx = kron_all(ops)

        ops[i] = sy
        ops[i+1] = sy
        term_yy = kron_all(ops)

        ops[i] = sz
        ops[i+1] = sz
        term_zz = kron_all(ops)

        bond_ham = 0.25 * (term_xx + term_yy) + gamma * 0.25 * term_zz

        if i % 2 == 0:  # odd bonds (0-indexed: 0, 2, 4)
            A += bond_ham
        else:  # even bonds (0-indexed: 1, 3)
            B += bond_ham

    # Uniform field: h * sum_i Sz_i, split half to A and half to B
    for i in range(N):
        ops = [I] * N
        ops[i] = sz
        field_term = kron_all(ops)
        A += (h / 2) * field_term
        B += (h / 2) * field_term

    _HAM_CACHE = (A, B)
    return A, B

def _exact_propagator(t):
    """Compute exp(-i * t * (A + B)) using eigendecomposition."""
    if t in _EXACT_CACHE:
        return _EXACT_CACHE[t]

    A, B = _build_hamiltonians()
    H = A + B
    # H is Hermitian, so use eigh
    evals, evecs = np.linalg.eigh(H)
    phase = np.exp(-1j * t * evals)
    U = (evecs * phase) @ evecs.conj().T
    _EXACT_CACHE[t] = U
    return U

def _product_formula(a, b, t):
    """Compute U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)."""
    A, B = _build_hamiltonians()
    U = np.eye(A.shape[0], dtype=complex)

    for k in range(len(a)):
        # exp(-i * b_k * t * B)
        if abs(b[k]) * t > 1e-15:
            evals_b, evecs_b = np.linalg.eigh(B)
            phase_b = np.exp(-1j * b[k] * t * evals_b)
            U = (evecs_b * phase_b) @ (evecs_b.conj().T @ U)
        # exp(-i * a_k * t * A)
        if abs(a[k]) * t > 1e-15:
            evals_a, evecs_a = np.linalg.eigh(A)
            phase_a = np.exp(-1j * a[k] * t * evals_a)
            U = (evecs_a * phase_a) @ (evecs_a.conj().T @ U)

    return U

def _compute_error(a, b):
    """Compute mean spectral norm error at t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    total = 0.0
    for t in times:
        U_pf = _product_formula(a, b, t)
        U_exact = _exact_propagator(t)
        diff = U_pf - U_exact
        # Spectral norm = largest singular value
        s = np.linalg.svd(diff, compute_uv=False)
        total += s[0]
    return total / len(times)

def coefficients(stages):
    """Optimized 12-stage product formula coefficients via Nelder-Mead."""
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    # Build Hamiltonians (cached)
    A, B = _build_hamiltonians()

    # Precompute eigendecompositions of A and B for fast product formula evaluation
    evals_a, evecs_a = np.linalg.eigh(A)
    evals_b, evecs_b = np.linalg.eigh(B)
    # Precompute evecs for efficient matrix multiplication
    evecs_a_dag = evecs_a.conj().T
    evecs_b_dag = evecs_b.conj().T

    def fast_product_formula(a, b, t):
        """Fast product formula using precomputed eigendecompositions."""
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            if abs(b[k]) * t > 1e-15:
                phase_b = np.exp(-1j * b[k] * t * evals_b)
                U = (evecs_b * phase_b) @ (evecs_b_dag @ U)
            if abs(a[k]) * t > 1e-15:
                phase_a = np.exp(-1j * a[k] * t * evals_a)
                U = (evecs_a * phase_a) @ (evecs_a_dag @ U)
        return U

    def fast_compute_error(a, b):
        """Fast error computation."""
        times = [0.25, 0.5, 1.0]
        total = 0.0
        for t in times:
            U_pf = fast_product_formula(a, b, t)
            U_exact = _exact_propagator(t)
            diff = U_pf - U_exact
            s = np.linalg.svd(diff, compute_uv=False)
            total += s[0]
        return total / len(times)

    # Initial guess: uniform distribution
    n = stages
    a_init = [1.0 / n] * n
    b_init = [1.0 / n] * n

    # Nelder-Mead optimization
    best_a = a_init[:]
    best_b = b_init[:]
    best_err = fast_compute_error(best_a, best_b)

    # Set up Nelder-Mead simplex
    # Parameters: a and b each have n elements, but sum(a)=1 and sum(b)=1
    # So we have 2*(n-1) free parameters
    # Use a reduced parameterization: a_i for i=0..n-2, a_{n-1} = 1 - sum(a_0..n-2)
    # Same for b

    def params_to_ab(params):
        """Convert reduced parameters to full a, b with sum constraint."""
        a = list(params[:n-1])
        a.append(1.0 - sum(a))
        b = list(params[n-1:2*n-2])
        b.append(1.0 - sum(b))
        return a, b

    def ab_to_params(a, b):
        """Convert full a, b to reduced parameters."""
        return list(a[:n-1]) + list(b[:n-1])

    def objective(params):
        a, b = params_to_ab(params)
        # Check bounds
        for val in a + b:
            if val < -50 or val > 50:
                return 1e10
        return fast_compute_error(a, b)

    # Initial simplex
    x0 = ab_to_params(a_init, b_init)
    simplex = [x0[:]]
    for i in range(2 * (n - 1)):
        xi = x0[:]
        xi[i] += 0.05
        simplex.append(xi)

    # Evaluate initial simplex
    fvals = [objective(v) for v in simplex]
    best_idx = int(np.argmin(fvals))
    best_val = fvals[best_idx]
    best_params = simplex[best_idx][:]

    # Nelder-Mead iterations
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    max_iters = 300
    for iteration in range(max_iters):
        if time.time() > deadline - 10.0:
            break

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i][:] for i in order]
        fvals = [fvals[i] for i in order]
        best_val = fvals[0]
        best_params = simplex[0][:]
        best_a, best_b = params_to_ab(best_params)

        # Centroid of all but worst point
        centroid = np.zeros(2 * (n - 1))
        for i in range(2 * (n - 1)):
            centroid += np.array(simplex[i])
        centroid /= (2 * (n - 1))

        # Reflection
        x_r = centroid + alpha * (centroid - np.array(simplex[-1]))
        f_r = objective(x_r)

        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = objective(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            if f_r < fvals[-1]:
                x_c = centroid + rho * (x_r - centroid)
            else:
                x_c = centroid - rho * (simplex[-1] - centroid)
            f_c = objective(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, 2 * (n - 1) + 1):
                    simplex[i] = simplex[0] + sigma * (np.array(simplex[i]) - np.array(simplex[0]))
                    fvals[i] = objective(simplex[i])

    # Final result
    best_a, best_b = params_to_ab(best_params)

    # Ensure bounds
    for i in range(n):
        if best_a[i] < -50:
            best_a[i] = -50
        if best_a[i] > 50:
            best_a[i] = 50
        if best_b[i] < -50:
            best_b[i] = -50
        if best_b[i] > 50:
            best_b[i] = 50

    # Normalize to ensure sum = 1
    sum_a = sum(best_a)
    sum_b = sum(best_b)
    if abs(sum_a) > 1e-15:
        best_a = [x / sum_a for x in best_a]
    if abs(sum_b) > 1e-15:
        best_b = [x / sum_b for x in best_b]

    # Final check
    final_err = fast_compute_error(best_a, best_b)
    if final_err < best_err:
        best_a = best_a
        best_b = best_b

    return best_a, best_b
