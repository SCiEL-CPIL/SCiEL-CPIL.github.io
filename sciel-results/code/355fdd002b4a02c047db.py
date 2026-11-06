"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain."""
import time
import numpy as np
from scipy.linalg import expm, eigh

# ---------------------------------------------------------------------------
# Build the Hamiltonians A (odd bonds + half field) and B (even bonds + half field)
# for a 6-site open XXZ chain with anisotropy Δ=0.8 and field h=0.3.
# ---------------------------------------------------------------------------

def _build_hamiltonians():
    """Return A, B, H = A+B as 64x64 matrices."""
    N = 6
    dim = 2 ** N
    Δ = 0.8
    h = 0.3

    # Single-site operators
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    # Build A: odd bonds (0-1, 2-3, 4-5) + half uniform field
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    for site in range(N):
        ops = [I2] * N
        ops[site] = sz
        A += (h / 2.0) * kron_all(ops)
        B += (h / 2.0) * kron_all(ops)

    for bond in range(N - 1):
        i, j = bond, bond + 1
        ops_sx = [I2] * N
        ops_sx[i] = sx
        ops_sx[j] = sx
        ops_sy = [I2] * N
        ops_sy[i] = sy
        ops_sy[j] = sy
        ops_sz = [I2] * N
        ops_sz[i] = sz
        ops_sz[j] = sz

        # XXZ interaction: sx_i sx_j + sy_i sy_j + Δ sz_i sz_j
        # = 0.5(sx_i sx_j + sy_i sy_j + Δ sz_i sz_j) * 2? 
        # Standard XXZ: J (sx_i sx_j + sy_i sy_j + Δ sz_i sz_j) with J=1
        # sx sx + sy sy = 0.5(s+ s- + s- s+)
        term = kron_all(ops_sx) @ kron_all(ops_sx)  # wrong, need element-wise
        # Actually: sx_i sx_j means tensor product where site i has sx, site j has sx
        sxij = kron_all(ops_sx)
        syij = kron_all(ops_sy)
        szij = kron_all(ops_sz)

        interaction = sxij @ sxij  # NO - sxij is already the full operator
        # Wait: sxij = sx_i ⊗ sx_j (with identities elsewhere), so sxij is already sx_i sx_j
        interaction = sxij + syij + Δ * szij

        if bond % 2 == 0:  # even bonds: 0-1, 2-3, 4-5
            B += interaction
        else:  # odd bonds: 1-2, 3-4
            A += interaction

    H = A + B
    return A, B, H


def _precompute():
    """Precompute eigendecompositions for fast expm evaluation."""
    A, B, H = _build_hamiltonians()
    # Eigendecompose A, B, H (they are Hermitian)
    evals_A, evecs_A = eigh(A)
    evals_B, evecs_B = eigh(B)
    evals_H, evecs_H = eigh(H)
    return A, B, H, evals_A, evecs_A, evals_B, evecs_B, evals_H, evecs_H


def _expm_hermitian(evals, evecs, t):
    """Compute exp(-i * t * M) given eigendecomposition of Hermitian M."""
    phases = np.exp(-1j * t * evals)
    return evecs @ np.diag(phases) @ evecs.conj().T


def _compute_error(a, b, A, B, H, evals_A, evecs_A, evals_B, evecs_B, evals_H, evecs_H):
    """Compute mean spectral-norm error over t = 0.25, 0.5, 1.0."""
    times = [0.25, 0.5, 1.0]
    errors = []
    for t in times:
        # Build product formula U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Order k = 1..12 (i.e., index 0..11 in our lists)
        U = np.eye(64, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B)
            if abs(b[k]) > 1e-15:
                U = _expm_hermitian(evals_B, evecs_B, b[k] * t) @ U
            # exp(-i a_k t A)
            if abs(a[k]) > 1e-15:
                U = _expm_hermitian(evals_A, evecs_A, a[k] * t) @ U

        # Exact propagator
        U_exact = _expm_hermitian(evals_H, evecs_H, t)

        # Spectral norm of difference
        diff = U - U_exact
        # Spectral norm = largest singular value
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0])

    return np.mean(errors)


def coefficients(stages):
    """Optimize 12-stage product formula coefficients.
    
    Returns (a, b): two lists of 12 finite numbers in [-50, 50] with sum(a)=sum(b)=1.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute
    A, B, H, evals_A, evecs_A, evals_B, evecs_B, evals_H, evecs_H = _precompute()

    # Initial guess: uniform Strang-like
    a0 = [1.0 / 12.0] * 12
    b0 = [1.0 / 12.0] * 12

    best_a = a0[:]
    best_b = b0[:]
    best_err = _compute_error(a0, b0, A, B, H, evals_A, evecs_A, evals_B, evecs_B, evals_H, evecs_H)

    # Nelder-Mead optimization
    from scipy.optimize import minimize

    def objective(x):
        a = list(x[:12])
        b = list(x[12:24])
        # Enforce bounds
        a = [max(-50.0, min(50.0, v)) for v in a]
        b = [max(-50.0, min(50.0, v)) for v in b]
        # Enforce sum constraints by rescaling
        sa = sum(a)
        sb = sum(b)
        if abs(sa) > 1e-12:
            a = [v / sa for v in a]
        else:
            a = [1.0/12.0]*12
        if abs(sb) > 1e-12:
            b = [v / sb for v in b]
        else:
            b = [1.0/12.0]*12
        return _compute_error(a, b, A, B, H, evals_A, evecs_A, evals_B, evecs_B, evals_H, evecs_H)

    # Multi-start Nelder-Mead
    np.random.seed(42)
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline:
            break

        if i == 0:
            x0 = np.array(a0 + b0)
        else:
            # Perturbed starting point
            x0 = np.array(a0 + b0)
            noise = np.random.normal(0, 0.05, size=24)
            x0 = x0 + noise

        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_err:
                x_opt = res.x
                a_opt = [max(-50.0, min(50.0, v)) for v in x_opt[:12]]
                b_opt = [max(-50.0, min(50.0, v)) for v in x_opt[12:24]]
                sa = sum(a_opt)
                sb = sum(b_opt)
                if abs(sa) > 1e-12:
                    a_opt = [v / sa for v in a_opt]
                if abs(sb) > 1e-12:
                    b_opt = [v / sb for v in b_opt]
                best_a = a_opt
                best_b = b_opt
                best_err = res.fun
        except Exception:
            pass

    return best_a, best_b
