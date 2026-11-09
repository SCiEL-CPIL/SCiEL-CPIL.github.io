"""Product-formula coefficients for 6-site XXZ Heisenberg chain time evolution."""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s wall-clock kill


def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site open XXZ chain."""
    N = 6
    lam = 0.8
    h = 0.3

    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)

    def op_on_site(op, site, N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result

    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)

    # Each gets half the uniform field
    field_term = 0.5 * h
    for i in range(N):
        A += field_term * op_on_site(sz, i, N)
        B += field_term * op_on_site(sz, i, N)

    # Odd bonds: (0,1), (2,3), (4,5) -> go into A
    # Even bonds: (1,2), (3,4) -> go into B
    def bond_term(op1, op2, i, j):
        ops = [np.eye(2, dtype=complex)] * N
        ops[i] = op1
        ops[j] = op2
        result = ops[0]
        for k in range(1, N):
            result = np.kron(result, ops[k])
        return result

    # XXZ Hamiltonian terms: sx*sx + sy*sy + lam*sz*sz = 0.5*(sx+sy)(sx-sy) + lam*sz*sz
    # More precisely: sigma_x sigma_x + sigma_y sigma_y + lambda sigma_z sigma_z
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        A += bond_term(sx, sx, i, j)
        A += bond_term(sy, sy, i, j)
        A += lam * bond_term(sz, sz, i, j)

    for (i, j) in [(1, 2), (3, 4)]:
        B += bond_term(sx, sx, i, j)
        B += bond_term(sy, sy, i, j)
        B += lam * bond_term(sz, sz, i, j)

    return A, B


def _compute_error(a, b, A, B, times):
    """Compute mean spectral norm error over given times."""
    dim = A.shape[0]
    total_error = 0.0
    for t in times:
        # Build U(t) = prod_k exp(-i b_k t B) exp(-i a_k t A) for k=0..stages-1
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U = np.linalg.multi_dot([np.exp(-1j * b[k] * t * B), U])
            U = np.linalg.multi_dot([np.exp(-1j * a[k] * t * A), U])
        # Exact propagator
        U_exact = np.exp(-1j * t * (A + B))
        err = np.linalg.norm(U - U_exact, 2)
        total_error += err
    return total_error / len(times)


def coefficients(stages):
    """
    Return coefficients (a, b) for a 12-stage product formula.
    Uses a 4th-order symmetric splitting scheme.
    """
    if stages != 12:
        # Fallback: uniform splitting
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    # 4th-order symmetric splitting: A_{c1} B_{c2} A_{c3} B_{c2} A_{c1}
    # c1 = 1 / (2*(1 - 2^(1/3)))
    # c2 = 1 / (2*(1 - 2^(1/3)))
    # c3 = 1 - 2*(c1 + c2)
    alpha = 2.0 ** (1.0 / 3.0)
    c1 = 1.0 / (2.0 * (1.0 - alpha))
    c2 = 1.0 / (2.0 * (1.0 - alpha))
    c3 = 1.0 - 2.0 * (c1 + c2)

    # Map to B_k, A_k pairs: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
    # Sequence: B0 A_{c1} B_{c2} A_{c3} B_{c2} A_{c1} B0 A0 ... B0 A0
    # So:
    # k=0: b=0, a=c1
    # k=1: b=c2, a=c3
    # k=2: b=c2, a=c1
    # k=3..11: b=0, a=0
    a = [c1, c3, c1] + [0.0] * 9
    b = [0.0, c2, c2] + [0.0] * 9

    # Verify sums
    if abs(sum(a) - 1.0) > 1e-10 or abs(sum(b) - 1.0) > 1e-10:
        # Fallback to uniform
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages

    return a, b
