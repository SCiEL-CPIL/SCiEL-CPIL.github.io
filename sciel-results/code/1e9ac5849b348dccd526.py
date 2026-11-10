"""QAOA parameter design: per-graph numerical optimization of two-layer angles."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 140.0  # safety margin below 150s wall-clock kill
_T0 = time.time()


def _check_deadline():
    return (time.time() - _T0) < _DEADLINE


def _build_adj(n, edges):
    A = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n and u != v:
            A[u, v] = 1
            A[v, u] = 1
    return A


def _maxcut_exact(n, edges):
    m = len(edges)
    if m == 0:
        return 0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best


def _qaoa_expectation(n, edges, gamma, beta):
    # Exact two-layer QAOA expected cut for unweighted MaxCut, n <= 10.
    # State-vector simulation with per-qubit gates (efficient reshape).
    if n > 10:
        return _qaoa_expectation_heuristic(n, edges, gamma, beta)

    dim = 1 << n
    # state indexed by bitstring, bit i = qubit i (LSB).
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)

    c, s = np.cos(beta), np.sin(beta)
    # exp(-i beta X) on one qubit: |0> -> c|0> - i s |1>; |1> -> -i s |0> + c|1>
    # Matrix in basis (|0>,|1>): [[c, -i s], [-i s, c]]
    Ux = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    def _apply_single_qubit(state, qubit, U):
        # state: 1-D array length 2^n, bit qubit is LSB-indexed.
        # Reshape to [2]*n with axis order such that qubit qubit is separated.
        # index = sum b_i 2^i. We reshape so that the fastest-varying axis is bit 0.
        shape = [2] * n
        # np.reshape with default C order: last axis varies fastest.
        # We want bit 0 (LSB) to vary fastest -> last axis.
        # So axis j in the reshaped array corresponds to bit (n-1-j).
        # qubit q -> axis (n-1-q).
        ax = n - 1 - qubit
        new_shape = list(shape)
        # move axis ax to front, apply U, move back.
        state_r = state.reshape(shape)
        state_r = np.moveaxis(state_r, ax, 0)
        # now shape[0] is the qubit dimension (2), apply U @ qubit
        state_r = np.tensordot(U, state_r, axes=(1, 0))
        # tensordot(U (2x2), state_r (2, ...)) over axis1 of U and axis0 of state_r
        # result shape (2, ...) with U's axis0 first.
        state_r = np.moveaxis(state_r, 0, ax)
        return state_r.reshape(dim)

    # Precompute edge list for C rotation. C = sum (1 - ZuZv)/2.
    # exp(-i gamma C) is a multi-qubit gate. For small n we build it as a
    # diagonal+off-diagonal? C is not diagonal. We build the full operator.
    # To keep it fast, build C as a sparse-ish dense matrix of size dim x dim.
    def _z_op(qubit):
        # returns dense matrix of Z on qubit (dim x dim)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        ops = [Z if i == qubit else np.eye(2) for i in range(n)]
        # qubit 0 = LSB = last factor in kron (C order, last axis fastest)
        result = ops[0]
        for o in ops[1:]:
            result = np.kron(result, o)
        return result

    I_dim = np.eye(dim, dtype=np.complex128)
    C = np.zeros((dim, dim), dtype=np.complex128)
    for u, v in edges:
        C += (I_dim - _z_op(u) @ _z_op(v)) / 2.0

    try:
        from scipy.linalg import expm
    except Exception:
        expm = None

    for _layer in range(2):
        # apply exp(-i beta sum X): product of single-qubit Ux (they commute).
        for i in range(n):
            state = _apply_single_qubit(state, i, Ux)
        # apply exp(-i gamma C)
        if expm is not None:
            U = expm(-1j * gamma * C)
        else:
            w, V = np.linalg.eig(-1j * gamma * C)
            U = V @ np.diag(np.exp(w)) @ np.linalg.inv(V)
        state = U @ state

    val = np.vdot(state, C @ state).real
    return float(val)


def _qaoa_expectation_heuristic(n, edges, gamma, beta):
    m = len(edges)
    if m == 0:
        return 0.0
    deg = np.zeros(n)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    corr = np.cos(2 * gamma) * np.cos(beta) ** 2
    return (m / 2.0) * (1 - corr)


def _objective(params, n, edges, opt):
    gamma, beta = params
    val = _qaoa_expectation(n, edges, gamma, beta)
    if opt <= 0:
        return 0.0
    return val / opt


def angles(n, edges, p):
    if not _check_deadline():
        return np.tile([0.35, 0.25], (p, 1))

    opt = _maxcut_exact(n, edges)

    best_params = np.array([0.35, 0.25])
    best_val = _qaoa_expectation(n, edges, *best_params)

    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    if minimize is not None and _check_deadline():
        starts = [
            [0.35, 0.25], [0.5, 0.3], [0.2, 0.4], [0.8, 0.2],
            [0.1, 0.1], [1.0, 0.5], [0.6, 0.1], [0.4, 0.6],
        ]
        for s in starts:
            if not _check_deadline():
                break
            try:
                res = minimize(
                    lambda x: -_objective(x, n, edges, opt),
                    np.array(s, dtype=float),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-8,
                             'adaptive': True},
                )
                cand = np.clip(res.x, 0.0, np.pi)
                v = _qaoa_expectation(n, edges, *cand)
                if v > best_val:
                    best_val = v
                    best_params = cand
            except Exception:
                pass

    return np.tile(best_params, (p, 1))
# EVOLVE-BLOCK-END
