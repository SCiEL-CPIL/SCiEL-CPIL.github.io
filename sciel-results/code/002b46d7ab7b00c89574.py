"""Optimize a fixed 3-layer Ry/CZ circuit to approximate Ising ground states."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_START = time.time()
_MARGIN = 3.0  # seconds of safety margin before the 150 s kill


def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _cz():
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1]], dtype=complex)


def _embed(op, k, n):
    """Embed a 2-qubit op acting on qubits (k, k+1) into n-qubit space."""
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=complex)
    for a in range(dim):
        if (a >> k) & 1:
            continue
        base = a
        for b in range(4):
            qa = (b >> 1) & 1  # qubit k
            qb = b & 1         # qubit k+1
            col = base | (qa << k) | (qb << (k + 1))
            row = base | (b >> 1 << k) | (b & 1 << (k + 1))
            out[row, col] = op[b >> 2, b & 3]
    return out


def _embed1(op, k, n):
    """Embed a 1-qubit op acting on qubit k into n-qubit space."""
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=complex)
    for a in range(dim):
        base = a & ~(1 << k)
        for b in range(2):
            col = base | (b << k)
            for r in range(2):
                row = base | (r << k)
                out[row, col] += op[r, b]
    return out


def _build_operators(n):
    """Return (H, ZZ_list, X_list) in the full Hilbert space."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    ZZ = []
    Xs = []
    for i in range(n):
        Xs.append(_embed1(np.array([[0, 1], [1, 0]], dtype=complex), i, n))
    for i in range(n - 1):
        ZZ.append(_embed(_cz(), i, n))
    return H, ZZ, Xs


def _apply_circuit(state, theta, n, layers, ZZ, Xs):
    """state: complex array; theta: (layers, n). Returns final state."""
    psi = state
    for l in range(layers):
        for i in range(n):
            psi = _embed1(_ry(theta[l, i]), i, n) @ psi
        if l < layers - 1:
            for i in range(n - 1):
                psi = ZZ[i] @ psi
    return psi


def _energy(psi, coupling, field, ZZ, Xs):
    return float(np.real(np.vdot(psi, (coupling * sum(ZZ) + field * sum(Xs)) @ psi)))


def _exact_ground(coupling, field, ZZ, Xs):
    H = coupling * sum(ZZ) + field * sum(Xs)
    w, v = np.linalg.eigh(H)
    return float(w[0])


def _objective(x, n, coupling, field, layers, ZZ, Xs, e0, dim):
    theta = x.reshape(layers, n)
    psi = _apply_circuit(np.zeros(dim, dtype=complex), theta, n, layers, ZZ, Xs)
    psi[0] = 1.0
    psi = _apply_circuit(psi, theta, n, layers, ZZ, Xs)
    return (_energy(psi, coupling, field, ZZ, Xs) - e0) / n


def _initial_theta(n, field, layers):
    """Heuristic start: single-qubit rotation toward the field direction."""
    # Product state with each qubit rotated by ~ arctan(field/coupling) toward |->.
    base = np.arctan2(field, 1.0)
    theta = np.zeros((layers, n))
    theta[0] = base
    theta[1] = 0.0
    theta[2] = -base
    return theta


def angles(n, coupling, field, layers):
    # Build operators and exact ground energy.
    ZZ, Xs = _build_operators(n)[1], _build_operators(n)[2]
    dim = 1 << n
    e0 = _exact_ground(coupling, field, ZZ, Xs)

    best_theta = _initial_theta(n, field, layers)
    best_val = None
    try:
        x0 = best_theta.ravel()
        res = minimize(
            _objective, x0,
            args=(n, coupling, field, layers, ZZ, Xs, e0, dim),
            method="L-BFGS-B",
            options={"maxiter": 2000, "maxfun": 4000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if res.x is not None:
            t = res.x.reshape(layers, n)
            # validate finiteness
            if np.all(np.isfinite(t)):
                best_theta = t
    except Exception:
        pass

    # Deadline guard: ensure we return well before the kill.
    if _remaining() < 0.0:
        pass  # best-effort; return whatever we have

    return best_theta
# EVOLVE-BLOCK-END
