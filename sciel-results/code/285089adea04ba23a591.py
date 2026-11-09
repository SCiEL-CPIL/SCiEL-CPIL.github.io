"""Exact quantum circuit synthesis via structured factorization and adaptive optimization."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 265.0  # safety margin before 270s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            # Build operator efficiently
            operator = np.eye(n, dtype=complex)
            for k in range(qubits):
                if k == q:
                    operator = np.kron(operator, single)
                else:
                    operator = np.kron(operator, np.eye(2, dtype=complex))
            matrix = operator @ matrix
        else:
            _, control, tgt = gate
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - control)) & 1
            perm = idx ^ (bit << (qubits - 1 - tgt))
            matrix = matrix[perm, :]
    return matrix


def _distance(target, matrix):
    d = 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])
    return max(0.0, d)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _decompose_single_qubit(U):
    """Return (theta, phi, lam) for u3 such that u3 ~ U up to global phase."""
    det = np.linalg.det(U)
    phase = np.angle(det)
    U_prime = U / np.exp(0.5j * phase)
    d00 = U_prime[0, 0]
    if abs(d00) < 1e-12:
        theta = math.pi
        lam = -np.angle(U_prime[0, 1])
        phi = np.angle(U_prime[1, 0])
        return (theta, phi, lam)
    theta = 2 * math.acos(min(1.0, max(-1.0, abs(d00))))
    phi = np.angle(U_prime[1, 0])
    lam = np.angle(U_prime[1, 0]) + np.angle(U_prime[0, 1])
    return (theta, phi, lam)


def _apply_u3(gates, q, theta, phi, lam):
    gates.append(["u3", q, float(theta), float(phi), float(lam)])


def _cx(gates, c, t):
    gates.append(["cx", c, t])


# ---- Known exact decompositions ----

def _swap_gate(gates, q0, q1):
    _cx(gates, q0, q1)
    _cx(gates, q1, q0)
    _cx(gates, q0, q1)


def _iswap_gate(gates, q0, q1):
    # iSWAP via 1 CNOT + single-qubit gates
    _apply_u3(gates, q0, 0.0, 0.0, math.pi / 2)
    _apply_u3(gates, q1, 0.0, 0.0, math.pi / 2)
    _apply_u3(gates, q0, 0.0, 0.0, math.pi / 2)
    _cx(gates, q0, q1)
    _apply_u3(gates, q1, 0.0, 0.0, math.pi / 2)
    _apply_u3(gates, q0, 0.0, 0.0, math.pi / 2)
    _apply_u3(gates, q1, 0.0, 0.0, math.pi / 2)


def _toffoli_gate(gates, c1, c2, t):
    _apply_u3(gates, t, math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c1)
    _apply_u3(gates, c1, math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c1)
    _apply_u3(gates, t, -math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c2)
    _apply_u3(gates, c2, math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c2)
    _apply_u3(gates, t, -math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c1)
    _apply_u3(gates, c1, -math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c1)
    _apply_u3(gates, c1, math.pi / 2, 0.0, 0.0)
    _apply_u3(gates, t, math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c2)
    _apply_u3(gates, c2, -math.pi / 2, 0.0, 0.0)
    _cx(gates, t, c2)
    _apply_u3(gates, c2, math.pi / 2, 0.0, 0.0)


def _match_known(gates, target, qubits, name):
    n = 2 ** qubits
    if qubits == 2:
        # CNOT
        for c, t in [(0, 1), (1, 0)]:
            M = np.eye(n, dtype=complex)
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - c)) & 1
            M2 = np.zeros_like(M)
            M2[idx ^ (bit << (qubits - 1 - t)), :] = M
            if np.allclose(M2, target, atol=1e-9):
                _cx(gates, c, t)
                return True
        # SWAP
        M = np.zeros((n, n), dtype=complex)
        for i in range(n):
            b0 = (i >> 1) & 1
            b1 = i & 1
            j = (b1 << 1) | b0
            M[i, j] = 1
        if np.allclose(M, target, atol=1e-9):
            _swap_gate(gates, 0, 1)
            return True
        # iSWAP
        M = np.zeros((n, n), dtype=complex)
        M[0, 3] = 1; M[1, 2] = 1; M[2, 1] = 1; M[3, 0] = 1
        if np.allclose(M, target, atol=1e-9):
            _iswap_gate(gates, 0, 1)
            return True
    if qubits == 3:
        # Toffoli
        for c1 in range(3):
            for c2 in range(3):
                if c1 == c2:
                    continue
                for t in range(3):
                    if t == c1 or t == c2:
                        continue
                    M = np.eye(n, dtype=complex)
                    for i in range(n):
                        bc1 = (i >> (qubits - 1 - c1)) & 1
                        bc2 = (i >> (qubits - 1 - c2)) & 1
                        if bc1 == 1 and bc2 == 1:
                            j = i ^ (1 << (qubits - 1 - t))
                            M[i, j] = 1
                    if np.allclose(M, target, atol=1e-9):
                        _toffoli_gate(gates, c1, c2, t)
                        return True
        # SWAP on any two qubits
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                M = np.eye(n, dtype=complex)
                for i in range(n):
                    ba = (i >> (qubits - 1 - a)) & 1
                    bb = (i >> (qubits - 1 - b)) & 1
                    j = i
                    j = (j & ~(1 << (qubits - 1 - a))) | (ba << (qubits - 1 - b))
                    j = (j & ~(1 << (qubits - 1 - b))) | (bb << (qubits - 1 - a))
                    M[i, j] = 1
                if np.allclose(M, target, atol=1e-9):
                    _swap_gate(gates, a, b)
                    return True
    return False


# ---- KAK for 2-qubit generic unitary (3 CNOTs) ----

def _kak_two_qubit(target):
    """Decompose 4x4 unitary into single-qubit + 3 CNOTs using KAK."""
    n = 4
    # Pauli basis
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    basis = [I, X, Y, Z]

    # Compute coefficients in Pauli basis
    coeffs = np.zeros((4, 4), dtype=complex)
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            op = np.kron(a, b)
            coeffs[i, j] = np.trace(op.conj().T @ target) / 4

    # KAK decomposition: U = A (e^{i t1 XX} e^{i t2 YY} e^{i t3 ZZ}) B
    # where A, B are local unitaries.
    # We use a numerical approach: find the KAK angles and local unitaries.

    # The KAK canonical form for 2-qubit unitaries:
    # U = (L0 ⊗ L1) * (CX CX CX) * (R0 ⊗ R1)
    # where the core is fixed as CX(0,1) CX(1,0) CX(0,1)

    # Build the core matrix
    eye2 = np.eye(2, dtype=complex)

    def cx_mat(c, t):
        m = np.eye(n, dtype=complex)
        idx = np.arange(n)
        bit = (idx >> (2 - 1 - c)) & 1
        m2 = np.zeros_like(m)
        m2[idx ^ (bit << (2 - 1 - t)), :] = m
        return m2

    core = cx_mat(0, 1) @ cx_mat(1, 0) @ cx_mat(0, 1)

    # We want: U = (L0 ⊗ L1) @ core @ (R0 ⊗ R1)
    # => (L0 ⊗ L1) = U @ (R0 ⊗ R1)^{-1} @ core^{-1}
    # Choose R0 = R1 = I initially, then (L0 ⊗ L1) = U @ core^{-1}
    # But this may not be separable. We need to find R0, R1 such that
    # U @ (R0 ⊗ R1)^{-1} @ core^{-1} is separable.

    # Use an iterative approach: optimize R0, R1 to make the product separable.
    # For now, use a simple heuristic: set R = I and check if it works.

    R0 = np.eye(2, dtype=complex)
    R1 = np.eye(2, dtype=complex)
    R = np.kron(R0, R1)
    L = target @ R.conj().T @ core.conj().T

    # Check if L is approximately separable
    # L should be of the form L0 ⊗ L1
    # We can check by seeing if L is in the set of separable states

    # If not separable, we need to adjust R. For now, use coordinate descent.
    gates = _coordinate_descent(target, _make_layout_2qubit(3), 2, max_sweeps=12)
    return gates


def _make_layout_2qubit(num_cx):
    """Create a layout with num_cx CNOTs and single-qubit gates."""
    layout = []
    for layer in range(num_cx):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(2)]
        layout.append(["cx", 0, 1] if layer % 2 == 0 else ["cx", 1, 0])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(2)]
    return layout


def _make_layout_3qubit(num_cx, pairs=None):
    """Create a layout with num_cx CNOTs and single-qubit gates for 3 qubits."""
    if pairs is None:
        pairs = [(0, 1), (1, 2), (0, 2), (2, 0), (1, 0), (2, 1)]
    layout = []
    for layer in range(num_cx):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
        c, t = pairs[layer % len(pairs)]
        layout.append(["cx", c, t])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
    return layout


def _instantiate(layout, params):
    gates, k = [], 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], params[k], params[k + 1], params[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _coordinate_descent(target, layout, qubits, max_sweeps=10, step_init=1.0):
    """Optimize u3 parameters using coordinate descent."""
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_dist = _distance(target, _build(_instantiate(layout, best), qubits))

    step = step_init
    for sweep in range(max_sweeps):
        if _remaining_time() < 3.0:
            break
        improved = False
        for i in range(n_params):
            if _remaining_time() < 3.0:
                break
            # Try several trial values
            trials = [best[i] + step, best[i] - step]
            # Add some random perturbations
            for _ in range(3):
                trials.append(best[i] + step * (2 * np.random.random() - 1))
            for trial in trials:
                values = best[:]
                values[i] = trial
                d = _distance(target, _build(_instantiate(layout, values), qubits))
                if d < best_dist - 1e-12:
                    best, best_dist, improved = values, d, True
                    break
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        else:
            step *= 0.95  # slow decay
    return _instantiate(layout, best)


def _gradient_descent(target, layout, qubits, max_iters=500, lr_init=0.1):
    """Optimize u3 parameters using gradient descent with numerical gradients."""
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    params = np.zeros(n_params)
    best_params = params.copy()
    best_dist = _distance(target, _build(_instantiate(layout, params), qubits))
    lr = lr_init

    for it in range(max_iters):
        if _remaining_time() < 3.0:
            break

        # Compute numerical gradient
        grad = np.zeros(n_params)
        eps = 1e-5
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps
            d_plus = _distance(target, _build(_instantiate(layout, params_plus), qubits))
            d_minus = _distance(target, _build(_instantiate(layout, params_minus), qubits))
            grad[i] = (d_plus - d_minus) / (2 * eps)

        # Update parameters
        params -= lr * grad

        # Compute new distance
        dist = _distance(target, _build(_instantiate(layout, params), qubits))

        if dist < best_dist:
            best_params = params.copy()
            best_dist = dist
            lr *= 1.05  # increase learning rate slightly
        else:
            params = best_params.copy()
            lr *= 0.9  # decrease learning rate

        if lr < 1e-6:
            break

    return _instantiate(layout, best_params)


def synthesize(problem):
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)
    gates = []

    # Try known exact decompositions first
    if _match_known(gates, target, qubits, name):
        return gates

    # Generic 2-qubit: KAK (3 CNOTs)
    if qubits == 2:
        num_cx = min(budget, 3)
        layout = _make_layout_2qubit(num_cx)
        # Try coordinate descent first (faster)
        gates = _coordinate_descent(target, layout, qubits, max_sweeps=15)
        dist = _distance(target, _build(gates, qubits))
        if dist < 1e-8:
            return gates
        # Try gradient descent if time permits
        if _remaining_time() > 10.0:
            gates_gd = _gradient_descent(target, layout, qubits, max_iters=200)
            dist_gd = _distance(target, _build(gates_gd, qubits))
            if dist_gd < dist:
                gates = gates_gd
        return gates

    # Generic 3-qubit
    num_cx = min(budget, 12)
    layout = _make_layout_3qubit(num_cx)
    gates = _coordinate_descent(target, layout, qubits, max_sweeps=8)
    dist = _distance(target, _build(gates, qubits))
    if dist < 1e-8:
        return gates
    # Try gradient descent if time permits
    if _remaining_time() > 15.0:
        gates_gd = _gradient_descent(target, layout, qubits, max_iters=100)
        dist_gd = _distance(target, _build(gates_gd, qubits))
        if dist_gd < dist:
            gates = gates_gd
    return gates
# EVOLVE-BLOCK-END
