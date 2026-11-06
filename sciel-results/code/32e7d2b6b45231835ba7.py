"""Exact quantum circuit synthesis within published CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 210.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


def _exact_2qubit(target, qubits=2):
    """Try to decompose a 2-qubit unitary exactly using standard gate sequences."""
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    iswap = np.array([[0, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 0]], dtype=complex)
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

    for ref_name, ref_mat, gates in [
        ("cnot", cnot, [_cx(0, 1)]),
        ("iswap", iswap, None),
        ("swap", swap, [_cx(0, 1), _cx(1, 0), _cx(0, 1)]),
    ]:
        if np.allclose(target, ref_mat, atol=1e-10):
            if ref_name == "iswap":
                gates = [
                    _u3(0, 0, 0, math.pi / 2),
                    _u3(1, 0, 0, math.pi / 2),
                    _u3(0, math.pi / 2, 0, 0),
                    _u3(1, math.pi / 2, 0, 0),
                    _cx(0, 1),
                    _cx(1, 0),
                    _u3(0, math.pi / 2, 0, 0),
                    _u3(1, math.pi / 2, 0, 0),
                    _u3(0, 0, 0, -math.pi / 2),
                    _u3(1, 0, 0, -math.pi / 2),
                ]
            return gates
    return None


def _exact_toffoli(target, qubits=3):
    """Check if target is Toffoli gate."""
    toffoli = np.eye(8, dtype=complex)
    toffoli[6, 6] = 0
    toffoli[7, 7] = 0
    toffoli[6, 7] = 1
    toffoli[7, 6] = 1
    if np.allclose(target, toffoli, atol=1e-10):
        gates = [
            _u3(2, math.pi / 2, 0, 0),
            _cx(1, 2),
            _u3(2, 0, 0, -math.pi / 4),
            _cx(0, 2),
            _u3(2, 0, 0, math.pi / 4),
            _cx(1, 2),
            _u3(2, 0, 0, -math.pi / 4),
            _cx(0, 2),
            _u3(2, 0, 0, math.pi / 4),
            _u3(2, math.pi / 2, 0, 0),
        ]
        return gates
    return None


def _make_gates(layout, params):
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(_u3(gate[1], params[k], params[k + 1], params[k + 2]))
            k += 3
        else:
            gates.append(_cx(gate[1], gate[2]))
    return gates


def _optimize_generic(target, qubits, budget):
    """Optimize single-qubit rotations around a CNOT ladder for generic unitaries."""
    dim = 2 ** qubits
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Build layout with initial and final rotations
    layout = []
    # Initial rotations
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    # CNOT layers
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
    # Final rotations
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize with small random values
    import random
    rng = random.Random(4242 + qubits + budget)
    best = [rng.uniform(-0.1, 0.1) for _ in range(n_params)]
    best_distance = _distance(target, _build(circuit(best), qubits))
    
    # Coordinate descent with adaptive steps
    step = 0.5
    for sweep in range(20):
        if _elapsed() > _DEADLINE - 10:
            break
        step = 0.5 / (1 + sweep * 0.3)
        improved = False
        for i in range(n_params):
            if _elapsed() > _DEADLINE - 5:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
        if not improved and sweep > 5:
            break
    
    # Final local refinement
    step = 0.01
    for i in range(n_params):
        if _elapsed() > _DEADLINE - 2:
            break
        for trial in (best[i] + step, best[i] - step):
            values = best[:]
            values[i] = trial
            value = _distance(target, _build(circuit(values), qubits))
            if value < best_distance:
                best, best_distance = values, value
    
    return circuit(best)


def synthesize(problem):
    """Fixed CNOT ladder with single-qubit rotations tuned by coordinate descent."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try exact decompositions first
    if qubits == 2:
        exact = _exact_2qubit(target, qubits)
        if exact is not None:
            return exact
    elif qubits == 3:
        exact = _exact_toffoli(target, qubits)
        if exact is not None:
            return exact
    
    # For generic unitaries, use optimization
    return _optimize_generic(target, qubits, budget)
# EVOLVE-BLOCK-END
