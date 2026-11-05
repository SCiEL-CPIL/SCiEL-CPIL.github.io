"""Combined program: structural template from current + refinement from inspiration 1."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 270.0 - 10.0  # safety margin


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _apply_single_to_matrix(matrix, qubit, single, qubits):
    """Apply single-qubit gate to specific qubit position in full unitary."""
    eye = np.eye(2, dtype=complex)
    operator = np.array([[1]], dtype=complex)
    for k in range(qubits):
        operator = np.kron(operator, single if k == qubit else eye)
    return operator @ matrix


def _apply_cnot_to_matrix(matrix, control, target, qubits):
    """Apply CNOT gate to full unitary."""
    index = np.arange(2 ** qubits)
    bit = (index >> (qubits - 1 - control)) & 1
    return matrix[index ^ (bit << (qubits - 1 - target)), :]


def _build_from_layout(layout, values, qubits):
    """Build unitary directly from layout and parameter values (faster than building gate list)."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            theta, phi, lam = values[k], values[k + 1], values[k + 2]
            k += 3
            single = _u3_matrix(theta, phi, lam)
            matrix = _apply_single_to_matrix(matrix, gate[1], single, qubits)
        else:
            matrix = _apply_cnot_to_matrix(matrix, gate[1], gate[2], qubits)
    return matrix


def _gates_from_layout(layout, values):
    """Convert layout + values to gate list."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _try_standard_gates(problem):
    """Check for known exact decompositions of standard gates."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits

    # Identity
    if np.allclose(target, np.eye(n), atol=1e-10):
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    if qubits == 2:
        # CNOT 0->1
        cx_01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        if np.allclose(target, cx_01, atol=1e-10):
            return [["cx", 0, 1]]
        # CNOT 1->0
        cx_10 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, cx_10, atol=1e-10):
            return [["cx", 1, 0]]
        # SWAP
        if budget >= 3:
            swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
            if np.allclose(target, swap, atol=1e-10):
                return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
        # iSWAP: [[0, i, 0, 0], [i, 0, 0, 0], [0, 0, 0, i], [0, 0, i, 0]]
        if budget >= 3:
            iswap = np.array([[0, 1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, 1j], [0, 0, 1j, 0]], dtype=complex)
            if np.allclose(target, iswap, atol=1e-10):
                # iSWAP decomposition using 3 CNOTs
                # Standard: iSWAP = (S⊗S) (H⊗H) CNOT(0,1) (H⊗H) (S†⊗S)
                # H = u3(pi/2, 0, pi)
                # S = u3(0, 0, pi/2)
                # S† = u3(0, 0, -pi/2)
                gates = [
                    ["u3", 0, 0, 0, math.pi / 2],   # S on q0
                    ["u3", 1, 0, 0, math.pi / 2],   # S on q1
                    ["u3", 0, math.pi / 2, 0, math.pi],  # H on q0
                    ["u3", 1, math.pi / 2, 0, math.pi],  # H on q1
                    ["cx", 0, 1],
                    ["u3", 0, math.pi / 2, 0, math.pi],  # H on q0
                    ["u3", 1, math.pi / 2, 0, math.pi],  # H on q1
                    ["u3", 0, 0, 0, -math.pi / 2],  # S† on q0
                    ["u3", 1, 0, 0, math.pi / 2],   # S on q1
                ]
                u = _build(gates, 2)
                if _distance(target, u) < 1e-8:
                    return gates
                # Try alternative: iSWAP with different phase convention
                gates2 = [
                    ["u3", 0, math.pi / 2, 0, 0],   # Rx(pi/2) on q0
                    ["u3", 1, math.pi / 2, 0, 0],   # Rx(pi/2) on q1
                    ["cx", 0, 1],
                    ["u3", 0, -math.pi / 2, 0, 0],  # Rx(-pi/2) on q0
                    ["u3", 1, -math.pi / 2, 0, 0],  # Rx(-pi/2) on q1
                    ["cx", 1, 0],
                    ["u3", 0, math.pi / 2, 0, 0],   # Rx(pi/2) on q0
                    ["u3", 1, math.pi / 2, 0, 0],   # Rx(pi/2) on q1
                    ["cx", 0, 1],
                ]
                u2 = _build(gates2, 2)
                if _distance(target, u2) < 1e-8:
                    return gates2

    if qubits == 3:
        # Toffoli (CCX): controls 0,1; target 2
        if budget >= 6:
            toffoli = np.eye(8, dtype=complex)
            # Flip bit 2 when bits 0 and 1 are set (MSB convention: qubit 0 is MSB)
            # Basis index: q0 q1 q2
            # Toffoli flips q2 when q0=1 and q1=1
            for i in range(8):
                q0 = (i >> 2) & 1
                q1 = (i >> 1) & 1
                q2 = i & 1
                if q0 == 1 and q1 == 1:
                    j = i ^ 1
                    toffoli[i, i] = 0
                    toffoli[j, i] = 1
            if np.allclose(target, toffoli, atol=1e-10):
                # Known 6-CNOT Toffoli decomposition
                # Using the standard decomposition:
                gates = [
                    ["u3", 2, math.pi / 2, 0, math.pi],  # H on q2
                    ["cx", 1, 2],
                    ["u3", 2, 0, 0, -math.pi / 4],       # T† on q2
                    ["cx", 0, 2],
                    ["u3", 2, 0, 0, math.pi / 4],        # T on q2
                    ["cx", 1, 2],
                    ["u3", 2, 0, 0, -math.pi / 4],       # T† on q2
                    ["u3", 1, 0, 0, math.pi / 4],        # T on q1
                    ["cx", 0, 1],
                    ["u3", 1, 0, 0, -math.pi / 4],       # T† on q1
                    ["u3", 2, 0, 0, math.pi / 4],        # T on q2
                    ["cx", 0, 2],
                    ["u3", 2, math.pi / 2, 0, math.pi],  # H on q2
                ]
                u = _build(gates, 3)
                if _distance(target, u) < 1e-8:
                    return gates
                # Try alternative Toffoli decomposition
                gates2 = [
                    ["u3", 2, math.pi / 2, 0, math.pi],  # H on q2
                    ["cx", 1, 2],
                    ["u3", 2, 0, 0, -math.pi / 4],
                    ["cx", 0, 2],
                    ["u3", 2, 0, 0, math.pi / 4],
                    ["cx", 1, 2],
                    ["u3", 2, 0, 0, -math.pi / 4],
                    ["u3", 1, 0, 0, math.pi / 4],
                    ["cx", 0, 1],
                    ["u3", 1, 0, 0, -math.pi / 4],
                    ["u3", 2, 0, 0, math.pi / 4],
                    ["cx", 0, 2],
                    ["u3", 2, math.pi / 2, 0, math.pi],
                ]
                u2 = _build(gates2, 3)
                if _distance(target, u2) < 1e-8:
                    return gates2

    return None


def _make_layout(qubits, cnot_budget, pair_order):
    """Create a circuit layout with u3 gates interleaved with CNOTs."""
    layout = []
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    for i in range(cnot_budget):
        c, t = pair_order[i % len(pair_order)]
        layout.append(["cx", c, t])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    return layout


def _optimize(layout, target, qubits, rng, max_sweeps=15):
    """Optimize u3 parameters using coordinate descent with random restarts."""
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    best = [0.0] * n_params
    best_dist = _distance(target, _build_from_layout(layout, best, qubits))

    for sweep in range(max_sweeps):
        if _time_remaining() < 15.0:
            break
        step = 1.0 / (1.0 + sweep * 0.5)
        improved = False
        for i in range(n_params):
            if _time_remaining() < 15.0:
                break
            # Try structured perturbations
            for trial in (best[i] + step, best[i] - step,
                          best[i] + rng.uniform(-math.pi, math.pi)):
                trial = trial % (2 * math.pi)
                if trial > math.pi:
                    trial -= 2 * math.pi
                values = best[:]
                values[i] = trial
                dist = _distance(target, _build_from_layout(layout, values, qubits))
                if dist < best_dist:
                    best, best_dist = values, dist
                    improved = True
        if not improved:
            break

    return best, best_dist


def synthesize(problem):
    """Synthesize a circuit for the given target unitary within the CNOT budget."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Try known exact decompositions first
    standard = _try_standard_gates(problem)
    if standard is not None:
        return standard

    # Generate candidate layouts
    if qubits == 2:
        pair_options = [
            [(0, 1)],
            [(1, 0)],
            [(0, 1), (1, 0)],
            [(1, 0), (0, 1)],
        ]
    else:
        pair_options = [
            [(0, 1), (0, 2), (1, 2)],
            [(0, 1), (1, 2), (0, 2)],
            [(0, 2), (0, 1), (1, 2)],
            [(1, 2), (0, 1), (0, 2)],
            [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)],
        ]

    best_gates = None
    best_dist = 1.0
    rng = random.Random(4242 + qubits * 100 + budget)

    # Try each pair ordering
    for pair_order in pair_options:
        if _time_remaining() < 20.0:
            break
        if len(pair_order) == 0:
            continue

        layout = _make_layout(qubits, budget, pair_order)

        # Check gate count
        n_gates = len(layout)
        if n_gates > 400:
            continue
        n_cnots = sum(1 for g in layout if g[0] == "cx")
        if n_cnots > budget:
            continue

        # Optimize this layout
        values, dist = _optimize(layout, target, qubits, rng)
        if dist < best_dist:
            best_dist = dist
            best_gates = _gates_from_layout(layout, values)

        # If we've found a very good solution, stop early
        if best_dist < 1e-10:
            break

    # If no valid solution found, return identity
    if best_gates is None:
        best_gates = [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    return best_gates
# EVOLVE-BLOCK-END
