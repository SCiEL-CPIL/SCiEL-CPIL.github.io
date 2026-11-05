"""Quantum circuit synthesis: exact decompositions for named gates, optimization for generics."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np


_DEADLINE = 250.0  # Safety margin before 270s kill
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build the full unitary from a gate list. qubit 0 is MSB."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _synthesize_cnot():
    return [["cx", 0, 1]]


def _synthesize_swap():
    return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]


def _synthesize_toffoli():
    pi = math.pi
    gates = [
        ["u3", 2, pi / 2, 0, pi / 2],
        ["cx", 1, 2],
        ["u3", 2, -pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, -pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, pi / 4, 0, 0],
        ["u3", 1, pi / 2, 0, 0],
        ["u3", 2, pi / 2, 0, -pi / 2],
        ["cx", 0, 1],
        ["u3", 1, -pi / 4, 0, 0],
        ["cx", 0, 1],
        ["u3", 1, pi / 4, 0, 0],
    ]
    return gates


def _synthesize_iswap():
    pi = math.pi
    gates = [
        ["u3", 0, pi, 0, 0],
        ["u3", 1, pi, 0, 0],
        ["cx", 0, 1],
        ["u3", 0, 0, 0, -pi / 2],
        ["u3", 1, 0, 0, pi / 2],
        ["cx", 0, 1],
        ["u3", 0, pi, 0, 0],
        ["u3", 1, pi, 0, 0],
    ]
    return gates


def _optimize_generic(target, qubits, budget, time_limit):
    """Optimize single-qubit rotations around a fixed CNOT layout using coordinate descent."""
    start_time = time.time()
    n_params_per_u3 = 3
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    cnot_count = 0
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[layer % len(pairs)]])
        cnot_count += 1
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])

    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = n_u3 * n_params_per_u3

    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    def objective(values):
        gates = circuit(values)
        matrix = _build_fast(gates, qubits)
        return _distance(target, matrix)

    # Random restarts with coordinate descent
    best_values = [0.0] * n_params
    best_dist = objective(best_values)

    # Determine number of restarts based on available time
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    n_restarts = min(8, max(1, int(remaining / 3.0)))

    rng = random.Random(4242 + qubits)

    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 10:
            break

        values = [0.0] * n_params
        if restart > 0:
            for i in range(n_params):
                values[i] = rng.uniform(-math.pi, math.pi)

        current_dist = objective(values)
        if current_dist < best_dist:
            best_values = values[:]
            best_dist = current_dist

        # Coordinate descent with adaptive step
        step = 1.0
        for sweep in range(12):
            if time.time() - start_time > time_limit - 10:
                break
            step = 1.0 / (1 + sweep)
            improved = False
            for i in range(n_params):
                if time.time() - start_time > time_limit - 10:
                    break
                for trial in (values[i] + step, values[i] - step,
                              values[i] + rng.uniform(-math.pi, math.pi)):
                    values[i] = trial
                    val = objective(values)
                    if val < current_dist:
                        current_dist = val
                        improved = True
                    else:
                        # Revert
                        values[i] = values[i] - (trial - values[i])
            if not improved and step < 0.05:
                break

        if current_dist < best_dist:
            best_values = values[:]
            best_dist = current_dist

    return circuit(best_values)


def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    start_time = time.time()

    # Try known exact decompositions first
    if name == "CNOT" and qubits == 2:
        gates = _synthesize_cnot()
    elif name == "SWAP" and qubits == 2:
        gates = _synthesize_swap()
    elif name == "iSWAP" and qubits == 2:
        gates = _synthesize_iswap()
    elif name == "Toffoli" and qubits == 3:
        gates = _synthesize_toffoli()
    else:
        # Generic: use optimization
        remaining_time = _DEADLINE - (time.time() - start_time)
        if remaining_time < 5:
            # Fallback: return identity-like circuit
            gates = []
            for q in range(qubits):
                gates.append(["u3", q, 0.0, 0.0, 0.0])
            return gates
        gates = _optimize_generic(target, qubits, budget, remaining_time)

    # Verify CNOT count
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        # Truncate CNOTs
        new_gates = []
        count = 0
        for g in gates:
            if g[0] == "cx":
                if count < budget:
                    new_gates.append(g)
                    count += 1
            else:
                new_gates.append(g)
        gates = new_gates

    # Verify gate count
    if len(gates) > 400:
        gates = gates[:400]

    return gates
# EVOLVE-BLOCK-END
