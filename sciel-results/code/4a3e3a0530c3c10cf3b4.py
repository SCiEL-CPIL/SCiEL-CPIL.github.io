"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 250.0


def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
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
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _cnot_op(control, target, qubits):
    size = 2 ** qubits
    op = np.eye(size, dtype=complex)
    for i in range(size):
        ctrl_bit = (i >> (qubits - 1 - control)) & 1
        if ctrl_bit == 1:
            j = i ^ (1 << (qubits - 1 - target))
            op[i, i] = 0
            op[i, j] = 1
    return op


def _identify_known_gate(target, qubits):
    size = 2 ** qubits
    if np.allclose(target, np.eye(size), atol=1e-10):
        return []
    if qubits == 2:
        for ctrl in range(2):
            for tgt in range(2):
                if ctrl == tgt:
                    continue
                cnot = _cnot_op(ctrl, tgt, 2)
                overlap = np.trace(target.conj().T @ cnot) / size
                if abs(abs(overlap) - 1.0) < 1e-8:
                    return [["cx", ctrl, tgt]]
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        overlap = np.trace(target.conj().T @ swap) / size
        if abs(abs(overlap) - 1.0) < 1e-8:
            return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        overlap = np.trace(target.conj().T @ iswap) / size
        if abs(abs(overlap) - 1.0) < 1e-8:
            return [
                ["u3", 0, math.pi/2, 0, math.pi/2],
                ["u3", 1, math.pi/2, 0, -math.pi/2],
                ["cx", 0, 1],
                ["cx", 1, 0],
                ["u3", 0, math.pi/2, 0, -math.pi/2],
                ["u3", 1, math.pi/2, 0, math.pi/2]
            ]
    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[7, 7] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        overlap = np.trace(target.conj().T @ toffoli) / 8
        if abs(abs(overlap) - 1.0) < 1e-8:
            return [
                ["cx", 0, 1],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 1, 2],
                ["u3", 2, math.pi/2, 0, -math.pi/2],
                ["cx", 0, 1],
                ["u3", 1, math.pi/2, 0, math.pi/2],
                ["cx", 1, 2],
                ["u3", 2, math.pi/2, 0, -math.pi/2]
            ]
    return None


def _factorize_residual(target, qubits, cnot_budget):
    size = 2 ** qubits
    residual = target.copy()
    gates = []
    cnots_used = 0
    det = np.linalg.det(residual)
    if abs(det) > 1e-10:
        global_phase = np.angle(det) / qubits
        residual = residual * np.exp(-1j * global_phase)
    max_layers = min(cnot_budget, 6)
    for layer in range(max_layers):
        if _remaining_time() < 10.0:
            break
        for q in range(qubits):
            if _remaining_time() < 5.0:
                break
            best_params = [0.0, 0.0, 0.0]
            best_cost = float('inf')
            for theta in np.linspace(0, 2*math.pi, 8):
                for phi in np.linspace(0, 2*math.pi, 4):
                    for lam in np.linspace(0, 2*math.pi, 4):
                        u3 = _u3_matrix(theta, phi, lam)
                        op = np.array([[1]], dtype=complex)
                        eye = np.eye(2, dtype=complex)
                        for k in range(qubits):
                            op = np.kron(op, u3 if k == q else eye)
                        new_residual = op @ residual
                        cost = np.sum(np.abs(new_residual) ** 2) - np.sum(np.abs(np.diag(new_residual)) ** 2)
                        if cost < best_cost:
                            best_cost = cost
                            best_params = [float(theta), float(phi), float(lam)]
            gates.append(["u3", q] + best_params)
            u3 = _u3_matrix(*best_params)
            op = np.array([[1]], dtype=complex)
            eye = np.eye(2, dtype=complex)
            for k in range(qubits):
                op = np.kron(op, u3 if k == q else eye)
            residual = op @ residual
        if cnots_used >= cnot_budget:
            break
        best_cnot = None
        best_cnot_cost = float('inf')
        for ctrl in range(qubits):
            for tgt in range(qubits):
                if ctrl == tgt:
                    continue
                cnot_op = _cnot_op(ctrl, tgt, qubits)
                new_residual = cnot_op @ residual
                cost = np.sum(np.abs(new_residual) ** 2) - np.sum(np.abs(np.diag(new_residual)) ** 2)
                if cost < best_cnot_cost:
                    best_cnot_cost = cost
                    best_cnot = (ctrl, tgt)
        if best_cnot is not None:
            ctrl, tgt = best_cnot
            gates.append(["cx", ctrl, tgt])
            cnot_op = _cnot_op(ctrl, tgt, qubits)
            residual = cnot_op @ residual
            cnots_used += 1
        diag_mean = np.trace(residual) / size
        if np.allclose(residual, np.eye(size) * diag_mean, atol=1e-6):
            break
    for q in range(qubits):
        if _remaining_time() < 3.0:
            break
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    return gates


def synthesize(problem):
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    known = _identify_known_gate(target, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            return known
    gates = _factorize_residual(target, qubits, budget)
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget or len(gates) > 400:
        return []
    return gates
# EVOLVE-BLOCK-END
