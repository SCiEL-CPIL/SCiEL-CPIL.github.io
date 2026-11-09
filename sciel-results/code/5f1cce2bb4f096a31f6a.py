"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 240.0  # Safety margin for 270s kill


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


def synthesize(problem):
    """Fixed CNOT ladder with single-qubit rotations tuned by coordinate descent with random restarts."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Build layout
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    num_u3 = sum(1 for g in layout if g[0] == "u3")
    param_count = 3 * num_u3

    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    best_gates = None
    best_distance = float('inf')

    # Random restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if _remaining_time() < 10.0:
            break
            
        rng = random.Random(4242 + qubits + restart)
        # Random initialization
        current_values = [rng.uniform(-math.pi, math.pi) for _ in range(param_count)]
        current_gates = circuit(current_values)
        current_distance = _distance(target, _build(current_gates, qubits))
        
        if current_distance < best_distance:
            best_distance = current_distance
            best_gates = current_gates

        # Coordinate Descent
        # Adaptive step size
        step = 0.5
        for sweep in range(20):  # More sweeps
            if _remaining_time() < 5.0:
                break
            improved = False
            for i in range(param_count):
                if _remaining_time() < 5.0:
                    break
                # Try a few random perturbations and local steps
                best_val = current_values[i]
                best_dist = current_distance
                
                # Local steps
                for trial in (current_values[i] + step, current_values[i] - step):
                    values = current_values[:]
                    values[i] = trial
                    gates = circuit(values)
                    dist = _distance(target, _build(gates, qubits))
                    if dist < best_dist:
                        best_dist = dist
                        best_val = trial

                # Random jump
                trial = rng.uniform(-math.pi, math.pi)
                values = current_values[:]
                values[i] = trial
                gates = circuit(values)
                dist = _distance(target, _build(gates, qubits))
                if dist < best_dist:
                    best_dist = dist
                    best_val = trial

                if best_val != current_values[i]:
                    current_values[i] = best_val
                    current_distance = best_dist
                    improved = True
                    if current_distance < best_distance:
                        best_distance = current_distance
                        best_gates = circuit(current_values)
            
            if not improved:
                break
        
        # If we found a very good solution, stop early
        if best_distance < 1e-6:
            break

    if best_gates is None:
        # Fallback
        best_gates = circuit([0.0] * param_count)
        
    return best_gates
# EVOLVE-BLOCK-END
