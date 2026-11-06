"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

import numpy as np


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


def _gate_matrix(gate, qubits):
    """Return the full 2^qubits x 2^qubits matrix for a single gate."""
    eye = np.eye(2, dtype=complex)
    if gate[0] == "u3":
        _, q, theta, phi, lam = gate
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
        operator = np.array([[1]], dtype=complex)
        for k in range(qubits):
            operator = np.kron(operator, single if k == q else eye)
        return operator
    else:
        _, control, target = gate
        n = 2 ** qubits
        mat = np.eye(n, dtype=complex)
        # CNOT: for each basis state, if control bit is 1, flip target bit
        for i in range(n):
            bits = [(i >> (qubits - 1 - k)) & 1 for k in range(qubits)]
            if bits[control]:
                bits[target] = 1 - bits[target]
            j = 0
            for k in range(qubits):
                j = (j << 1) | bits[k]
            mat[j, i] = 1.0
        return mat


def _apply_gates(gates, qubits, initial=None):
    """Apply gates to initial matrix (default identity)."""
    if initial is None:
        matrix = np.eye(2 ** qubits, dtype=complex)
    else:
        matrix = initial.copy()
    for gate in gates:
        gmat = _gate_matrix(gate, qubits)
        matrix = gmat @ matrix
    return matrix


def synthesize(problem):
    """
    Fixed CNOT ladder with single-qubit rotations tuned by coordinate descent.
    Uses a different approach: start with known good decompositions for standard gates,
    and use iterative refinement with random restarts and gradient-free optimization.
    """
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Define standard gate layouts based on name
    # For known gates, we can use known optimal decompositions
    # CNOT: 1 CNOT, no single-qubit gates needed (it's already a CNOT)
    # iSWAP: 3 CNOTs
    # SWAP: 3 CNOTs
    # Toffoli: 6 CNOTs
    
    # Build a fixed CNOT layout that stays within budget
    # Use a standard ladder pattern
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits + len(name))
    
    # Determine number of CNOTs to use (up to budget)
    n_cnots = min(budget, 6)  # Use at most 6 CNOTs for efficiency
    
    # Build layout: single-qubit gates before and between CNOTs
    layout = []
    for layer in range(n_cnots):
        # Single-qubit gates on all qubits
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        # One CNOT
        layout.append(["cx", *pairs[layer % len(pairs)]])
    # Final single-qubit gates
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Count parameters
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
    
    # Initialize with zeros
    best = [0.0] * n_params
    best_distance = _distance(target, _build(circuit(best), qubits))
    
    # Coordinate descent with multiple sweeps
    for sweep in range(15):
        step = 0.5 / (1 + sweep * 0.5)
        improved = False
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
            # Also try a random perturbation
            trial = best[i] + rng.uniform(-math.pi, math.pi) * step
            values = best[:]
            values[i] = trial
            value = _distance(target, _build(circuit(values), qubits))
            if value < best_distance:
                best, best_distance = values, value
                improved = True
        if not improved:
            break
    
    # If distance is still large, try random restarts
    if best_distance > 1e-6:
        for restart in range(5):
            candidate = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            dist = _distance(target, _build(circuit(candidate), qubits))
            if dist < best_distance:
                best, best_distance = candidate, dist
                # Refine
                for sweep in range(10):
                    step = 0.3 / (1 + sweep * 0.3)
                    for i in range(n_params):
                        for trial in (best[i] + step, best[i] - step):
                            values = best[:]
                            values[i] = trial
                            value = _distance(target, _build(circuit(values), qubits))
                            if value < best_distance:
                                best, best_distance = values, value
    
    return circuit(best)
# EVOLVE-BLOCK-END
