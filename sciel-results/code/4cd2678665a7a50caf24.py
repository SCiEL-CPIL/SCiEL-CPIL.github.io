"""Quantum circuit synthesis: exact decompositions for named gates, optimization for generics."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np


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
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _known_gates_2qubit(problem):
    """Return exact gate list for known 2-qubit gates, or None."""
    name = problem.get("name", "")
    target = _target(problem)
    qubits = 2
    
    # CNOT: control=0, target=1
    if "cnot" in name.lower() or "cx" in name.lower():
        # Verify it's CNOT
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=complex)
        if np.allclose(target, expected, atol=1e-10):
            return [["cx", 0, 1]]
    
    # iSWAP
    if "iswap" in name.lower() or "isw" in name.lower():
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1j, 0],
                             [0, 1j, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, expected, atol=1e-10):
            # iSWAP = (H⊗H) · CNOT · (Ry(π/4)⊗Ry(π/4)) · CNOT · (H⊗H)
            # Let's use a known decomposition
            gates = []
            # H on both
            gates.append(["u3", 0, math.pi, 0, math.pi])
            gates.append(["u3", 1, math.pi, 0, math.pi])
            # CNOT 0->1
            gates.append(["cx", 0, 1])
            # Ry(π/4) on both: u3(π/2, 0, π/2) gives Ry(π/2)? No.
            # Ry(θ) = u3(θ, 0, π/2)
            # Ry(π/4) = u3(π/4, 0, π/2)
            gates.append(["u3", 0, math.pi / 4, 0, math.pi / 2])
            gates.append(["u3", 1, math.pi / 4, 0, math.pi / 2])
            # CNOT 0->1
            gates.append(["cx", 0, 1])
            # H on both
            gates.append(["u3", 0, math.pi, 0, math.pi])
            gates.append(["u3", 1, math.pi, 0, math.pi])
            return gates
    
    # SWAP
    if "swap" in name.lower() and "iswap" not in name.lower():
        expected = np.array([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, expected, atol=1e-10):
            gates = []
            gates.append(["cx", 0, 1])
            gates.append(["cx", 1, 0])
            gates.append(["cx", 0, 1])
            return gates
    
    return None


def _known_gates_3qubit(problem):
    """Return exact gate list for known 3-qubit gates, or None."""
    name = problem.get("name", "")
    target = _target(problem)
    qubits = 3
    
    # Toffoli: control=0,1 target=2
    if "toffoli" in name.lower() or "ccx" in name.lower() or "t" in name.lower():
        # Check if it's Toffoli
        expected = np.eye(8, dtype=complex)
        # Toffoli flips target when both controls are 1: |110> -> |111>, |111> -> |110>
        expected[6, 6] = 0
        expected[6, 7] = 1
        expected[7, 6] = 1
        expected[7, 7] = 0
        if np.allclose(target, expected, atol=1e-10):
            # Standard Toffoli decomposition with 6 CNOTs
            gates = []
            # H on target (qubit 2)
            gates.append(["u3", 2, math.pi, 0, math.pi])
            # T† on target
            gates.append(["u3", 2, math.pi / 4, 0, 0])
            # CNOT 1->2
            gates.append(["cx", 1, 2])
            # T on target
            gates.append(["u3", 2, -math.pi / 4, 0, 0])
            # CNOT 0->2
            gates.append(["cx", 0, 2])
            # T† on target
            gates.append(["u3", 2, math.pi / 4, 0, 0])
            # CNOT 1->2
            gates.append(["cx", 1, 2])
            # T on target
            gates.append(["u3", 2, -math.pi / 4, 0, 0])
            # CNOT 0->2
            gates.append(["cx", 0, 2])
            # T† on control 1
            gates.append(["u3", 1, math.pi / 4, 0, 0])
            # T† on control 0
            gates.append(["u3", 0, math.pi / 4, 0, 0])
            # CNOT 0->1
            gates.append(["cx", 0, 1])
            # T on control 1
            gates.append(["u3", 1, -math.pi / 4, 0, 0])
            # T on control 0
            gates.append(["u3", 0, -math.pi / 4, 0, 0])
            # H on target
            gates.append(["u3", 2, math.pi, 0, math.pi])
            return gates
    
    return None


def _decompose_2qubit(target, qubits, budget):
    """Decompose a 2-qubit unitary into at most budget CNOTs + single-qubit gates."""
    # For 2 qubits, any unitary can be decomposed with at most 3 CNOTs
    # Use the standard KAK decomposition approach
    # But for simplicity, let's use a direct optimization approach
    
    # First, try to see if it can be done with fewer CNOTs
    # Try 0 CNOTs: both qubits independent
    # Try 1 CNOT: controlled rotation
    # Try 2 CNOTs
    # Try 3 CNOTs
    
    # For now, use a fixed layout with 3 CNOTs and optimize single-qubit parameters
    # Layout: U3(0) U3(1) CX(0,1) U3(0) U3(1) CX(1,0) U3(0) U3(1) CX(0,1) U3(0) U3(1)
    
    if budget < 3:
        # Use fewer CNOTs
        if budget == 0:
            layout = []
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
            layout.append(["u3", 0, 0.0, 0.0, 0.0])
            layout.append(["u3", 1, 0.0, 0.0, 0.0])
        elif budget == 1:
            layout = []
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
            layout.append(["cx", 0, 1])
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
        else:  # budget == 2
            layout = []
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
            layout.append(["cx", 0, 1])
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
            layout.append(["cx", 1, 0])
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
    else:
        layout = []
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", 0, 1])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", 1, 0])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", 0, 1])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    params = [0.0] * (3 * num_u3)
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Optimize using coordinate descent
    best = params[:]
    best_distance = _distance(target, _build_fast(circuit(best), qubits))
    
    rng = random.Random(4242)
    start_time = time.time()
    deadline = 200  # Leave margin
    
    sweep = 0
    while time.time() < deadline:
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(len(best)):
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
            if time.time() > deadline:
                break
        if not improved:
            break
        sweep += 1
    
    return circuit(best)


def _decompose_3qubit(target, qubits, budget):
    """Decompose a 3-qubit unitary into at most budget CNOTs + single-qubit gates."""
    # For 3 qubits, the minimum CNOT count for a generic unitary is 15
    # But we may have a larger budget. Use a fixed layout.
    
    # Standard layout: layers of CNOTs with single-qubit gates between
    # Use a repeating pattern of CNOTs
    
    pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    
    layout = []
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    for i in range(budget):
        c, t = pairs[i % len(pairs)]
        layout.append(["cx", c, t])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    params = [0.0] * (3 * num_u3)
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Optimize using coordinate descent
    best = params[:]
    best_distance = _distance(target, _build_fast(circuit(best), qubits))
    
    rng = random.Random(4243)
    start_time = time.time()
    deadline = 200  # Leave margin
    
    sweep = 0
    while time.time() < deadline:
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(len(best)):
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
            if time.time() > deadline:
                break
        if not improved:
            break
        sweep += 1
    
    return circuit(best)


def synthesize(problem):
    """Synthesize a quantum circuit for the given problem."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    if qubits == 2:
        known = _known_gates_2qubit(problem)
        if known is not None:
            return known
        return _decompose_2qubit(target, qubits, budget)
    elif qubits == 3:
        known = _known_gates_3qubit(problem)
        if known is not None:
            return known
        return _decompose_3qubit(target, qubits, budget)
    else:
        # Fallback: return identity
        return []
# EVOLVE-BLOCK-END
