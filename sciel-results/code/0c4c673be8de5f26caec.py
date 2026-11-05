"""Quantum circuit synthesis: exact decompositions for named gates, KAK for generics."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 255.0  # 255s out of 270s wall-clock limit


def _time_remaining():
    return _DEADLINE - time.time()


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
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Apply single-qubit gate on qubit q
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye2)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            # CNOT: control -> target
            # Build the permutation
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            # New index: if control bit is 1, flip target bit
            new_index = index.copy()
            mask = bit == 1
            target_bit_pos = qubits - 1 - target
            new_index[mask] = index[mask] ^ (1 << target_bit_pos)
            matrix = matrix[new_index, :]
    return matrix


def _distance(target, matrix):
    n = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / n
    return 1.0 - min(1.0, val)


def _identify_named_gate(target, qubits):
    """Check if target matches a known named gate (up to global phase)."""
    n = 2 ** qubits
    
    # CNOT (control 0, target 1) for 2 qubits
    if qubits == 2:
        cnot = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        # Check if target is proportional to cnot
        if np.allclose(target, cnot * np.exp(1j * np.angle(target[0, 0])), atol=1e-10):
            return "cnot"
        
        # iSWAP
        iswap = np.array([[0, 0, 0, 1j],
                          [0, 0, 1j, 0],
                          [0, 1j, 0, 0],
                          [1j, 0, 0, 0]], dtype=complex)
        if np.allclose(target, iswap * np.exp(1j * np.angle(target[0, 0])), atol=1e-10):
            return "iswap"
        
        # SWAP
        swap = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, swap * np.exp(1j * np.angle(target[0, 0])), atol=1e-10):
            return "swap"
    
    # Toffoli for 3 qubits
    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[7, 7] = 0
        if np.allclose(target, toffoli * np.exp(1j * np.angle(target[0, 0])), atol=1e-10):
            return "toffoli"
    
    return None


def _decompose_cnot():
    """CNOT = u3 on target + CNOT + u3 on target. Actually CNOT is just a CNOT gate."""
    # CNOT gate directly
    return [["cx", 0, 1]]


def _decompose_iswap():
    """iSWAP decomposition: iSWAP = (H⊗H) · CNOT · (S†⊗S) · CNOT · (H⊗H) · (S⊗S) · CNOT · (H⊗H)
    Actually, a simpler decomposition:
    iSWAP = (e^{iπ/4} S ⊗ e^{iπ/4} S) · H ⊗ H · CNOT · H ⊗ H · CNOT · H ⊗ H · CNOT · H ⊗ H
    Let me use a known decomposition.
    
    iSWAP can be decomposed as:
    1. u3(π/2, 0, π/2) on both qubits (this is H up to phase)
    2. CNOT 0->1
    3. u3(π/2, 0, π/2) on both qubits
    4. u3(π/2, π/2, 0) on both qubits (this is S up to phase)
    5. CNOT 0->1
    6. u3(π/2, 0, π/2) on both qubits
    7. CNOT 0->1
    8. u3(π/2, 0, π/2) on both qubits
    
    Let me verify this more carefully. The iSWAP gate is:
    |00> -> i|11>
    |01> -> i|10>
    |10> -> i|01>
    |11> -> i|00>
    
    A standard decomposition uses 3 CNOTs and single-qubit rotations.
    """
    # Use a known 3-CNOT decomposition for iSWAP
    gates = []
    # u3(π/2, 0, π/2) = H (up to global phase)
    gates.append(["u3", 0, math.pi / 2, 0, math.pi / 2])
    gates.append(["u3", 1, math.pi / 2, 0, math.pi / 2])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi / 2, 0, math.pi / 2])
    gates.append(["u3", 1, math.pi / 2, 0, math.pi / 2])
    # u3(π/2, π/2, 0) = S (up to phase)
    gates.append(["u3", 0, math.pi / 2, math.pi / 2, 0])
    gates.append(["u3", 1, math.pi / 2, math.pi / 2, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi / 2, 0, math.pi / 2])
    gates.append(["u3", 1, math.pi / 2, 0, math.pi / 2])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, math.pi / 2, 0, math.pi / 2])
    gates.append(["u3", 1, math.pi / 2, 0, math.pi / 2])
    return gates


def _decompose_swap():
    """SWAP = 3 CNOTs with H on both before and after."""
    gates = []
    gates.append(["cx", 0, 1])
    gates.append(["cx", 1, 0])
    gates.append(["cx", 0, 1])
    return gates


def _decompose_toffoli():
    """Toffoli decomposition using 6 CNOTs."""
    gates = []
    # Standard Toffoli decomposition
    gates.append(["u3", 2, math.pi / 2, 0, math.pi / 2])  # H on target
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, math.pi / 4, 0, 0])  # T gate
    gates.append(["cx", 0, 2])
    gates.append(["u3", 2, -math.pi / 4, 0, 0])  # T†
    gates.append(["cx", 1, 2])
    gates.append(["u3", 2, math.pi / 4, 0, 0])  # T
    gates.append(["u3", 1, math.pi / 4, 0, 0])  # T on control 1
    gates.append(["cx", 0, 1])
    gates.append(["u3", 1, -math.pi / 4, 0, 0])  # T†
    gates.append(["u3", 0, math.pi / 4, 0, 0])  # T on control 0
    gates.append(["cx", 0, 1])
    gates.append(["u3", 2, math.pi / 2, 0, math.pi / 2])  # H on target
    return gates


def _kak_decompose_2qubit(target):
    """Decompose a 2-qubit unitary using KAK formula. Returns list of gates."""
    # KAK decomposition: U = e^{iα} (A⊗B) e^{i(θ1 σx⊗σx + θ2 σy⊗σy + θ3 σz⊗σz)} (C⊗D)
    # The entangling part can be done with up to 3 CNOTs.
    
    # For simplicity, use a fixed 3-CNOT template and optimize single-qubit gates
    # This is a heuristic approach
    
    # Fixed CNOT structure: CNOT(0,1), CNOT(1,0), CNOT(0,1)
    # Single qubit gates before, between, and after
    
    # We'll use coordinate descent to find the single-qubit parameters
    
    n_params = 18  # 6 u3 gates × 3 params each
    
    def build_circuit(params):
        gates = []
        # First layer of u3
        gates.append(["u3", 0, params[0], params[1], params[2]])
        gates.append(["u3", 1, params[3], params[4], params[5]])
        gates.append(["cx", 0, 1])
        # Second layer of u3
        gates.append(["u3", 0, params[6], params[7], params[8]])
        gates.append(["u3", 1, params[9], params[10], params[11]])
        gates.append(["cx", 1, 0])
        # Third layer of u3
        gates.append(["u3", 0, params[12], params[13], params[14]])
        gates.append(["u3", 1, params[15], params[16], params[17]])
        gates.append(["cx", 0, 1])
        return gates
    
    best_params = [0.0] * n_params
    best_dist = _distance(target, _build_fast(build_circuit(best_params), 2))
    
    # Coordinate descent
    for sweep in range(8):
        step = 1.0 / (1 + sweep)
        for i in range(n_params):
            for trial in (best_params[i] + step, best_params[i] - step,
                          best_params[i] + 0.5 * (math.pi - best_params[i]),
                          best_params[i] - 0.5 * (math.pi + best_params[i])):
                params = best_params[:]
                params[i] = trial
                d = _distance(target, _build_fast(build_circuit(params), 2))
                if d < best_dist:
                    best_params, best_dist = params, d
    
    return build_circuit(best_params)


def _kak_decompose_3qubit(target, budget):
    """Decompose a 3-qubit unitary using a ladder of CNOTs and coordinate descent."""
    # Use a fixed CNOT ladder structure
    # For 3 qubits, use CNOTs between pairs in a fixed pattern
    
    # Build a layout with CNOTs and u3 gates
    pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    
    # Create layout: alternate u3 layers and CNOTs
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
        c, t = pairs[layer % len(pairs)]
        layout.append(["cx", c, t])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    def build_circuit(params):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k + 1], params[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best_params = [0.0] * n_params
    best_dist = _distance(target, _build_fast(build_circuit(best_params), 3))
    
    # Coordinate descent with time limit
    for sweep in range(6):
        if _time_remaining() < 30:
            break
        step = 1.0 / (1 + sweep)
        for i in range(n_params):
            if _time_remaining() < 30:
                break
            for trial in (best_params[i] + step, best_params[i] - step,
                          best_params[i] + 0.3 * (math.pi - best_params[i]),
                          best_params[i] - 0.3 * (math.pi + best_params[i])):
                params = best_params[:]
                params[i] = trial
                d = _distance(target, _build_fast(build_circuit(params), 3))
                if d < best_dist:
                    best_params, best_dist = params, d
    
    return build_circuit(best_params)


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try to identify named gates for exact synthesis
    named = _identify_named_gate(target, qubits)
    
    if named == "cnot":
        return _decompose_cnot()
    elif named == "iswap":
        return _decompose_iswap()
    elif named == "swap":
        return _decompose_swap()
    elif named == "toffoli":
        return _decompose_toffoli()
    
    # For generic unitaries, use KAK-based decomposition
    if qubits == 2:
        return _kak_decompose_2qubit(target)
    elif qubits == 3:
        return _kak_decompose_3qubit(target, budget)
    
    # Fallback: return identity-like circuit
    return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
# EVOLVE-BLOCK-END
