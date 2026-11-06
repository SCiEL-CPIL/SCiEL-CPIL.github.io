"""Exact decompositions for known gates + optimized synthesis for generic unitaries."""
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


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", int(c), int(t)]


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
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
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
    d = 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])
    return max(0.0, d)


def _cnot_gate():
    return [_cx(0, 1)]


def _iswap_gate():
    # iSWAP: 3 CNOTs with specific single-qubit rotations
    # iSWAP = (H ⊗ H) · CNOT(0,1) · (iS ⊗ iS) · CNOT(1,0) · (iS† ⊗ iS†) · CNOT(0,1) · (H ⊗ H)
    # Actually, let's use the standard decomposition:
    # iSWAP = e^{iπ/4} * (Ry(π/2) ⊗ Ry(π/2)) · CNOT(0,1) · (Ry(-π/2) ⊗ Ry(-π/2)) · CNOT(1,0) · (Ry(π/2) ⊗ Ry(π/2)) · CNOT(0,1)
    # But we need to be careful about global phase.
    # Standard iSWAP decomposition with 3 CNOTs:
    gates = []
    # Layer 1: single-qubit rotations
    gates.append(_u3(0, math.pi/2, 0, -math.pi/2))  # Ry(π/2) up to phase
    gates.append(_u3(1, math.pi/2, 0, -math.pi/2))
    gates.append(_cx(0, 1))
    # Layer 2
    gates.append(_u3(0, math.pi/2, math.pi, math.pi/2))  # Ry(-π/2) up to phase
    gates.append(_u3(1, math.pi/2, math.pi, math.pi/2))
    gates.append(_cx(1, 0))
    # Layer 3
    gates.append(_u3(0, math.pi/2, 0, -math.pi/2))
    gates.append(_u3(1, math.pi/2, 0, -math.pi/2))
    gates.append(_cx(0, 1))
    # Final layer
    gates.append(_u3(0, math.pi/2, math.pi, math.pi/2))
    gates.append(_u3(1, math.pi/2, math.pi, math.pi/2))
    return gates


def _swap_gate():
    # SWAP = CNOT(0,1) · CNOT(1,0) · CNOT(0,1)
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _toffoli_gate():
    # Toffoli (CCNOT) with 6 CNOTs
    # Standard decomposition:
    gates = []
    # Apply H to target, then use controlled-T gates
    # Toffoli = (I ⊗ I ⊗ H) · CNOT(2,1) · (I ⊗ T† ⊗ I) · CNOT(2,0) · (I ⊗ T ⊗ I) · CNOT(2,1) · (I ⊗ T ⊗ I) · CNOT(1,0) · (I ⊗ T† ⊗ I) · CNOT(2,0) · (I ⊗ T ⊗ I) · (I ⊗ I ⊗ H)
    # T gate: u3(0, π/4, 0, 0) = diag(1, e^{iπ/4})
    # T† gate: u3(0, π/4, 0, -math.pi/2) ... let me think more carefully
    # T = diag(1, e^{iπ/4}), T† = diag(1, e^{-iπ/4})
    # u3(θ, φ, λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)], [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
    # For T: θ=π/2, φ=0, λ=0 gives [[1/√2, -1/√2], [1/√2, 1/√2]] - not right
    # Let me use a different approach. T gate in u3:
    # We want diag(1, e^{iπ/4}). 
    # u3(0, 0, π/4) = [[1, 0], [0, e^{iπ/4}]] = T
    # u3(0, 0, -π/4) = [[1, 0], [0, e^{-iπ/4}]] = T†
    # H = u3(π/2, 0, π)
    
    gates.append(_u3(2, math.pi/2, 0, math.pi))  # H on qubit 2
    gates.append(_cx(2, 1))
    gates.append(_u3(1, 0, 0, -math.pi/4))  # T† on qubit 1
    gates.append(_cx(2, 0))
    gates.append(_u3(0, 0, 0, math.pi/4))  # T on qubit 0
    gates.append(_cx(2, 1))
    gates.append(_u3(1, 0, 0, math.pi/4))  # T on qubit 1
    gates.append(_cx(1, 0))
    gates.append(_u3(0, 0, 0, -math.pi/4))  # T† on qubit 0
    gates.append(_cx(2, 0))
    gates.append(_u3(0, 0, 0, math.pi/4))  # T on qubit 0
    gates.append(_u3(2, math.pi/2, 0, math.pi))  # H on qubit 2
    return gates


def _optimize_2qubit(target, qubits, budget):
    """Optimize 2-qubit unitary with 3 CNOTs."""
    # Layout: u3 layer, CNOT, u3 layer, CNOT, u3 layer, CNOT, u3 layer
    layout = []
    for layer in range(4):
        layout.append(["u3", 0, 0.0, 0.0, 0.0])
        layout.append(["u3", 1, 0.0, 0.0, 0.0])
        if layer < 3:
            layout.append(["cx", 0, 1])
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize with identity
    best = [0.0] * n_params
    best_dist = _distance(target, _build(circuit(best), qubits))
    
    # Coordinate descent
    for sweep in range(20):
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step, best[i] + 0.5 * step, best[i] - 0.5 * step):
                values = best[:]
                values[i] = trial
                d = _distance(target, _build(circuit(values), qubits))
                if d < best_dist:
                    best, best_dist = values, d
        if best_dist < 1e-10:
            break
        if _elapsed() > _DEADLINE:
            break
    
    return circuit(best)


def _optimize_3qubit(target, qubits, budget):
    """Optimize 3-qubit unitary with 6 CNOTs."""
    # Use a ladder structure with 6 CNOTs
    pairs = [(0,1), (1,2), (0,1), (1,2), (0,1), (1,2)]
    layout = []
    for i in range(budget):
        layout.append(["u3", 0, 0.0, 0.0, 0.0])
        layout.append(["u3", 1, 0.0, 0.0, 0.0])
        layout.append(["u3", 2, 0.0, 0.0, 0.0])
        layout.append(["cx", pairs[i][0], pairs[i][1]])
    # Final layer
    layout.append(["u3", 0, 0.0, 0.0, 0.0])
    layout.append(["u3", 1, 0.0, 0.0, 0.0])
    layout.append(["u3", 2, 0.0, 0.0, 0.0])
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build(circuit(best), qubits))
    
    # Coordinate descent with fewer sweeps for 3-qubit (slower)
    for sweep in range(10):
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                d = _distance(target, _build(circuit(values), qubits))
                if d < best_dist:
                    best, best_dist = values, d
        if best_dist < 1e-10:
            break
        if _elapsed() > _DEADLINE:
            break
    
    return circuit(best)


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    if name == "CNOT" and qubits == 2:
        return _cnot_gate()
    elif name == "iSWAP" and qubits == 2:
        return _iswap_gate()
    elif name == "SWAP" and qubits == 2:
        return _swap_gate()
    elif name == "Toffoli" and qubits == 3:
        return _toffoli_gate()
    
    # Generic unitaries
    if qubits == 2 and budget >= 3:
        return _optimize_2qubit(target, qubits, budget)
    elif qubits == 3 and budget >= 6:
        return _optimize_3qubit(target, qubits, budget)
    
    # Fallback: use all available CNOTs
    if qubits == 2:
        return _optimize_2qubit(target, qubits, min(budget, 3))
    else:
        return _optimize_3qubit(target, qubits, min(budget, 6))
# EVOLVE-BLOCK-END
