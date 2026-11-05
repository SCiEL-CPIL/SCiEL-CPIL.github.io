"""Quantum circuit synthesis: exact decompositions for named gates, QSD for generics."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_DEADLINE = 260  # safety margin before 270s wall-clock kill
_START = time.time()


def _elapsed():
    return time.time() - _START


def _time_exceeded():
    return _elapsed() > _DEADLINE


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


def _synthesize_cnot(problem):
    target = _target(problem)
    expected = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
    if np.allclose(target, expected, atol=1e-10):
        return [["cx", 0, 1]]
    return None


def _synthesize_iswap(problem):
    target = _target(problem)
    expected = np.array([[1, 0, 0, 0],
                         [0, 0, 1j, 0],
                         [0, 1j, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
    if np.allclose(target, expected, atol=1e-10):
        # iSWAP decomposition using 2 CNOTs
        # iSWAP = (H⊗H) · CNOT · (Ry(π/4)⊗Ry(π/4)) · CNOT · (H⊗H)
        gates = []
        # H on both
        gates.append(["u3", 0, math.pi, 0, math.pi])
        gates.append(["u3", 1, math.pi, 0, math.pi])
        # CNOT 0->1
        gates.append(["cx", 0, 1])
        # Ry(π/4) on both
        gates.append(["u3", 0, math.pi / 4, 0, math.pi / 2])
        gates.append(["u3", 1, math.pi / 4, 0, math.pi / 2])
        # CNOT 0->1
        gates.append(["cx", 0, 1])
        # H on both
        gates.append(["u3", 0, math.pi, 0, math.pi])
        gates.append(["u3", 1, math.pi, 0, math.pi])
        return gates
    return None


def _synthesize_swap(problem):
    target = _target(problem)
    expected = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
    if np.allclose(target, expected, atol=1e-10):
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    return None


def _synthesize_toffoli(problem):
    target = _target(problem)
    expected = np.eye(8, dtype=complex)
    expected[6, 6] = 0
    expected[6, 7] = 1
    expected[7, 6] = 1
    expected[7, 7] = 0
    if np.allclose(target, expected, atol=1e-10):
        # Standard Toffoli decomposition with 6 CNOTs
        gates = []
        gates.append(["u3", 2, math.pi, 0, math.pi])
        gates.append(["u3", 2, math.pi / 4, 0, 0])
        gates.append(["cx", 1, 2])
        gates.append(["u3", 2, -math.pi / 4, 0, 0])
        gates.append(["cx", 0, 2])
        gates.append(["u3", 2, math.pi / 4, 0, 0])
        gates.append(["cx", 1, 2])
        gates.append(["u3", 2, -math.pi / 4, 0, 0])
        gates.append(["cx", 0, 2])
        gates.append(["u3", 1, math.pi / 4, 0, 0])
        gates.append(["u3", 0, math.pi / 4, 0, 0])
        gates.append(["cx", 0, 1])
        gates.append(["u3", 1, -math.pi / 4, 0, 0])
        gates.append(["u3", 0, -math.pi / 4, 0, 0])
        gates.append(["u3", 2, math.pi, 0, math.pi])
        return gates
    return None


def _is_known_gate(problem):
    name = problem.get("name", "").lower()
    target = _target(problem)
    qubits = problem["qubits"]
    
    if qubits == 2:
        if "cnot" in name or "cx" in name:
            result = _synthesize_cnot(problem)
            if result:
                return result
        if "iswap" in name or "isw" in name:
            result = _synthesize_iswap(problem)
            if result:
                return result
        if "swap" in name and "iswap" not in name:
            result = _synthesize_swap(problem)
            if result:
                return result
    elif qubits == 3:
        if "toffoli" in name or "ccx" in name:
            result = _synthesize_toffoli(problem)
            if result:
                return result
    
    return None


def _decompose_u3(target):
    """Decompose a 2x2 unitary into u3 parameters (theta, phi, lam)."""
    # target = [[a, b], [c, d]]
    # u3(theta, phi, lam) = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
    #                         [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    a, b = target[0, 0], target[0, 1]
    c, d = target[1, 0], target[1, 1]
    
    # Handle edge cases
    if abs(a) < 1e-12:
        # cos(theta/2) ~ 0, theta ~ pi
        theta = math.pi
        # a = cos(theta/2) ~ 0, so use b and c
        # b = -exp(i*lam)*sin(theta/2) => exp(i*lam) = -b/sin(theta/2)
        # c = exp(i*phi)*sin(theta/2) => exp(i*phi) = c/sin(theta/2)
        if abs(b) > 1e-12:
            lam = np.angle(-b / abs(b))  # sin(theta/2)=1 when theta=pi
            lam = np.angle(-b)
        else:
            lam = 0
        if abs(c) > 1e-12:
            phi = np.angle(c)
        else:
            phi = 0
        return theta, phi, lam
    else:
        theta = 2 * math.acos(min(1.0, abs(a)))
        if theta < 1e-12:
            return 0.0, 0.0, 0.0
        # a = cos(theta/2), so a is real and positive (up to global phase)
        # Actually a may have a phase. Let's be more careful.
        # The u3 form has a = cos(theta/2) which is real.
        # So we need to account for global phase.
        # Let's use a different approach: match the structure.
        
        # From u3: a = cos(theta/2), b = -exp(i*lam)*sin(theta/2)
        # c = exp(i*phi)*sin(theta/2), d = exp(i*(phi+lam))*cos(theta/2)
        # So a and d should be related: d = exp(i*(phi+lam)) * a
        # And c = exp(i*phi) * sin(theta/2)
        
        # If a is not real, there's a global phase. But u3 as defined has real a.
        # So the target must be absorbable. Let's just solve directly.
        
        # cos(theta/2) = |a| (taking magnitude)
        # But a might have phase. In the standard u3, a is real.
        # This means the target 2x2 unitary must have a[0,0] real (up to global phase).
        # Since we can always multiply by global phase, let's work with the matrix.
        
        # Better approach: use the fact that for any 2x2 unitary U,
        # we can write U = e^{i*gamma} * u3(theta, phi, lam)
        # where e^{i*gamma} is a global phase that doesn't matter for circuits.
        
        # So let's absorb the global phase.
        # a = e^{i*gamma} * cos(theta/2)
        # cos(theta/2) = |a|
        # e^{i*gamma} = a / |a|
        
        cos_half = abs(a)
        if cos_half > 1.0:
            cos_half = 1.0
        theta = 2 * math.acos(cos_half)
        
        if theta < 1e-12:
            return 0.0, 0.0, 0.0
        
        sin_half = math.sin(theta / 2)
        global_phase = a / cos_half if cos_half > 1e-12 else 1.0
        
        # b = e^{i*gamma} * (-exp(i*lam) * sin(theta/2))
        # b / (e^{i*gamma} * sin_half) = -exp(i*lam)
        # exp(i*lam) = -b / (e^{i*gamma} * sin_half)
        if sin_half > 1e-12:
            exp_i_lam = -b / (global_phase * sin_half)
            lam = np.angle(exp_i_lam)
        else:
            lam = 0.0
        
        # c = e^{i*gamma} * exp(i*phi) * sin(theta/2)
        # exp(i*phi) = c / (e^{i*gamma} * sin_half)
        if sin_half > 1e-12:
            exp_i_phi = c / (global_phase * sin_half)
            phi = np.angle(exp_i_phi)
        else:
            phi = 0.0
        
        return theta, phi, lam


def _qsdecompose_2qubit(target, budget):
    """Quantum Shannon Decomposition for 2-qubit unitary."""
    # For 2 qubits, use the standard decomposition.
    # U = (A ⊗ B) · CNOT · (C ⊗ D) · CNOT · (E ⊗ F)
    # with appropriate single-qubit gates.
    # 
    # A simpler approach: use the known fact that any 2-qubit unitary
    # can be decomposed with at most 3 CNOTs.
    # 
    # Let's use a recursive approach:
    # U = (A0 ⊗ A1) · CX(0,1) · (B0 ⊗ B1) · CX(0,1) · (C0 ⊗ C1)
    # 
    # Actually, let's use a more direct method.
    # 
    # For 2 qubits, we can use the following decomposition:
    # U = (U00 ⊗ U11) · CX · (U01 ⊗ U10) · CX · (U00' ⊗ U11')
    # where the single-qubit gates are determined by the target.
    
    # Let's use a practical approach: try to find the decomposition
    # by matching the structure.
    
    # A simpler and more robust approach for 2 qubits:
    # Use the fact that U can be written as:
    # U = (A ⊗ B) · CX(0,1) · (C ⊗ D) · CX(1,0) · (E ⊗ F)
    # or similar layouts.
    
    # For now, let's use a coordinate descent approach but with better initialization.
    # Actually, let's try to use QSD properly.
    
    # QSD for 2 qubits:
    # U = (A ⊗ I) · CX(0,1) · (B ⊗ C) · CX(0,1) · (D ⊗ E)
    # This uses 2 CNOTs. But we might need 3 for a generic unitary.
    
    # Let's use a 3-CNOT layout and optimize.
    # Layout: (A0⊗A1) CX(0,1) (B0⊗B1) CX(1,0) (C0⊗C1) CX(0,1) (D0⊗D1)
    
    if budget < 1:
        # 0 CNOTs: U = A ⊗ B
        # This only works if U is a product state.
        # Try to decompose as product.
        # If U[0,0] != 0, A = U[0:2, 0:2] / U[0,0], B = U[2:4, 2:4] / U[0,0]
        # This is approximate. Let's just return identity for now.
        return []
    
    # Use coordinate descent with the 3-CNOT layout
    layout = []
    for q in range(2):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    layout.append(["cx", 0, 1])
    for q in range(2):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    layout.append(["cx", 1, 0])
    for q in range(2):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    layout.append(["cx", 0, 1])
    for q in range(2):
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
    
    best = params[:]
    best_distance = _distance(target, _build_fast(circuit(best), 2))
    
    rng = random.Random(4242)
    deadline = _DEADLINE - 10
    
    sweep = 0
    while _elapsed() < deadline and sweep < 50:
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(len(best)):
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), 2))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
            if _time_exceeded():
                break
        if not improved:
            break
        sweep += 1
    
    return circuit(best)


def _qsdecompose_3qubit(target, budget):
    """Quantum Shannon Decomposition for 3-qubit unitary."""
    # For 3 qubits, use a recursive QSD approach.
    # U = (A ⊗ I4) · CX(0,1) · (B ⊗ C) · CX(0,2) · (D ⊗ E) · ...
    # 
    # A practical approach: use a fixed layout of CNOTs and optimize.
    # The minimum number of CNOTs for a generic 3-qubit unitary is 15,
    # but we may have more budget.
    
    if budget < 1:
        return []
    
    # Use a repeating pattern of CNOTs
    pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    
    layout = []
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    num_cnots = min(budget, 15)  # Cap at 15 for efficiency
    for i in range(num_cnots):
        c, t = pairs[i % len(pairs)]
        layout.append(["cx", c, t])
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    for q in range(3):
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
    
    best = params[:]
    best_distance = _distance(target, _build_fast(circuit(best), 3))
    
    rng = random.Random(4243)
    deadline = _DEADLINE - 10
    
    sweep = 0
    while _elapsed() < deadline and sweep < 30:
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(len(best)):
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), 3))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
            if _time_exceeded():
                break
        if not improved:
            break
        sweep += 1
    
    return circuit(best)


def synthesize(problem):
    """Synthesize a quantum circuit for the given problem."""
    if _time_exceeded():
        return []
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Check for known gates
    known = _is_known_gate(problem)
    if known is not None:
        return known
    
    target = _target(problem)
    
    if qubits == 2:
        return _qsdecompose_2qubit(target, budget)
    elif qubits == 3:
        return _qsdecompose_3qubit(target, budget)
    else:
        return []
# EVOLVE-BLOCK-END
