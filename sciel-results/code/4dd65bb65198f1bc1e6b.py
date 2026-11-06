"""Quantum circuit synthesis with exact known-gate shortcuts and coordinate descent for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _apply_u3_to_matrix(matrix, q, qubits, theta, phi, lam):
    """Apply u3 gate to qubit q in-place on the matrix (rows)."""
    n = 2 ** qubits
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    bit_pos = qubits - 1 - q
    shift = 1 << bit_pos
    new_matrix = np.zeros_like(matrix)
    for i in range(n):
        if (i & shift) == 0:
            j = i | shift
            new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
            new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
    return new_matrix


def _apply_cnot_to_matrix(matrix, control, target, qubits):
    """Apply CNOT gate to the matrix (rows)."""
    n = 2 ** qubits
    cshift = 1 << (qubits - 1 - control)
    tshift = 1 << (qubits - 1 - target)
    perm = np.arange(n)
    for i in range(n):
        if (i & cshift):
            perm[i] = i ^ tshift
    return matrix[perm, :]


def _build_matrix(gates, qubits):
    """Build the full unitary from a gate list."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            matrix = _apply_u3_to_matrix(matrix, q, qubits, theta, phi, lam)
        else:
            _, control, target_q = gate
            matrix = _apply_cnot_to_matrix(matrix, control, target_q, qubits)
    return matrix


def _distance(target, matrix):
    n = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _build_fast(gates, qubits, target):
    """Build circuit and compute distance in one pass."""
    matrix = _build_matrix(gates, qubits)
    return _distance(target, matrix)


def _known_cnot_2q():
    """CNOT gate on 2 qubits: qubit 0 -> qubit 1."""
    return [_cx(0, 1)]


def _known_iswap_2q():
    """iSWAP gate on 2 qubits: qubit 0 <-> qubit 1 with phase i."""
    # iSWAP = [[0, 0, 0, i], [0, 0, i, 0], [0, i, 0, 0], [i, 0, 0, 0]]
    # Decomposition: 
    # u3(0, pi/2, 0, pi/2)  ->  |0>->(|0>+i|1>)/sqrt2, |1>->(i|0>+|1>)/sqrt2 ... 
    # Actually let's use a known decomposition:
    # iSWAP = (H on both) * (CNOT 0->1) * (H on both) * (S on 0) * (S on 1) * (CNOT 1->0) * (H on both) * (S on 0) * (S on 1)
    # Hmm, let me use a simpler known decomposition.
    # iSWAP = [[0,0,0,i],[0,0,i,0],[0,i,0,0],[i,0,0,0]]
    # One known decomposition:
    # u3(0, pi, 0, pi/2) -> i*Z on q0? No.
    # Let me think: iSWAP = exp(i*pi/4 * (X0*X1 + Y0*Y1))
    # Standard decomposition uses 3 CNOTs but we only have budget of... let me check.
    # For 2-qubit gates the budget should be enough.
    # Known decomposition of iSWAP:
    # u3(0, pi/2, 0, pi/2)
    # u3(1, pi/2, 0, pi/2)
    # cx(0,1)
    # u3(0, pi/2, pi/2, 0)
    # u3(1, pi/2, pi/2, 0)
    # cx(0,1)
    # u3(0, pi/2, 0, pi/2)
    # u3(1, pi/2, 0, pi/2)
    # Wait, that's 2 CNOTs. Let me verify mentally...
    # Actually a cleaner decomposition:
    # iSWAP = (Ry(pi/2) on q0) * (Ry(pi/2) on q1) * CNOT(0,1) * (Ry(-pi/2) on q0) * (Ry(-pi/2) on q1) * CNOT(0,1) * (Ry(pi/2) on q0) * (Ry(pi/2) on q1)
    # Hmm, let me just use a verified decomposition.
    # iSWAP matrix in basis |00>,|01>,|10>,|11>:
    # [[0, 0, 0, i], [0, 0, i, 0], [0, i, 0, 0], [i, 0, 0, 0]]
    # This is equivalent to: swap with phase i on off-diagonal.
    # Known: iSWAP = (S on q0) * (S on q1) * (H on q0) * (H on q1) * CNOT(0,1) * (H on q0) * (H on q1) * CNOT(1,0) * (H on q0) * (H on q1) * (S† on q0) * (S† on q1)
    # That's 2 CNOTs.
    # Let me use: u3(q, pi/2, 0, pi/2) = (|0> + i|1>)/sqrt2 mapping... 
    # Actually u3(theta, phi, lam) = [[cos(t/2), -e^{i*lam}sin(t/2)], [e^{i*phi}sin(t/2), e^{i(phi+lam)}cos(t/2)]]
    # u3(pi/2, 0, pi/2) = [[cos(pi/4), -i*sin(pi/4)], [0, i*cos(pi/4)]] = 1/sqrt(2)*[[1, -i], [0, i]]
    # This is getting complex. Let me just use a standard 3-CNOT decomposition or find the right one.
    # For now, let's use a simple approach: the iSWAP can be done with:
    # u3(0, pi, 0, pi) = i*I? No, u3(pi, 0, pi) = [[cos(pi/2), -e^{ipi}sin(pi/2)], [e^{i0}sin(pi/2), e^{i(pi+pi)}cos(pi/2)]] = [[0, i], [1, 0]]
    # That's a Y/2 rotation with phase... 
    # Let me just hardcode a working decomposition. I'll use the fact that:
    # iSWAP = (Ry(pi/2) on 0) * (Ry(pi/2) on 1) * CNOT(0,1) * (Ry(-pi/2) on 0) * (Ry(-pi/2) on 1) * CNOT(0,1) * (Ry(pi/2) on 0) * (Ry(pi/2) on 1)
    # Ry(t) = u3(t, 0, 0)
    gates = []
    gates.append(_u3(0, math.pi/2, 0, 0))
    gates.append(_u3(1, math.pi/2, 0, 0))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, -math.pi/2, 0, 0))
    gates.append(_u3(1, -math.pi/2, 0, 0))
    gates.append(_cx(0, 1))
    gates.append(_u3(0, math.pi/2, 0, 0))
    gates.append(_u3(1, math.pi/2, 0, 0))
    return gates


def _known_swap_2q():
    """SWAP gate on 2 qubits: 3 CNOTs."""
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _known_toffoli_3q():
    """Toffoli gate on 3 qubits: controls q0, q1, target q2."""
    # Standard Toffoli decomposition with 6 CNOTs
    # u3(2, pi/8, 0, 0)
    # cx(1, 2)
    # u3(2, -pi/8, 0, 0)
    # cx(0, 2)
    # u3(2, pi/8, 0, 0)
    # cx(1, 2)
    # u3(2, -pi/8, 0, 0)
    # u3(1, pi/8, 0, 0)
    # u3(2, pi/8, 0, 0)
    # cx(0, 1)
    # u3(1, -pi/8, 0, 0)
    # u3(2, -pi/8, 0, 0)
    # cx(0, 2)
    # u3(1, pi/8, 0, 0)
    # u3(2, pi/8, 0, 0)
    # cx(1, 2)
    # u3(2, -pi/8, 0, 0)
    # u3(1, 3*pi/8, 0, pi/2)
    # u3(2, 3*pi/8, 0, pi/2)
    # u3(0, pi/2, 0, 0)
    gates = []
    p8 = math.pi / 8
    gates.append(_u3(2, p8, 0, 0))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, -p8, 0, 0))
    gates.append(_cx(0, 2))
    gates.append(_u3(2, p8, 0, 0))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, -p8, 0, 0))
    gates.append(_u3(1, p8, 0, 0))
    gates.append(_u3(2, p8, 0, 0))
    gates.append(_cx(0, 1))
    gates.append(_u3(1, -p8, 0, 0))
    gates.append(_u3(2, -p8, 0, 0))
    gates.append(_cx(0, 2))
    gates.append(_u3(1, p8, 0, 0))
    gates.append(_u3(2, p8, 0, 0))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, -p8, 0, 0))
    gates.append(_u3(1, 3*p8, 0, math.pi/2))
    gates.append(_u3(2, 3*p8, 0, math.pi/2))
    gates.append(_u3(0, math.pi/2, 0, 0))
    return gates


def _try_known_circuit(problem):
    """Try to identify the target as a known gate and return its exact decomposition."""
    target = _target(problem)
    qubits = problem["qubits"]
    name = problem.get("name", "")
    
    # Check for CNOT (2 qubits)
    if qubits == 2:
        cnot_matrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 1, 0]], dtype=complex)
        # Check if target matches CNOT up to global phase
        diff = target - cnot_matrix
        if np.max(np.abs(diff)) < 1e-10:
            return _known_cnot_2q()
        
        # Check iSWAP
        iswap_matrix = np.array([[0, 0, 0, 1j],
                                  [0, 0, 1j, 0],
                                  [0, 1j, 0, 0],
                                  [1j, 0, 0, 0]], dtype=complex)
        diff = target - iswap_matrix
        if np.max(np.abs(diff)) < 1e-10:
            return _known_iswap_2q()
        
        # Check SWAP
        swap_matrix = np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]], dtype=complex)
        diff = target - swap_matrix
        if np.max(np.abs(diff)) < 1e-10:
            return _known_swap_2q()
    
    # Check for Toffoli (3 qubits)
    if qubits == 3:
        # Toffoli: |110> -> |111>, others unchanged
        toffoli_matrix = np.eye(8, dtype=complex)
        # In basis |q2 q1 q0> where q0 is LSB... wait, qubit 0 is MSB
        # So basis index: q0*4 + q1*2 + q2*1
        # Toffoli: control q0, q1; target q2
        # |110> (q0=1,q1=1,q2=0) -> |111> (q0=1,q1=1,q2=1)
        # Index of |110>: 1*4+1*2+0*1 = 6
        # Index of |111>: 1*4+1*2+1*1 = 7
        toffoli_matrix[6, 6] = 0
        toffoli_matrix[6, 7] = 1
        toffoli_matrix[7, 6] = 1
        toffoli_matrix[7, 7] = 0
        
        diff = target - toffoli_matrix
        if np.max(np.abs(diff)) < 1e-10:
            return _known_toffoli_3q()
    
    return None


def _coordinate_descent(problem, target, qubits, budget):
    """Coordinate descent to find single-qubit rotations for a fixed CNOT ladder."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits)
    
    # Build layout: alternating u3 layers and CNOTs
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _build_fast(circuit(best), qubits, target)
    
    # Coordinate descent with adaptive step sizes
    sweeps = 30
    for sweep in range(sweeps):
        if time.time() - _START_TIME > _DEADLINE:
            break
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(n_params):
            if time.time() - _START_TIME > _DEADLINE:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value
            # Random perturbation
            if rng.random() < 0.3:
                trial = best[i] + rng.uniform(-math.pi, math.pi)
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value
    
    # Polish pass with smaller steps
    for sweep in range(15):
        if time.time() - _START_TIME > _DEADLINE:
            break
        step = 0.1 / (1 + sweep * 0.5)
        for i in range(n_params):
            if time.time() - _START_TIME > _DEADLINE:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value
    
    return circuit(best)


def synthesize(problem):
    """Synthesize a circuit for the given target unitary."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # First try known gate shortcuts
    known = _try_known_circuit(problem)
    if known is not None:
        # Verify the known circuit
        dist = _build_fast(known, qubits, target)
        if dist < 1e-6:
            return known
        # If verification fails, fall through to numerical optimization
    
    # Use coordinate descent for generic unitaries
    return _coordinate_descent(problem, target, qubits, budget)
# EVOLVE-BLOCK-END
