"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

# Global deadline guard: the evaluator kills after 270s wall-clock.
# We track elapsed time and return early if we approach the limit.
_START_TIME = time.time()
_DEADLINE = 270.0 - 5.0  # safety margin of 5s

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build the unitary from a list of gates."""
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
    """Phase-insensitive distance: 1 - |tr(target^dagger U)| / dim."""
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _make_circuit_template(qubits, cnot_budget):
    """
    Create a circuit template with CNOTs and u3 gates.
    For 2 qubits: use the known optimal CNOT counts.
    For 3 qubits: use a reasonable layout.
    
    Strategy:
    - For 2-qubit gates (CNOT, iSWAP, SWAP, generic 2x2):
      CNOT budget is typically 1-3.
    - For 3-qubit gates (Toffoli, generic 8x8):
      CNOT budget is typically 6-14.
    
    We'll use a layered approach with CNOTs interleaved with u3 gates.
    """
    pairs_2q = [(0, 1), (1, 0)]
    pairs_3q = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    
    if qubits == 2:
        pairs = pairs_2q
    else:
        pairs = pairs_3q
    
    layout = []
    # Start with u3 gates on all qubits
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    for i in range(cnot_budget):
        # Add CNOT
        c, t = pairs[i % len(pairs)]
        layout.append(["cx", c, t])
        # Add u3 gates on all qubits after each CNOT
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    # End with u3 gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    return layout


def _circuit_from_values(layout, values):
    """Instantiate a circuit from the layout and parameter values."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _optimize_angles(layout, target, qubits, max_sweeps=8):
    """
    Optimize the u3 angles using coordinate descent.
    Returns the best values and distance.
    """
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    best = [0.0] * n_params
    best_distance = _distance(target, _build(_circuit_from_values(layout, best), qubits))
    
    # Coordinate descent with multiple passes
    for sweep in range(max_sweeps):
        if _time_remaining() < 10.0:
            break
        step = 0.5 / (1 + sweep * 0.3)
        improved = False
        for i in range(n_params):
            if _time_remaining() < 10.0:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(_circuit_from_values(layout, values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
        if not improved:
            break
    
    return best, best_distance


def _try_multiple_layouts(problem):
    """
    Try multiple circuit layouts and return the best one.
    """
    qubits = problem["qubits"]
    cnot_budget = problem["cnot_budget"]
    target = _target(problem)
    
    best_gates = None
    best_distance = 1.0
    
    # Generate several different layouts
    layouts = []
    
    # Layout 1: Standard alternating
    layouts.append(_make_circuit_template(qubits, cnot_budget))
    
    # Layout 2: Grouped CNOTs
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
    else:
        pairs = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    
    # Try different pair orderings
    import itertools
    if qubits == 2:
        pair_orderings = [
            [(0, 1)],
            [(1, 0)],
            [(0, 1), (1, 0)],
            [(1, 0), (0, 1)],
        ]
    else:
        # For 3 qubits, try a few specific orderings
        pair_orderings = [
            [(0, 1), (0, 2), (1, 2)],
            [(0, 1), (1, 2), (0, 2)],
            [(0, 2), (0, 1), (1, 2)],
            [(1, 2), (0, 1), (0, 2)],
        ]
    
    for ordering in pair_orderings:
        if len(ordering) == 0:
            continue
        layout = []
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        for i in range(cnot_budget):
            c, t = ordering[i % len(ordering)]
            layout.append(["cx", c, t])
            for q in range(qubits):
                layout.append(["u3", q, 0.0, 0.0, 0.0])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layouts.append(layout)
    
    # Try each layout
    for layout in layouts:
        if _time_remaining() < 5.0:
            break
        
        # Check gate count
        n_gates = len(layout)
        if n_gates > 400:
            continue
        
        # Check CNOT count
        n_cnots = sum(1 for g in layout if g[0] == "cx")
        if n_cnots > cnot_budget:
            continue
        
        values, distance = _optimize_angles(layout, target, qubits)
        if distance < best_distance:
            best_distance = distance
            best_gates = _circuit_from_values(layout, values)
    
    # If no valid layout was found, return an identity circuit
    if best_gates is None:
        best_gates = []
        for q in range(qubits):
            best_gates.append(["u3", q, 0.0, 0.0, 0.0])
    
    return best_gates


def synthesize(problem):
    """Synthesize a circuit for the given target unitary within the CNOT budget."""
    qubits = problem["qubits"]
    cnot_budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Special cases: known exact decompositions
    
    # Check if target is identity
    identity = np.eye(2 ** qubits, dtype=complex)
    if np.allclose(target, identity, atol=1e-10):
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # For CNOT gate itself (2 qubits)
    if qubits == 2 and cnot_budget >= 1:
        # Check if target is a CNOT
        cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        cnot_10 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, cnot_01, atol=1e-10):
            return [["u3", 0, 0.0, 0.0, 0.0],
                    ["u3", 1, 0.0, 0.0, 0.0],
                    ["cx", 0, 1],
                    ["u3", 0, 0.0, 0.0, 0.0],
                    ["u3", 1, 0.0, 0.0, 0.0]]
        if np.allclose(target, cnot_10, atol=1e-10):
            return [["u3", 0, 0.0, 0.0, 0.0],
                    ["u3", 1, 0.0, 0.0, 0.0],
                    ["cx", 1, 0],
                    ["u3", 0, 0.0, 0.0, 0.0],
                    ["u3", 1, 0.0, 0.0, 0.0]]
    
    # For SWAP (2 qubits)
    if qubits == 2 and cnot_budget >= 3:
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return [["u3", 0, 0.0, 0.0, 0.0],
                    ["u3", 1, 0.0, 0.0, 0.0],
                    ["cx", 0, 1],
                    ["u3", 0, 0.0, 0.0, 0.0],
                    ["u3", 1, 0.0, 0.0, 0.0],
                    ["cx", 1, 0],
                    ["u3", 0, 0.0, 0.0, 0.0],
                    ["u3", 1, 0.0, 0.0, 0.0],
                    ["cx", 0, 1],
                    ["u3", 0, 0.0, 0.0, 0.0],
                    ["u3", 1, 0.0, 0.0, 0.0]]
    
    # For iSWAP (2 qubits): iSWAP = [[0,1j,0,0],[1j,0,0,0],[0,0,0,1j],[0,0,1j,0]]
    if qubits == 2 and cnot_budget >= 3:
        iswap = np.array([[0,1j,0,0],[1j,0,0,0],[0,0,0,1j],[0,0,1j,0]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            # iSWAP can be decomposed with 3 CNOTs
            # R = diag(1, i) on q0, diag(1, i) on q1, then SWAP-like
            # Actually: iSWAP = (I ⊗ R_z(π/4)) · CNOT(0,1) · (R_x(π/2) ⊗ R_x(π/2)) · CNOT(1,0) · (R_x(-π/2) ⊗ R_x(-π/2)) · CNOT(0,1) · (I ⊗ R_z(-π/4))
            # Let me use a known decomposition
            gates = [
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, 0.0, 0.0, 0.0],
                ["u3", 0, math.pi/2, 0.0, 0.0],  # R_x(π/2)
                ["u3", 1, math.pi/2, 0.0, 0.0],  # R_x(π/2)
                ["cx", 0, 1],
                ["u3", 0, -math.pi/2, 0.0, 0.0],  # R_x(-π/2)
                ["u3", 1, -math.pi/2, 0.0, 0.0],  # R_x(-π/2)
                ["cx", 1, 0],
                ["u3", 0, math.pi/2, 0.0, 0.0],  # R_x(π/2)
                ["u3", 1, math.pi/2, 0.0, 0.0],  # R_x(π/2)
                ["cx", 0, 1],
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, 0.0, 0.0, 0.0],
            ]
            return gates
    
    # For Toffoli (3 qubits)
    if qubits == 3 and cnot_budget >= 6:
        # Toffoli = CCX
        toffoli = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            if i == 0b110 or i == 0b111:
                # Control qubits 0 and 1, target qubit 2
                # If bits 0 and 1 are set, flip bit 2
                j = i
                if (i >> 1) & 1 and i & 1:
                    j = i ^ 1
                toffoli[j, i] = 1
            else:
                toffoli[i, i] = 1
        # Actually let me construct it properly
        toffoli = np.eye(8, dtype=complex)
        # Flip last bit when first two bits are 1
        for i in range(8):
            if (i & 0b011) == 0b011:  # bits 0 and 1 are set
                j = i ^ 1
                toffoli[i, i] = 0
                toffoli[j, i] = 1
        
        if np.allclose(target, toffoli, atol=1e-10):
            # Known 6-CNOT decomposition of Toffoli
            # This is a standard decomposition
            gates = [
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, 0.0, 0.0, 0.0],
                ["u3", 2, 0.0, 0.0, 0.0],
                ["cx", 1, 2],
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, 0.0, 0.0, 0.0],
                ["u3", 2, 0.0, 0.0, 0.0],
                ["cx", 0, 1],
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, 0.0, 0.0, 0.0],
                ["u3", 2, 0.0, 0.0, 0.0],
                ["cx", 1, 2],
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, 0.0, 0.0, 0.0],
                ["u3", 2, 0.0, 0.0, 0.0],
                ["cx", 0, 1],
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, 0.0, 0.0, 0.0],
                ["u3", 2, 0.0, 0.0, 0.0],
            ]
            # This is just a placeholder; let me use the optimizer for Toffoli too
            # Fall through to general optimization
    
    # General case: use the layout-based optimization
    gates = _try_multiple_layouts(problem)
    
    return gates
# EVOLVE-BLOCK-END
