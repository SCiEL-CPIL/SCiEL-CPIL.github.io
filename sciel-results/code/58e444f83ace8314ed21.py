"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Global time tracking
_START_TIME = time.time()
TIME_LIMIT = 240  # Safety margin for 270s wall-clock limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build the unitary from a gate list efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build the full operator by kron
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


def _synthesize_cnot():
    """CNOT gate: just one CNOT."""
    return [["cx", 0, 1]]


def _synthesize_iswap():
    """iSWAP gate: 2 CNOTs + single qubit rotations."""
    # iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
    # Standard 2-CNOT decomposition
    return [
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
        ["cx", 0, 1],
        ["u3", 0, math.pi/2, math.pi/2, 0],
        ["u3", 1, math.pi/2, math.pi/2, 0],
        ["cx", 1, 0],
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
    ]


def _synthesize_swap():
    """SWAP gate: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    """Toffoli gate: 6 CNOTs (standard decomposition)."""
    return [
        ["u3", 2, math.pi/2, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, -math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, -math.pi/2],
        ["u3", 1, math.pi/4, 0, math.pi/4],
        ["u3", 2, math.pi/4, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 1, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, 0, math.pi/2],
    ]


def _kak_decompose_2qubit(U):
    """
    KAK decomposition for 2-qubit unitary U.
    Returns (A, B, K, Bdag, Adag) such that U ~ A K B where K is the 2-qubit part.
    Actually returns gates directly.
    """
    # For 2 qubits, any unitary can be decomposed as:
    # U = (A ⊗ B) * K * (C ⊗ D)
    # where K is a canonical 2-qubit gate depending on 3 parameters.
    
    # Use the standard KAK decomposition via singular values of a specific matrix.
    # This is complex; let's use a simpler approach for the generic case.
    
    # Actually, for 2 qubits with budget >= 3, we can always synthesize exactly.
    # The standard result is that any 2-qubit unitary needs at most 3 CNOTs.
    
    # Let's implement a direct synthesis using the KAK parameters.
    # This requires computing the KAK parameters, which is non-trivial.
    
    # Alternative: use the fact that any 2-qubit unitary can be written as:
    # U = exp(i(theta1 X⊗X + theta2 Y⊗Y + theta3 Z⊗Z)) * single-qubit gates
    # We can find the optimal single-qubit gates and the 3 CNOT parameters.
    
    # For now, return None to indicate we should use the generic approach.
    return None


def _generic_synthesis_2qubit(problem):
    """
    Generic synthesis for 2-qubit unitaries using KAK decomposition.
    """
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    if budget < 1:
        # Can't do much
        return [["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0]]
    
    # For 2 qubits, we can use a parameterized circuit with up to 3 CNOTs
    # and optimize the single-qubit parameters.
    
    # Layout: for budget CNOTs, create a sequence
    # U = A0 * CX * A1 * CX * ... * An
    # where Ai are single-qubit gates on both qubits
    
    # Let's use a layout with 'budget' CNOTs alternating directions
    layout = []
    num_u3 = 0
    for i in range(budget):
        # Add single-qubit gates before CNOT
        for q in range(2):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            num_u3 += 1
        # Add CNOT
        if i % 2 == 0:
            layout.append(["cx", 0, 1])
        else:
            layout.append(["cx", 1, 0])
    # Add final single-qubit gates
    for q in range(2):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        num_u3 += 1
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    n_params = 3 * num_u3
    
    # Use Nelder-Mead or coordinate descent
    best = None
    best_dist = float('inf')
    
    max_starts = 3
    for start in range(max_starts):
        if _time_remaining() < 20:
            break
        
        if start == 0:
            values = np.zeros(n_params)
        else:
            values = np.random.uniform(-math.pi, math.pi, n_params)
        
        current_dist = _distance(target, _build_fast(circuit(values), qubits))
        if current_dist < best_dist:
            best_dist = current_dist
            best = values.copy()
        
        # Coordinate descent
        max_sweeps = 15
        for sweep in range(max_sweeps):
            if _time_remaining() < 20:
                break
            step = 0.5 / (1 + sweep * 0.2)
            improved = False
            for i in range(n_params):
                if _time_remaining() < 20:
                    break
                for direction in (1, -1):
                    trial = values[i] + direction * step
                    values[i] = trial
                    val = _distance(target, _build_fast(circuit(values), qubits))
                    if val < current_dist:
                        current_dist = val
                        improved = True
                        break
                    values[i] = trial - direction * step
                if _time_remaining() < 20:
                    break
            if not improved:
                break
        
        if current_dist < best_dist:
            best_dist = current_dist
            best = values.copy()
    
    if best is None:
        best = np.zeros(n_params)
    
    return circuit(best)


def _generic_synthesis_3qubit(problem):
    """
    Generic synthesis for 3-qubit unitaries.
    """
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # For 3 qubits, the CNOT budget is typically around 6-8
    # Use a layered approach with CNOTs and single-qubit gates
    
    # Create a layout with 'budget' CNOTs
    # Use a pattern that covers all pairs
    pairs = [(0,1), (1,2), (0,2), (2,1), (1,0), (2,0)]
    
    layout = []
    num_u3 = 0
    for i in range(budget):
        # Add single-qubit gates before CNOT
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            num_u3 += 1
        # Add CNOT
        c, t = pairs[i % len(pairs)]
        layout.append(["cx", c, t])
    # Add final single-qubit gates
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        num_u3 += 1
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    n_params = 3 * num_u3
    
    best = None
    best_dist = float('inf')
    
    max_starts = 3
    for start in range(max_starts):
        if _time_remaining() < 20:
            break
        
        if start == 0:
            values = np.zeros(n_params)
        else:
            values = np.random.uniform(-math.pi, math.pi, n_params)
        
        current_dist = _distance(target, _build_fast(circuit(values), qubits))
        if current_dist < best_dist:
            best_dist = current_dist
            best = values.copy()
        
        # Coordinate descent
        max_sweeps = 10
        for sweep in range(max_sweeps):
            if _time_remaining() < 20:
                break
            step = 0.5 / (1 + sweep * 0.2)
            improved = False
            for i in range(n_params):
                if _time_remaining() < 20:
                    break
                for direction in (1, -1):
                    trial = values[i] + direction * step
                    values[i] = trial
                    val = _distance(target, _build_fast(circuit(values), qubits))
                    if val < current_dist:
                        current_dist = val
                        improved = True
                        break
                    values[i] = trial - direction * step
                if _time_remaining() < 20:
                    break
            if not improved:
                break
        
        if current_dist < best_dist:
            best_dist = current_dist
            best = values.copy()
    
    if best is None:
        best = np.zeros(n_params)
    
    return circuit(best)


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Try to use known exact syntheses first
    if name == "CNOT" and qubits == 2:
        gates = _synthesize_cnot()
    elif name == "iSWAP" and qubits == 2:
        gates = _synthesize_iswap()
    elif name == "SWAP" and qubits == 2:
        gates = _synthesize_swap()
    elif name == "Toffoli" and qubits == 3:
        gates = _synthesize_toffoli()
    else:
        # Generic synthesis
        if qubits == 2:
            gates = _generic_synthesis_2qubit(problem)
        else:
            gates = _generic_synthesis_3qubit(problem)
    
    # Verify the result
    target = _target(problem)
    matrix = _build_fast(gates, qubits)
    dist = _distance(target, matrix)
    
    # If the known synthesis didn't work well, fall back to generic
    if dist > 0.01:
        if qubits == 2:
            gates = _generic_synthesis_2qubit(problem)
        else:
            gates = _generic_synthesis_3qubit(problem)
    
    return gates
# EVOLVE-BLOCK-END
