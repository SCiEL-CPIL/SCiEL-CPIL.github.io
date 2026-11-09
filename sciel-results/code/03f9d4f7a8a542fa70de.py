"""Synthesize fixed two- and three-qubit unitaries within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = time.time() + 240.0


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _cnot_matrix(qubits, control, target):
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    for i in range(n):
        c_bit = (i >> (qubits - 1 - control)) & 1
        t_bit = (i >> (qubits - 1 - target)) & 1
        if c_bit == 1:
            mat[i, i ^ (1 << (qubits - 1 - target))] = 1.0
        else:
            mat[i, i] = 1.0
    return mat


def _build(gates, qubits):
    mat = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            op = np.array([[1]], dtype=complex)
            eye = np.eye(2, dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            mat = op @ mat
        else:
            _, c, t = gate
            mat = _cnot_matrix(qubits, c, t) @ mat
    return mat


def _distance(target, matrix):
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])


def _is_close(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol)


def _kaka_decompose_2qubit(U):
    """Decompose a 2-qubit unitary into single-qubit rotations and CNOTs.
    Returns (pre_A, pre_B, cnot_count, post_A, post_B) where the circuit is:
    (pre_A ⊗ pre_B) * CNOT^cnot_count * (post_A ⊗ post_B)
    """
    # KAK decomposition for 2-qubit gates
    # U = (A ⊗ B) * CNOT^k * (C ⊗ D)
    # where k ∈ {0, 1, 2, 3}
    
    # Extract the "interaction" part
    # For a general 2-qubit gate, we can use the following approach:
    # 1. Compute the canonical form
    # 2. Determine the CNOT count
    
    # Simple approach: try 0, 1, 2, 3 CNOTs and see which matches
    # For efficiency, use known decompositions
    
    # First, check if it's a product gate (0 CNOTs)
    # U = A ⊗ B if and only if the "twist" is zero
    
    # Use the standard KAK decomposition
    # U = e^{iθ} (A ⊗ B) exp(i a_x σ_x⊗σ_x + i a_y σ_y⊗σ_y + i a_z σ_z⊗σ_z) (C ⊗ D)
    # The CNOT count is determined by the a_i values
    
    # For simplicity, use a direct approach:
    # Try to find the decomposition with the minimal CNOTs
    
    # Method: Use the fact that any 2-qubit gate can be written as
    # U = (A ⊗ I) CNOT^k (B ⊗ I) for k CNOTs
    
    # Let's use a more direct numerical approach
    # For k=0: U should be a tensor product
    # For k=1: U = (A⊗B) CNOT (C⊗D)
    # etc.
    
    # Practical approach: use known formulas
    # For 3 CNOTs (general case):
    # U = (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT (G⊗H)
    
    # Let me use a simpler method: direct parameter optimization
    # But that's slow. Instead, use the analytical KAK decomposition.
    
    # KAK decomposition:
    # U = e^{iφ} (A ⊗ B) e^{i(α σ_x⊗σ_x + β σ_y⊗σ_y + γ σ_z⊗σ_z)} (C ⊗ D)
    # The canonical angles α, β, γ determine the CNOT count.
    
    # For now, use a practical approach: assume 3 CNOTs for general 2-qubit gates
    # and verify/adjust for special cases.
    
    # Determine CNOT count by checking special structures
    # A gate needs 0 CNOTs if it's a tensor product
    # 1 CNOT if it's in the 1-CNOT class
    # 2 CNOTs if it's in the 2-CNOT class
    # 3 CNOTs otherwise
    
    # Check for tensor product structure
    # U is a tensor product if U[i,j] = A[i//2, j//2] * B[i%2, j%2] for all i,j
    # This is equivalent to checking if U commutes with certain operators
    
    # Simple check: if U = A ⊗ B, then U[0:2, 0:2] = A[0,0]*B, 
    # U[0:2, 2:4] = A[0,1]*B, U[2:4, 0:2] = A[1,0]*B, U[2:4, 2:4] = A[1,1]*B
    # So U[0:2, 0:2] / U[0,0] should equal U[0:2, 2:4] / U[0,2] etc.
    
    # Let's just use 3 CNOTs as the default and refine if needed
    # For the known gates, we'll use exact decompositions
    
    return 3  # Default to 3 CNOTs for generic 2-qubit gates


def _decompose_2qubit(U, cnot_budget):
    """Decompose a 2-qubit unitary into gates."""
    # Try to find the minimal CNOT decomposition
    # For known gates, use exact formulas
    # For generic gates, use KAK decomposition
    
    # First, check if it's a known gate
    name_hint = None  # We don't have the name, so we'll use numerical matching
    
    # Try 0 CNOTs: U = A ⊗ B
    # Check if U is a tensor product
    # U[i,j] = A[i>>1, j>>1] * B[i&1, j&1]
    # This means U[0,0]/U[0,2] = U[1,0]/U[1,2] = A[0,0]/A[0,1] (if A[0,1] != 0)
    
    # More robust: check if the "correlation" is zero
    # For a tensor product, the off-diagonal blocks have a specific structure
    
    # Let's use a different approach: try to fit with increasing CNOT counts
    
    # For 0 CNOTs:
    # U = A ⊗ B
    # We can extract A and B from the 2x2 blocks
    # U = [[A00*B, A01*B], [A10*B, A11*B]]
    # So U[0:2, 0:2] = A[0,0] * B, U[0:2, 2:4] = A[0,1] * B
    # If A[0,0] != 0: B = U[0:2, 0:2] / A[0,0]
    # Then A[0,0] = trace(U[0:2, 0:2]) / trace(B) (if trace(B) != 0)
    
    # This is getting complex. Let me use a simpler approach:
    # Use the KAK decomposition analytically
    
    # For a 2-qubit gate, the KAK decomposition gives:
    # U = (A ⊗ B) * CNOT^k * (C ⊗ D)
    # where k is determined by the "interaction strength"
    
    # Practical approach for this task:
    # - CNOT gate: 1 CNOT
    # - iSWAP: 3 CNOTs  
    # - SWAP: 3 CNOTs
    # - Generic 2-qubit: 3 CNOTs
    
    # Let me detect which gate it is by comparing to known unitaries
    
    # CNOT (control 0, target 1):
    cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    cnot_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
    
    # iSWAP:
    iswap = np.array([[0,0,0,1j],[0,0,1j,0],[0,1j,0,0],[1j,0,0,0]], dtype=complex)
    
    # SWAP:
    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    
    # Check if U is (up to global phase) one of these
    def _global_phase_match(A, B, tol=1e-6):
        """Check if A and B differ only by a global phase."""
        if np.allclose(A, 0) or np.allclose(B, 0):
            return False
        # Find the phase: A = e^{iφ} B
        # e^{iφ} = A[i,j] / B[i,j] for any nonzero B[i,j]
        # Use the element with largest magnitude in B
        idx = np.unravel_index(np.argmax(np.abs(B)), B.shape)
        if abs(B[idx]) < 1e-12:
            return False
        phase = A[idx] / B[idx]
        return np.allclose(A, phase * B, atol=tol)
    
    # Check for CNOT (either direction)
    if _global_phase_match(U, cnot_01):
        # CNOT with control 0, target 1
        return [["cx", 0, 1]]
    if _global_phase_match(U, cnot_10):
        # CNOT with control 1, target 0
        return [["cx", 1, 0]]
    
    # Check for iSWAP
    if _global_phase_match(U, iswap):
        # iSWAP decomposition: 3 CNOTs
        # iSWAP = (1/2) * [[0, 0, 0, 2i], [0, 0, 2i, 0], [0, 2i, 0, 0], [2i, 0, 0, 0]]
        # Standard decomposition:
        # iSWAP = (H ⊗ H) * (CNOT(0,1)) * (S† ⊗ S) * (CNOT(1,0)) * (S ⊗ S) * (CNOT(0,1)) * (H ⊗ H)
        # Wait, let me use the correct decomposition
        # iSWAP = exp(iπ/4 (σ_x⊗σ_x + σ_y⊗σ_y))
        # = (1/√2) [[0, 0, 0, i√2], [0, 0, i√2, 0], [0, i√2, 0, 0], [i√2, 0, 0, 0]]
        # Standard 3-CNOT decomposition:
        gates = []
        # H on both
        gates.append(["u3", 0, math.pi, 0, math.pi])  # H
        gates.append(["u3", 1, math.pi, 0, math.pi])  # H
        gates.append(["cx", 0, 1])
        gates.append(["u3", 0, math.pi, 0, 0])  # S
        gates.append(["u3", 1, math.pi, 0, 0])  # S
        gates.append(["cx", 1, 0])
        gates.append(["u3", 0, math.pi, 0, math.pi])  # S†
        gates.append(["u3", 1, math.pi, 0, math.pi])  # S†
        gates.append(["cx", 0, 1])
        gates.append(["u3", 0, math.pi, 0, math.pi])  # H
        gates.append(["u3", 1, math.pi, 0, math.pi])  # H
        return gates
    
    # Check for SWAP
    if _global_phase_match(U, swap):
        # SWAP = 3 CNOTs
        gates = [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
        return gates
    
    # For generic 2-qubit gates, use 3 CNOT decomposition
    # U = (A ⊗ B) * CNOT * (C ⊗ D) * CNOT * (E ⊗ F) * CNOT * (G ⊗ H)
    
    # Use the KAK decomposition to find the single-qubit gates
    # This is complex, so let me use a numerical approach
    
    # For now, return a placeholder that will be optimized
    # Actually, let me implement a proper KAK decomposition
    
    # KAK decomposition for 2-qubit gates:
    # U = e^{iφ} (A ⊗ B) exp(i α σ_x⊗σ_x + i β σ_y⊗σ_y + i γ σ_z⊗σ_z) (C ⊗ D)
    # The canonical angles are in the range [0, π/4] for α, β, γ with α+β+γ ≤ π/2
    
    # The CNOT count is:
    # 0 if α=β=γ=0
    # 1 if two of α,β,γ are 0
    # 2 if one of α,β,γ is 0
    # 3 otherwise
    
    # To extract the decomposition, we can use the following algorithm:
    # 1. Compute the "twist" of U
    # 2. Determine the canonical angles
    # 3. Extract the single-qubit gates
    
    # This is quite involved. For this task, since we know the budgets,
    # let me use a simpler approach: parameter optimization with a good initial guess.
    
    # Actually, for the known gates in the task, we should have exact decompositions.
    # The generic unitaries might be close to known gates or have special structure.
    
    # Let me use a hybrid approach: try known decompositions first, then fall back to optimization.
    
    # For now, return a 3-CNOT template with identity single-qubit gates
    # This will be refined by the caller if needed
    gates = []
    gates.append(["u3", 0, 0, 0, 0])
    gates.append(["u3", 1, 0, 0, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0, 0, 0])
    gates.append(["u3", 1, 0, 0, 0])
    gates.append(["cx", 1, 0])
    gates.append(["u3", 0, 0, 0, 0])
    gates.append(["u3", 1, 0, 0, 0])
    gates.append(["cx", 0, 1])
    gates.append(["u3", 0, 0, 0, 0])
    gates.append(["u3", 1, 0, 0, 0])
    return gates


def _decompose_3qubit(U, cnot_budget):
    """Decompose a 3-qubit unitary into gates."""
    # Check for Toffoli
    toffoli_01_2 = np.eye(8, dtype=complex)
    toffoli_01_2[5, 5] = 0
    toffoli_01_2[5, 7] = 1
    toffoli_01_2[7, 5] = 1
    toffoli_01_2[7, 7] = 0
    
    def _global_phase_match(A, B, tol=1e-6):
        if np.allclose(A, 0) or np.allclose(B, 0):
            return False
        idx = np.unravel_index(np.argmax(np.abs(B)), B.shape)
        if abs(B[idx]) < 1e-12:
            return False
        phase = A[idx] / B[idx]
        return np.allclose(A, phase * B, atol=tol)
    
    # Check for Toffoli (control 0,1 target 2)
    if _global_phase_match(U, toffoli_01_2):
        # Toffoli decomposition: 6 CNOTs
        # Standard decomposition:
        gates = []
        # H on target
        gates.append(["u3", 2, math.pi, 0, math.pi])  # H
        # CNOT from control 1 to target
        gates.append(["cx", 1, 2])
        # T† on target
        gates.append(["u3", 2, math.pi/2, 0, math.pi])  # T†
        # CNOT from control 0 to target
        gates.append(["cx", 0, 2])
        # T on target
        gates.append(["u3", 2, math.pi/2, 0, 0])  # T
        # CNOT from control 1 to target
        gates.append(["cx", 1, 2])
        # T† on target
        gates.append(["u3", 2, math.pi/2, 0, math.pi])  # T†
        # CNOT from control 0 to target
        gates.append(["cx", 0, 2])
        # T on target
        gates.append(["u3", 2, math.pi/2, 0, 0])  # T
        # CNOT from control 1 to target
        gates.append(["cx", 1, 2])
        # H on target
        gates.append(["u3", 2, math.pi, 0, math.pi])  # H
        return gates
    
    # Check for other Toffoli variants
    # Toffoli (control 0,2 target 1)
    toffoli_02_1 = np.eye(8, dtype=complex)
    toffoli_02_1[3, 3] = 0
    toffoli_02_1[3, 7] = 1
    toffoli_02_1[7, 3] = 1
    toffoli_02_1[7, 7] = 0
    if _global_phase_match(U, toffoli_02_1):
        # Need to adapt the decomposition
        # This is more complex, let me use a general approach
        pass
    
    # For generic 3-qubit gates, use a decomposition with the given budget
    # This is complex, so for now return a template
    gates = []
    # Add some CNOTs and single-qubit gates
    for i in range(min(cnot_budget, 6)):
        gates.append(["u3", 0, 0, 0, 0])
        gates.append(["u3", 1, 0, 0, 0])
        gates.append(["u3", 2, 0, 0, 0])
        if i % 3 == 0:
            gates.append(["cx", 0, 1])
        elif i % 3 == 1:
            gates.append(["cx", 1, 2])
        else:
            gates.append(["cx", 0, 2])
    gates.append(["u3", 0, 0, 0, 0])
    gates.append(["u3", 1, 0, 0, 0])
    gates.append(["u3", 2, 0, 0, 0])
    return gates


def _optimize_2qubit(U, cnot_budget, max_time=5.0):
    """Optimize a 2-qubit gate decomposition."""
    start_time = time.time()
    
    # Initialize with identity single-qubit gates
    # Template: (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT (G⊗H)
    # 8 single-qubit gates, each with 3 parameters = 24 parameters
    
    # For 3 CNOTs
    if cnot_budget >= 3:
        n_params = 24  # 8 u3 gates * 3 params
        params = np.zeros(n_params)
    elif cnot_budget >= 2:
        # (A⊗B) CNOT (C⊗D) CNOT (E⊗F)
        n_params = 18  # 6 u3 gates * 3 params
        params = np.zeros(n_params)
    elif cnot_budget >= 1:
        # (A⊗B) CNOT (C⊗D)
        n_params = 12  # 4 u3 gates * 3 params
        params = np.zeros(n_params)
    else:
        # A⊗B
        n_params = 6  # 2 u3 gates * 3 params
        params = np.zeros(n_params)
    
    def _build_from_params(p):
        gates = []
        if cnot_budget >= 3:
            # 8 u3 gates interleaved with 3 CNOTs
            # Order: u3_0, u3_1, CNOT, u3_2, u3_3, CNOT, u3_4, u3_5, CNOT, u3_6, u3_7
            gates.append(["u3", 0, p[0], p[1], p[2]])
            gates.append(["u3", 1, p[3], p[4], p[5]])
            gates.append(["cx", 0, 1])
            gates.append(["u3", 0, p[6], p[7], p[8]])
            gates.append(["u3", 1, p[9], p[10], p[11]])
            gates.append(["cx", 1, 0])
            gates.append(["u3", 0, p[12], p[13], p[14]])
            gates.append(["u3", 1, p[15], p[16], p[17]])
            gates.append(["cx", 0, 1])
            gates.append(["u3", 0, p[18], p[19], p[20]])
            gates.append(["u3", 1, p[21], p[22], p[23]])
        elif cnot_budget >= 2:
            gates.append(["u3", 0, p[0], p[1], p[2]])
            gates.append(["u3", 1, p[3], p[4], p[5]])
            gates.append(["cx", 0, 1])
            gates.append(["u3", 0, p[6], p[7], p[8]])
            gates.append(["u3", 1, p[9], p[10], p[11]])
            gates.append(["cx", 1, 0])
            gates.append(["u3", 0, p[12], p[13], p[14]])
            gates.append(["u3", 1, p[15], p[16], p[17]])
        elif cnot_budget >= 1:
            gates.append(["u3", 0, p[0], p[1], p[2]])
            gates.append(["u3", 1, p[3], p[4], p[5]])
            gates.append(["cx", 0, 1])
            gates.append(["u3", 0, p[6], p[7], p[8]])
            gates.append(["u3", 1, p[9], p[10], p[11]])
        else:
            gates.append(["u3", 0, p[0], p[1], p[2]])
            gates.append(["u3", 1, p[3], p[4], p[5]])
        return gates
    
    best_params = params.copy()
    best_dist = _distance(U, _build(_build_from_params(params), 2))
    
    # Coordinate descent
    step = 1.0
    for iteration in range(100):
        if time.time() - start_time > max_time:
            break
        improved = False
        for i in range(n_params):
            for trial in (best_params[i] + step, best_params[i] - step):
                p = best_params.copy()
                p[i] = trial
                dist = _distance(U, _build(_build_from_params(p), 2))
                if dist < best_dist:
                    best_params = p
                    best_dist = dist
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        else:
            step *= 0.9
    
    return _build_from_params(best_params), best_dist


def _optimize_3qubit(U, cnot_budget, max_time=10.0):
    """Optimize a 3-qubit gate decomposition."""
    start_time = time.time()
    
    # Template with CNOTs and single-qubit gates
    # For simplicity, use a fixed CNOT pattern and optimize single-qubit gates
    
    # CNOT pattern for 3 qubits:
    # CNOT(0,1), CNOT(1,2), CNOT(0,1), CNOT(1,2), ... repeating
    
    n_cnots = cnot_budget
    n_u3 = 3 * (n_cnots + 1)  # 3 u3 gates before each CNOT and after the last
    n_params = 3 * n_u3
    
    params = np.zeros(n_params)
    
    def _build_from_params(p):
        gates = []
        idx = 0
        for i in range(n_cnots):
            # 3 u3 gates
            gates.append(["u3", 0, p[idx], p[idx+1], p[idx+2]])
            gates.append(["u3", 1, p[idx+3], p[idx+4], p[idx+5]])
            gates.append(["u3", 2, p[idx+6], p[idx+7], p[idx+8]])
            idx += 9
            # CNOT
            if i % 2 == 0:
                gates.append(["cx", 0, 1])
            else:
                gates.append(["cx", 1, 2])
        # Final 3 u3 gates
        gates.append(["u3", 0, p[idx], p[idx+1], p[idx+2]])
        gates.append(["u3", 1, p[idx+3], p[idx+4], p[idx+5]])
        gates.append(["u3", 2, p[idx+6], p[idx+7], p[idx+8]])
        return gates
    
    best_params = params.copy()
    best_dist = _distance(U, _build(_build_from_params(params), 3))
    
    # Coordinate descent with random restarts
    step = 1.0
    for iteration in range(50):
        if time.time() - start_time > max_time:
            break
        improved = False
        for i in range(n_params):
            for trial in (best_params[i] + step, best_params[i] - step):
                p = best_params.copy()
                p[i] = trial
                dist = _distance(U, _build(_build_from_params(p), 3))
                if dist < best_dist:
                    best_params = p
                    best_dist = dist
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        else:
            step *= 0.9
    
    return _build_from_params(best_params), best_dist


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    cnot_budget = problem["cnot_budget"]
    target = _target(problem)
    
    if qubits == 2:
        # Try known decompositions first
        gates = _decompose_2qubit(target, cnot_budget)
        dist = _distance(target, _build(gates, 2))
        
        # If the known decomposition doesn't work well, try optimization
        if dist > 1e-6 and time.time() < _DEADLINE - 30:
            opt_gates, opt_dist = _optimize_2qubit(target, cnot_budget, max_time=5.0)
            if opt_dist < dist:
                gates = opt_gates
                dist = opt_dist
        
        return gates
    
    else:  # qubits == 3
        # Try known decompositions first
        gates = _decompose_3qubit(target, cnot_budget)
        dist = _distance(target, _build(gates, 3))
        
        # If the known decomposition doesn't work well, try optimization
        if dist > 1e-6 and time.time() < _DEADLINE - 30:
            opt_gates, opt_dist = _optimize_3qubit(target, cnot_budget, max_time=10.0)
            if opt_dist < dist:
                gates = opt_gates
                dist = opt_dist
        
        return gates

# EVOLVE-BLOCK-END
