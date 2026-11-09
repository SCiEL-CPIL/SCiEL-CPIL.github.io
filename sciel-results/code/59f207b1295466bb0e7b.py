import math
import time
import numpy as np

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 230.0  # Safety margin from 270s kill

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

def _u3(theta, phi, lam):
    return ["u3", 0, theta, phi, lam] # Placeholder, qubit index will be set

def _apply_u3(gates, qubit, theta, phi, lam):
    gates.append(["u3", qubit, theta, phi, lam])

def _cx(gates, control, target):
    gates.append(["cx", control, target])

def _decompose_2qubit_kak(U):
    """
    Decompose a 2-qubit unitary U into a circuit with 3 CNOTs.
    Returns a list of gates for qubits 0 and 1.
    """
    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Compute the KAK parameters
    # U = e^{i alpha} (A kron B) exp(i (b1 X⊗X + b2 Y⊗Y + b3 Z⊗Z)) (C kron D)
    
    # We use the standard algorithm to extract A, B, C, D and b1, b2, b3.
    # Reference: "Synthesis of Two-Qubit Quantum Gates" by Shende et al.
    
    # Step 1: Compute U_eff = U * (exp(-i alpha) I)
    # The global phase alpha is such that det(U_eff) = 1.
    det_U = np.linalg.det(U)
    alpha = -0.5j * np.log(det_U)
    U_eff = U * np.exp(-alpha)
    
    # Step 2: Compute the "canonical form" parameters
    # Let M = U_eff
    # We define:
    # a1 = 0.5 * Tr(M (X⊗X))
    # a2 = 0.5 * Tr(M (Y⊗Y))
    # a3 = 0.5 * Tr(M (Z⊗Z))
    # But this is not quite right. The correct way is to use the basis of su(2)⊗su(2).
    
    # Let's use a more robust numerical method or a known decomposition.
    # For simplicity and robustness, we can use the fact that any 2-qubit gate can be written as:
    # U = (A1 ⊗ A2) CNOT (B1 ⊗ B2) CNOT (C1 ⊗ C2) CNOT (D1 ⊗ D2)
    
    # This is a standard 3-CNOT decomposition.
    # We can find A, B, C, D by solving a system of equations or using optimization.
    # However, there is an analytic solution using the KAK decomposition.
    
    # Let's implement the KAK decomposition properly.
    # U = e^{i\alpha} U_1 (U_2 ⊗ U_3) e^{i(b_1 \sigma_x \otimes \sigma_x + b_2 \sigma_y \otimes \sigma_y + b_3 \sigma_z \otimes \sigma_z)} (V_2 ⊗ V_3)
    
    # We will use the method from "Efficient Quantum Circuits for Arbitrary Two-Qubit Operations"
    
    # Define Pauli matrices
    sigma_x = X
    sigma_y = Y
    sigma_z = Z
    
    # Compute the coefficients
    # The local invariants are:
    # K1 = 0.5 * Tr(U (X⊗X))
    # K2 = 0.5 * Tr(U (Y⊗Y))
    # K3 = 0.5 * Tr(U (Z⊗Z))
    
    K1 = 0.5 * np.trace(U_eff @ np.kron(X, X))
    K2 = 0.5 * np.trace(U_eff @ np.kron(Y, Y))
    K3 = 0.5 * np.trace(U_eff @ np.kron(Z, Z))
    
    # The angles b1, b2, b3 are related to K1, K2, K3 by:
    # b1 = arccos(K1) / 2 ? No.
    # Actually, the canonical form is:
    # exp(i (b1 X⊗X + b2 Y⊗Y + b3 Z⊗Z))
    # The eigenvalues of this operator are exp(±i(b1+b2+b3)), exp(±i(b1+b2-b3)), ...
    
    # A simpler approach for exact synthesis within budget:
    # Use the known decomposition for SWAP, iSWAP, CNOT.
    # For generic 2-qubit, we can use a 3-CNOT decomposition.
    
    # Let's try to match the target to known gates first.
    # If it's a generic gate, we can use a numerical optimizer to find the single-qubit rotations
    # for a fixed 3-CNOT topology.
    
    # Topology: U3(0) U3(1) CX(0,1) U3(0) U3(1) CX(0,1) U3(0) U3(1) CX(0,1) U3(0) U3(1)
    
    # We will use a simple coordinate descent or Nelder-Mead to find the 15 parameters (5 u3 per qubit * 3 layers? No, 2 u3 per qubit per layer + 1 u3 at end? 
    # Standard 3-CNOT decomposition has 12 u3 gates (2 per qubit per CNOT layer) + 2 final u3? 
    # Actually, the standard decomposition uses 12 single-qubit gates.
    
    # Let's define the circuit structure:
    # [U3(0), U3(1), CX(0,1), U3(0), U3(1), CX(0,1), U3(0), U3(1), CX(0,1), U3(0), U3(1)]
    
    # Parameters: 12 angles (theta, phi, lam for 4 pairs of qubits? No, 2 qubits * 3 layers * 2 params? 
    # u3 has 3 params. 2 qubits * 3 layers = 6 u3 gates * 3 params = 18 params. Plus 2 final u3? 
    # Let's just use 12 u3 gates: 2 per layer (one for each qubit) for 3 layers, plus 2 final? 
    # Actually, the minimal 3-CNOT decomposition uses 12 u3 gates.
    
    # We will optimize the 12 u3 gates.
    
    def circuit_from_params(params):
        gates = []
        idx = 0
        for layer in range(3):
            _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
            _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
            idx += 6
            if layer < 2:
                _cx(gates, 0, 1)
        # Wait, the structure is U3 U3 CX U3 U3 CX U3 U3 CX U3 U3
        # So 3 CX gates.
        # Let's rebuild:
        gates = []
        idx = 0
        # Layer 1
        _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
        _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
        idx += 6
        _cx(gates, 0, 1)
        # Layer 2
        _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
        _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
        idx += 6
        _cx(gates, 0, 1)
        # Layer 3
        _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
        _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
        idx += 6
        _cx(gates, 0, 1)
        # Final
        _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
        _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
        idx += 6
        return gates

    # Initial guess: zeros
    best_params = [0.0] * 18
    best_dist = 1.0
    
    # Optimization
    # We have 270s total. We should spend most of it on the hardest gates.
    # If this is a generic 2-qubit gate, we have plenty of time.
    
    # Use Nelder-Mead or simple coordinate descent.
    # Given the time, coordinate descent is safer.
    
    def eval_dist(params):
        gates = circuit_from_params(params)
        U_calc = _build(gates, 2)
        return _distance(U, U_calc)
    
    best_dist = eval_dist(best_params)
    
    # Coordinate descent
    step = 0.5
    for sweep in range(50):
        step = 0.5 / (1 + sweep * 0.1)
        improved = False
        for i in range(18):
            for trial in (best_params[i] + step, best_params[i] - step):
                new_params = best_params[:]
                new_params[i] = trial
                d = eval_dist(new_params)
                if d < best_dist:
                    best_params = new_params
                    best_dist = d
                    improved = True
        if not improved:
            break
            
    gates = circuit_from_params(best_params)
    return gates

def _decompose_swap():
    gates = []
    _cx(gates, 0, 1)
    _cx(gates, 1, 0)
    _cx(gates, 0, 1)
    return gates

def _decompose_iswap():
    gates = []
    # iSWAP = 1/sqrt(2) * [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    # Standard decomposition:
    _apply_u3(gates, 0, math.pi/2, 0, math.pi/2)
    _apply_u3(gates, 1, math.pi/2, 0, math.pi/2)
    _cx(gates, 0, 1)
    _apply_u3(gates, 0, math.pi/2, -math.pi/2, 0)
    _apply_u3(gates, 1, math.pi/2, math.pi/2, 0)
    _cx(gates, 0, 1)
    _apply_u3(gates, 0, math.pi/2, -math.pi/2, 0)
    _apply_u3(gates, 1, math.pi/2, math.pi/2, 0)
    _cx(gates, 0, 1)
    _apply_u3(gates, 0, math.pi/2, 0, math.pi/2)
    _apply_u3(gates, 1, math.pi/2, 0, math.pi/2)
    return gates

def _decompose_toffoli():
    gates = []
    # Toffoli (CCX) on qubits 0, 1, 2 (0 is MSB, so 0 is control1, 1 is control2, 2 is target)
    # Standard 7-CNOT decomposition
    _apply_u3(gates, 2, math.pi/4, 0, 0)
    _cx(gates, 1, 2)
    _apply_u3(gates, 2, -math.pi/4, 0, 0)
    _cx(gates, 0, 2)
    _apply_u3(gates, 2, math.pi/4, 0, 0)
    _cx(gates, 1, 2)
    _apply_u3(gates, 2, -math.pi/4, 0, 0)
    _apply_u3(gates, 1, math.pi/2, 0, 0)
    _apply_u3(gates, 1, 0, 0, math.pi/2)
    _cx(gates, 0, 1)
    _apply_u3(gates, 1, math.pi/2, 0, 0)
    _apply_u3(gates, 1, 0, 0, -math.pi/2)
    _cx(gates, 0, 1)
    _apply_u3(gates, 0, math.pi/2, 0, 0)
    _apply_u3(gates, 0, 0, 0, math.pi/2)
    _cx(gates, 1, 0)
    _apply_u3(gates, 1, math.pi/2, 0, 0)
    _apply_u3(gates, 1, 0, 0, -math.pi/2)
    _cx(gates, 1, 0)
    _apply_u3(gates, 1, math.pi/2, 0, 0)
    _apply_u3(gates, 1, 0, 0, math.pi/2)
    return gates

def _identify_gate(target, qubits):
    """
    Try to identify if the target is a known gate.
    Returns a string identifier or None.
    """
    if qubits == 2:
        # Check CNOT
        cnot = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot):
            return "cnot"
        # Check SWAP
        swap = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=complex)
        if np.allclose(target, swap):
            return "swap"
        # Check iSWAP
        iswap = 1/np.sqrt(2) * np.array([[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)
        if np.allclose(target, iswap):
            return "iswap"
    elif qubits == 3:
        # Check Toffoli
        toffoli = np.zeros((8,8), dtype=complex)
        for i in range(8):
            toffoli[i,i] = 1
        toffoli[6,7] = 0
        toffoli[7,6] = 1
        toffoli[6,6] = 0
        toffoli[7,7] = 0
        # Actually, Toffoli flips |110> to |111> and vice versa.
        # Index 6 is 110, Index 7 is 111.
        toffoli = np.eye(8, dtype=complex)
        toffoli[6,7] = 1
        toffoli[7,6] = 1
        toffoli[6,6] = 0
        toffoli[7,7] = 0
        if np.allclose(target, toffoli):
            return "toffoli"
            
    return None

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Identify known gates
    gate_type = _identify_gate(target, qubits)
    
    if gate_type == "cnot":
        return [["cx", 0, 1]]
    elif gate_type == "swap":
        gates = _decompose_swap()
        return gates
    elif gate_type == "iswap":
        gates = _decompose_iswap()
        return gates
    elif gate_type == "toffoli":
        gates = _decompose_toffoli()
        return gates
        
    # If not a known gate, use generic decomposition
    if qubits == 2:
        return _decompose_2qubit_kak(target)
    elif qubits == 3:
        # Check if it's a 2-qubit gate on a subset
        # Try pairs (0,1), (0,2), (1,2)
        # If the unitary is block diagonal with respect to the 3rd qubit, we can decompose it.
        
        # Check pair (0,1)
        # If U is of the form I ⊗ U2, then U[0:4, 0:4] == U[4:8, 4:8]
        # And U[0:4, 4:8] == 0
        if np.allclose(target[0:4, 4:8], 0) and np.allclose(target[4:8, 0:4], 0):
            if np.allclose(target[0:4, 0:4], target[4:8, 4:8]):
                # It acts on qubits 0 and 1
                u2 = target[0:4, 0:4]
                gates_2q = _decompose_2qubit_kak(u2)
                # Map gates to 3-qubit circuit
                # The gates_2q are for qubits 0 and 1.
                # We need to ensure they are applied to qubits 0 and 1 in the 3-qubit system.
                # The _decompose_2qubit_kak returns gates with qubit indices 0 and 1.
                # This is correct for the 3-qubit system if we just return them.
                return gates_2q
                
        # Check pair (0,2)
        # This is harder to check directly without reordering.
        # For now, assume if it's not (0,1), it might be (0,2) or (1,2) or generic.
        # Given the time, we will fallback to a generic optimizer for 3-qubit if needed.
        # But let's try a simple heuristic: if it's not (0,1), try (1,2)
        # Reorder basis: |q0 q1 q2> -> |q0 q2 q1> ?
        
        # For simplicity and robustness, if it's not a known gate or (0,1) 2-qubit gate,
        # we will use a generic 3-qubit optimizer.
        
        # Generic 3-qubit optimizer
        # Structure: Layers of U3 and CX.
        # Budget is likely 7 or 9.
        # Let's use a 7-CNOT ladder.
        
        def circuit_3q(params):
            gates = []
            idx = 0
            # 3 layers of CX
            # Layer 1: CX(0,1)
            _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
            _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
            _apply_u3(gates, 2, params[idx+6], params[idx+7], params[idx+8])
            idx += 9
            _cx(gates, 0, 1)
            
            # Layer 2: CX(1,2)
            _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
            _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
            _apply_u3(gates, 2, params[idx+6], params[idx+7], params[idx+8])
            idx += 9
            _cx(gates, 1, 2)
            
            # Layer 3: CX(0,1)
            _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
            _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
            _apply_u3(gates, 2, params[idx+6], params[idx+7], params[idx+8])
            idx += 9
            _cx(gates, 0, 1)
            
            # Final
            _apply_u3(gates, 0, params[idx], params[idx+1], params[idx+2])
            _apply_u3(gates, 1, params[idx+3], params[idx+4], params[idx+5])
            _apply_u3(gates, 2, params[idx+6], params[idx+7], params[idx+8])
            idx += 9
            return gates

        best_params = [0.0] * 36
        best_dist = 1.0
        
        def eval_dist_3q(params):
            gates = circuit_3q(params)
            U_calc = _build(gates, 3)
            return _distance(target, U_calc)
            
        best_dist = eval_dist_3q(best_params)
        
        # Coordinate descent
        step = 0.5
        for sweep in range(50):
            if _remaining_time() < 10:
                break
            step = 0.5 / (1 + sweep * 0.1)
            improved = False
            for i in range(36):
                for trial in (best_params[i] + step, best_params[i] - step):
                    new_params = best_params[:]
                    new_params[i] = trial
                    d = eval_dist_3q(new_params)
                    if d < best_dist:
                        best_params = new_params
                        best_dist = d
                        improved = True
            if not improved:
                break
                
        return circuit_3q(best_params)

# EVOLVE-BLOCK-END
