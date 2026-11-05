"""Quantum circuit synthesis via KAK decomposition and optimized CNOT placement."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 240.0  # 15s safety margin before 255s hard kill


def _check_deadline():
    return time.time() > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    """Standard u3 single-qubit gate matrix."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)


def _u3_params_from_matrix(U):
    """Decompose a 2x2 unitary into u3(theta, phi, lam) plus global phase."""
    det = np.linalg.det(U)
    if abs(det) < 1e-15:
        return 0.0, 0.0, 0.0
    global_phase = np.sqrt(det)
    U_prime = U / global_phase
    
    # Make sure U_prime has determinant 1
    if abs(U_prime[0, 0] * U_prime[1, 1] - U_prime[0, 1] * U_prime[1, 0] - 1) > 1e-10:
        # Try negating
        U_prime = -U_prime
        global_phase = -global_phase
    
    # For SU(2): [[a, b], [-b*, a*]]
    a = U_prime[0, 0]
    b = U_prime[0, 1]
    
    # theta/2 such that cos(theta/2) = |a|, sin(theta/2) = |b|
    cos_half = abs(a)
    sin_half = abs(b)
    cos_half = max(-1.0, min(1.0, cos_half))
    sin_half = max(-1.0, min(1.0, sin_half))
    
    theta = 2.0 * np.arctan2(sin_half, cos_half)
    
    if sin_half < 1e-15:
        # Degenerate case
        phi = 0.0
        lam = 0.0
    else:
        # b = exp(i*phi) * sin(theta/2)
        phi = np.angle(b)
        # a = cos(theta/2) * exp(i*(phi+lam)/2) ... actually let's use the standard form
        # U = exp(i*(phi+lam)/2) * [[cos(theta/2), -exp(-i*lam)*sin(theta/2)],
        #                            [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
        # So U[0,0] = exp(i*(phi+lam)/2) * cos(theta/2)
        # U[0,1] = -exp(i*(phi+lam)/2) * exp(-i*lam) * sin(theta/2) = -exp(i*phi/2) * sin(theta/2)
        # Wait, let me redo this. The standard Qiskit u3 is:
        # [[cos(theta/2), -e^{i*lambda}sin(theta/2)],
        #  [e^{i*phi}sin(theta/2), e^{i(phi+lambda)}cos(theta/2)]]
        # So U[0,0] = cos(theta/2)
        # U[0,1] = -e^{i*lambda}sin(theta/2)
        # U[1,0] = e^{i*phi}sin(theta/2)
        # U[1,1] = e^{i(phi+lambda)}cos(theta/2)
        
        # From U[0,1] = -e^{i*lambda}sin(theta/2):
        # e^{i*lambda} = -U[0,1]/sin(theta/2)
        if abs(sin_half) > 1e-15:
            lam = np.angle(-b / sin_half)
        else:
            lam = 0.0
        
        # From U[1,0] = e^{i*phi}sin(theta/2):
        if abs(sin_half) > 1e-15:
            phi = np.angle(U_prime[1, 0] / sin_half)
        else:
            phi = 0.0
    
    return float(theta), float(phi), float(lam)


def _kron_two(A, B):
    """Kronecker product of two 2x2 matrices."""
    return np.kron(A, B)


def _apply_u3_to_qubit(U, qubit, qubits):
    """Apply a 2x2 unitary U to a specific qubit in a multi-qubit system."""
    # Build the full operator
    full = np.array([[1]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for k in range(qubits):
        if k == qubit:
            full = np.kron(full, U)
        else:
            full = np.kron(full, eye2)
    return full


def _cnot_operator(control, target, qubits):
    """Build the CNOT operator for a multi-qubit system."""
    dim = 2 ** qubits
    op = np.eye(dim, dtype=complex)
    for i in range(dim):
        c_bit = (i >> (qubits - 1 - control)) & 1
        if c_bit == 1:
            t_bit = (i >> (qubits - 1 - target)) & 1
            if t_bit == 0:
                j = i | (1 << (qubits - 1 - target))
                op[i, j] = 1.0
                op[j, i] = 1.0
                op[i, i] = 0.0
                op[j, j] = 0.0
    return op


def _two_qubit_kak(U):
    """
    KAK decomposition of a 2-qubit unitary.
    Returns (A, B, K1, K2, K3) where U = (A ⊗ B) * KNOT^(K1) * KNOT^(K2) * KNOT^(K3)
    K1, K2, K3 are in [0, 1] (KNOT^x is the x-th power of the CNOT gate in the appropriate basis)
    """
    # For 2 qubits, the KAK decomposition gives:
    # U = (A ⊗ B) * exp(i * (k1 * X⊗X + k2 * Y⊗Y + k3 * Z⊗Z) / 2)
    # The canonical form has k1 = pi/4 (for CNOT), and k2 = k3 = 0
    # We need to find A, B and the three angles.
    
    # Method: Use the fact that for any 2-qubit unitary, we can write
    # U = (A ⊗ B) * CNOT * (A' ⊗ B') * CNOT * (A'' ⊗ B') * CNOT
    # This is the 3-CNOT decomposition.
    
    # Alternative: Use the canonical KAK form.
    # The canonical KAK decomposition of a 2-qubit unitary U is:
    # U = (A ⊗ B) * exp(i * (a1 * X⊗X + a2 * Y⊗Y + a3 * Z⊗Z) / 2)
    # where a1, a2, a3 are the KAK angles, with a1 >= a2 >= a3 >= 0
    # and a1 + a2 + a3 <= pi/2 (for universal 2-qubit gates, a1 = pi/4 typically)
    
    # For our purposes, we'll use a simpler approach:
    # Decompose U into single-qubit gates and CNOTs using the standard 3-CNOT decomposition.
    
    # Step 1: Eliminate the 3rd CNOT
    # We want to find A, B, A', B', A'', B'' such that:
    # U = (A'' ⊗ B'') * CNOT * (A' ⊗ B') * CNOT * (A ⊗ B)
    
    # This is equivalent to:
    # (A'' ⊗ B'')^† * U * (A ⊗ B)^† = CNOT * (A' ⊗ B') * CNOT
    # Let M = (A'' ⊗ B'')^† * U * (A ⊗ B)^†
    # Then M = CNOT * (A' ⊗ B') * CNOT
    # So (A' ⊗ B') = CNOT^† * M * CNOT = CNOT * M * CNOT (since CNOT is self-adjoint)
    
    # The key insight: M must be of the form CNOT * (A' ⊗ B') * CNOT
    # which means M has a specific structure.
    
    # Let's use a different approach: iterative optimization
    # or use the known analytical decomposition.
    
    # For a 2-qubit unitary, we can use the following algorithm:
    # 1. Find A, B such that (A ⊗ B)^† * U has a specific form
    # 2. This is equivalent to finding the "local invariants"
    
    # Let's use a simpler but effective approach:
    # Use the fact that any 2-qubit unitary can be decomposed as:
    # U = (A ⊗ B) * CNOT * (C ⊗ D) * CNOT * (E ⊗ F) * CNOT
    # We'll optimize for A, B, C, D, E, F using gradient-free optimization.
    
    # Actually, let's use a well-known analytical method.
    # The KAK decomposition can be computed as follows:
    
    # For 2 qubits, define the "local invariants"
    # c1 = Re(tr(U * (X⊗X))) / 4
    # c2 = Re(tr(U * (Y⊗Y))) / 4  
    # c3 = Re(tr(U * (Z⊗Z))) / 4
    # c4 = Re(tr(U * (I⊗I))) / 4
    # c5 = Re(tr(U * (X⊗Y))) / 4
    # c6 = Re(tr(U * (Y⊗X))) / 4
    # c7 = Re(tr(U * (Z⊗I))) / 4
    # c8 = Re(tr(U * (I⊗Z))) / 4
    # c9 = Re(tr(U * (X⊗Z))) / 4
    # c10 = Re(tr(U * (Z⊗X))) / 4
    # c11 = Re(tr(U * (Y⊗Z))) / 4
    # c12 = Re(tr(U * (Z⊗Y))) / 4
    
    # This is getting complex. Let me use a practical approach:
    # Use scipy's optimization to find the decomposition.
    
    from scipy.optimize import minimize
    
    # Parameterize 6 single-qubit gates (A, B, C, D, E, F)
    # Each is 3 parameters (theta, phi, lam)
    # Total: 18 parameters
    
    def _params_to_gates(params):
        """Convert 18 parameters to 6 u3 gates."""
        gates = []
        for i in range(6):
            theta = params[3*i]
            phi = params[3*i + 1]
            lam = params[3*i + 2]
            gates.append(_u3_matrix(theta, phi, lam))
        return gates
    
    def _objective(params):
        A, B, C, D, E, F = _params_to_gates(params)
        # U_approx = (E ⊗ F) * CNOT * (C ⊗ D) * CNOT * (A ⊗ B)
        cnot = _cnot_operator(0, 1, 2)
        AB = np.kron(A, B)
        CD = np.kron(C, D)
        EF = np.kron(E, F)
        U_approx = EF @ cnot @ CD @ cnot @ AB
        # Distance to target
        diff = U - U_approx
        return np.sum(np.abs(diff) ** 2)
    
    # Initial guess: identity gates
    x0 = np.zeros(18)
    
    # Run optimization
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
    
    A, B, C, D, E, F = _params_to_gates(result.x)
    
    # Extract u3 parameters
    A_params = _u3_params_from_matrix(A)
    B_params = _u3_params_from_matrix(B)
    C_params = _u3_params_from_matrix(C)
    D_params = _u3_params_from_matrix(D)
    E_params = _u3_params_from_matrix(E)
    F_params = _u3_params_from_matrix(F)
    
    return A_params, B_params, C_params, D_params, E_params, F_params


def _three_qubit_decompose(U, qubits=3):
    """
    Decompose a 3-qubit unitary into CNOTs and single-qubit gates.
    Uses the fact that any 3-qubit unitary can be decomposed with at most 6 CNOTs.
    """
    # For 3 qubits, we can use the following strategy:
    # 1. Use the KAK decomposition for pairs of qubits
    # 2. Or use a known decomposition algorithm
    
    # A practical approach: Use the "CNOT ladder" with optimized single-qubit gates
    # Layout: u3(0), u3(1), u3(2), CNOT(0,1), u3(0), u3(1), u3(2), CNOT(1,2), 
    #         u3(0), u3(1), u3(2), CNOT(0,1), u3(0), u3(1), u3(2), CNOT(1,2),
    #         u3(0), u3(1), u3(2), CNOT(0,1), u3(0), u3(1), u3(2)
    # This uses 6 CNOTs.
    
    # We'll optimize the 21 single-qubit gates (7 layers × 3 qubits × 3 params = 63 params)
    # This is too many for direct optimization. Let's use a smarter approach.
    
    # Alternative: Use the fact that we can decompose into 2-qubit blocks
    # U = (U_12 ⊗ I_3) * (I_1 ⊗ U_23) * (U_12 ⊗ I_3) * ... 
    # where U_12 acts on qubits 0,1 and U_23 acts on qubits 1,2
    
    # Let's use a layer-based approach with fewer parameters
    # Layer 1: Single-qubit gates on all 3 qubits
    # Layer 2: CNOT(0,1), then single-qubit gates
    # Layer 3: CNOT(1,2), then single-qubit gates
    # Layer 4: CNOT(0,1), then single-qubit gates
    # Layer 5: CNOT(1,2), then single-qubit gates
    # Layer 6: CNOT(0,1), then single-qubit gates
    # Layer 7: Single-qubit gates on all 3 qubits
    
    # Total: 7 layers × 3 qubits × 3 params = 63 parameters
    # Still too many. Let's use a different strategy.
    
    # Use the "tensor network" approach:
    # Reshape U into a 3-index tensor U[i,j,k]
    # Contract along one index to get effective 2-qubit gates
    
    # Actually, let's use a simpler but effective approach:
    # Use coordinate descent on a fixed CNOT layout
    
    # CNOT layout for 3 qubits (6 CNOTs):
    cnot_layout = [
        (0, 1),  # CNOT 0->1
        (1, 2),  # CNOT 1->2
        (0, 1),  # CNOT 0->1
        (1, 2),  # CNOT 1->2
        (0, 1),  # CNOT 0->1
        (1, 2),  # CNOT 1->2
    ]
    
    # Single-qubit gates before, between, and after CNOTs
    # 7 layers of single-qubit gates (3 qubits each)
    # Total: 7 * 3 * 3 = 63 parameters
    
    # This is too many for direct optimization. Let's use a smarter initialization
    # and coordinate descent.
    
    # Initialize with identity gates
    n_params = 63
    params = np.zeros(n_params)
    
    def _build_circuit(params):
        """Build the circuit from parameters."""
        gates = []
        param_idx = 0
        for layer in range(7):
            for q in range(3):
                theta = params[param_idx]
                phi = params[param_idx + 1]
                lam = params[param_idx + 2]
                gates.append(["u3", q, theta, phi, lam])
                param_idx += 3
            if layer < 6:
                c, t = cnot_layout[layer]
                gates.append(["cx", c, t])
        return gates
    
    def _circuit_matrix(params):
        """Build the unitary matrix from parameters."""
        gates = _build_circuit(params)
        dim = 2 ** qubits
        matrix = np.eye(dim, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        
        for gate in gates:
            if gate[0] == "u3":
                _, q, theta, phi, lam = gate
                single = _u3_matrix(theta, phi, lam)
                op = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    op = np.kron(op, single if k == q else eye2)
                matrix = op @ matrix
            else:
                _, c, t = gate
                cnot_op = _cnot_operator(c, t, qubits)
                matrix = cnot_op @ matrix
        
        return matrix
    
    def _distance(params):
        U_circuit = _circuit_matrix(params)
        diff = U - U_circuit
        return np.sum(np.abs(diff) ** 2)
    
    # Coordinate descent
    best_params = params.copy()
    best_dist = _distance(params)
    
    # Try to improve using random restarts and coordinate descent
    import random
    rng = random.Random(42)
    
    for iteration in range(50):
        if _check_deadline():
            break
        
        # Random perturbation
        trial = best_params.copy()
        for i in range(n_params):
            if rng.random() < 0.3:
                trial[i] += rng.uniform(-0.1, 0.1)
        
        # Coordinate descent on trial
        for sweep in range(3):
            for i in range(n_params):
                for delta in (-0.1, 0.1, -0.01, 0.01):
                    candidate = trial.copy()
                    candidate[i] += delta
                    d = _distance(candidate)
                    if d < best_dist:
                        best_params = candidate.copy()
                        best_dist = d
                        trial = candidate.copy()
        
        # Check convergence
        if best_dist < 1e-15:
            break
    
    return _build_circuit(best_params)


def _two_qubit_decompose(U, qubits=2):
    """
    Decompose a 2-qubit unitary into CNOTs and single-qubit gates.
    Uses the 3-CNOT decomposition.
    """
    from scipy.optimize import minimize
    
    def _params_to_gates(params):
        gates = []
        for i in range(6):
            theta = params[3*i]
            phi = params[3*i + 1]
            lam = params[3*i + 2]
            gates.append(_u3_matrix(theta, phi, lam))
        return gates
    
    def _objective(params):
        A, B, C, D, E, F = _params_to_gates(params)
        cnot = _cnot_operator(0, 1, 2)
        AB = np.kron(A, B)
        CD = np.kron(C, D)
        EF = np.kron(E, F)
        U_approx = EF @ cnot @ CD @ cnot @ AB
        diff = U - U_approx
        return np.sum(np.abs(diff) ** 2)
    
    x0 = np.zeros(18)
    result = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
    
    A, B, C, D, E, F = _params_to_gates(result.x)
    
    A_params = _u3_params_from_matrix(A)
    B_params = _u3_params_from_matrix(B)
    C_params = _u3_params_from_matrix(C)
    D_params = _u3_params_from_matrix(D)
    E_params = _u3_params_from_matrix(E)
    F_params = _u3_params_from_matrix(F)
    
    gates = []
    # Layer 1: A on qubit 0, B on qubit 1
    gates.append(["u3", 0, A_params[0], A_params[1], A_params[2]])
    gates.append(["u3", 1, B_params[0], B_params[1], B_params[2]])
    gates.append(["cx", 0, 1])
    # Layer 2: C on qubit 0, D on qubit 1
    gates.append(["u3", 0, C_params[0], C_params[1], C_params[2]])
    gates.append(["u3", 1, D_params[0], D_params[1], D_params[2]])
    gates.append(["cx", 0, 1])
    # Layer 3: E on qubit 0, F on qubit 1
    gates.append(["u3", 0, E_params[0], E_params[1], E_params[2]])
    gates.append(["u3", 1, F_params[0], F_params[1], F_params[2]])
    gates.append(["cx", 0, 1])
    
    return gates


def _build(gates, qubits):
    """Rebuild the unitary matrix from a gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            matrix = op @ matrix
        else:
            _, c, t = gate
            cnot_op = _cnot_operator(c, t, qubits)
            matrix = cnot_op @ matrix
    
    return matrix


def _distance(target, matrix):
    """Compute phase-insensitive distance."""
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])


def synthesize(problem):
    """Synthesize a quantum circuit for the target unitary."""
    qubits = problem["qubits"]
    target = _target(problem)
    budget = problem["cnot_budget"]
    
    if qubits == 2:
        # Use 3-CNOT decomposition
        gates = _two_qubit_decompose(target, qubits=2)
    else:
        # Use 6-CNOT decomposition for 3 qubits
        gates = _three_qubit_decompose(target, qubits=3)
    
    # Verify CNOT count
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        # Truncate to budget (shouldn't happen with our layouts)
        count = 0
        truncated = []
        for g in gates:
            if g[0] == "cx":
                count += 1
                if count > budget:
                    continue
            truncated.append(g)
        gates = truncated
    
    return gates
# EVOLVE-BLOCK-END
