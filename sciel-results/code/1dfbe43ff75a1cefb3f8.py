"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
TIME_LIMIT = 250  # Safety margin for 270s wall-clock limit


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


def _u3_from_matrix(M):
    """Extract (theta, phi, lam) from a 2x2 unitary M."""
    if np.allclose(M, np.zeros((2, 2))):
        return (0.0, 0.0, 0.0)
    
    # M = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
    #      [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    
    # Use the (0,0) element to determine theta
    m00 = M[0, 0]
    cos_half = abs(m00)
    if cos_half > 1.0: cos_half = 1.0
    if cos_half < 0.0: cos_half = 0.0
    
    cos_half = np.clip(cos_half, 0, 1)
    theta = 2.0 * math.acos(cos_half)
    sin_half = math.sin(theta / 2)
    
    if sin_half < 1e-12:
        # Diagonal case
        phi = 0.0
        lam = 0.0
        if abs(m00) > 1e-12:
            phase00 = np.angle(m00)
        if abs(M[1, 1]) > 1e-12:
            phase11 = np.angle(M[1, 1])
            # M[1,1] = exp(i*(phi+lam))*cos(theta/2)
            # phi + lam = phase11
            # We can choose phi=0, lam=phase11
            lam = phase11
            phi = 0.0
        else:
            phi = 0.0
            lam = 0.0
    else:
        # m01 = -exp(i*lam)*sin(theta/2)
        # exp(i*lam) = -m01 / sin_half
        if abs(m01) > 1e-12:
            lam = np.angle(-m01 / sin_half)
        else:
            lam = 0.0
        
        # m10 = exp(i*phi)*sin(theta/2)
        # exp(i*phi) = m10 / sin_half
        if abs(M[1, 0]) > 1e-12:
            phi = np.angle(M[1, 0] / sin_half)
        else:
            phi = 0.0
    
    return (theta, phi, lam)


def _synthesize_cnot():
    """CNOT gate: just one CNOT."""
    return [["cx", 0, 1]]


def _synthesize_iswap():
    """iSWAP gate exact synthesis."""
    return [
        ["u3", 0, math.pi/2, 0, -math.pi/2],
        ["u3", 1, math.pi/2, -math.pi/2, 0],
        ["cx", 1, 0],
        ["u3", 0, 0, 0, math.pi/2],
        ["u3", 1, 0, 0, -math.pi/2],
        ["cx", 0, 1],
        ["u3", 0, 0, 0, -math.pi/2],
        ["u3", 1, 0, 0, math.pi/2],
    ]


def _synthesize_swap():
    """SWAP gate: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    """Toffoli gate: 6 CNOTs."""
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


def _qsd_2qubit(target, cnots=3):
    """
    Use QSD for 2-qubit unitaries.
    U = (A0 ⊗ A1) CNOT (B0 ⊗ B1) CNOT (C0 ⊗ C1)
    """
    n = 4
    # Split U into blocks based on the target qubit
    # CNOT(0,1): 
    # U = [[U00, U01], [U10, U11]] where each block is 2x2
    # Actually, for CNOT(0,1) with control=0, target=1:
    # The structure is more complex. Let's use the standard QSD algorithm.
    
    # Standard QSD for 2 qubits:
    # U = U_A CNOT U_B CNOT U_C
    # Where U_A, U_B, U_C are single-qubit gates.
    
    # Algorithm:
    # 1. U_C = U
    # 2. Apply CNOT(0,1)^dagger (which is CNOT(0,1)) to the right
    # 3. Now we have V = U_C CNOT
    # 4. V should have the form [[A, 0], [0, B]] for some single-qubit A, B? No.
    
    # Let's use the explicit formula from Shende, Markov, Bullock (2006)
    
    # U = [[U00, U01, U02, U03],
    #      [U10, U11, U12, U13],
    #      [U20, U21, U22, U23],
    #      [U30, U31, U32, U33]]
    
    # For CNOT(0,1) (control 0, target 1):
    # The matrix in computational basis |q0 q1>:
    # |00> -> |00>
    # |01> -> |01>
    # |10> -> |11>
    # |11> -> |10>
    
    # So CNOT matrix:
    # CNOT = [[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]]
    
    # QSD: U = A0⊗A1 * CNOT * B0⊗B1 * CNOT * C0⊗C1
    
    # Step 1: C0⊗C1 = U * CNOT^dagger * ... this is getting complicated.
    # Let's just use the known decomposition for the specific targets.
    
    # For generic 2-qubit, we'll use optimization.
    return None


def _generic_2qubit(problem):
    """Generic 2-qubit synthesis using optimization."""
    qubits = 2
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    if budget >= 3:
        # Use 3-CNOT topology
        cnots = [[0, 1], [1, 0], [0, 1]]
    elif budget >= 2:
        cnots = [[0, 1], [1, 0]]
    else:
        cnots = [[0, 1]]
    
    # Build layout
    layout = []
    for i, (c, t) in enumerate(cnots):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", c, t])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
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
    
    def objective(values):
        gates = circuit(values)
        U = _build_fast(gates, qubits)
        return _distance(target, U)
    
    # Multiple random starts
    best_dist = float('inf')
    best_vals = None
    
    n_starts = 10
    for s in range(n_starts):
        if _time_remaining() < 5:
            break
        if s == 0:
            x0 = np.zeros(n_params)
        else:
            x0 = np.random.uniform(-np.pi, np.pi, n_params)
        
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if res.fun < best_dist:
            best_dist = res.fun
            best_vals = res.x
    
    if best_vals is None:
        best_vals = np.zeros(n_params)
    
    return circuit(best_vals)


def _generic_3qubit(problem):
    """Generic 3-qubit synthesis using optimization."""
    qubits = 3
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Use a standard 6-CNOT topology
    if budget >= 6:
        cnots = [[0, 1], [1, 2], [0, 1], [0, 2], [1, 2], [0, 2]]
    elif budget >= 3:
        cnots = [[0, 1], [1, 2], [0, 2]]
    else:
        cnots = [[0, 1]]
    
    layout = []
    for i, (c, t) in enumerate(cnots):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", c, t])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
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
    
    def objective(values):
        gates = circuit(values)
        U = _build_fast(gates, qubits)
        return _distance(target, U)
    
    best_dist = float('inf')
    best_vals = None
    
    n_starts = 5
    for s in range(n_starts):
        if _time_remaining() < 5:
            break
        if s == 0:
            x0 = np.zeros(n_params)
        else:
            x0 = np.random.uniform(-np.pi, np.pi, n_params)
        
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        if res.fun < best_dist:
            best_dist = res.fun
            best_vals = res.x
    
    if best_vals is None:
        best_vals = np.zeros(n_params)
    
    return circuit(best_vals)


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    if name == "CNOT" and qubits == 2:
        gates = _synthesize_cnot()
    elif name == "iSWAP" and qubits == 2:
        gates = _synthesize_iswap()
    elif name == "SWAP" and qubits == 2:
        gates = _synthesize_swap()
    elif name == "Toffoli" and qubits == 3:
        gates = _synthesize_toffoli()
    else:
        if qubits == 2:
            gates = _generic_2qubit(problem)
        else:
            gates = _generic_3qubit(problem)
    
    # Verify and potentially improve
    target = _target(problem)
    matrix = _build_fast(gates, qubits)
    dist = _distance(target, matrix)
    
    if dist > 1e-6 and _time_remaining() > 20:
        # Try to improve with more optimization
        if qubits == 2:
            gates = _generic_2qubit(problem)
        else:
            gates = _generic_3qubit(problem)
    
    return gates
