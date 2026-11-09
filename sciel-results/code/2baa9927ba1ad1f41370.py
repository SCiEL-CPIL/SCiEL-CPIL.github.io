"""Synthesize fixed two- and three-qubit unitaries within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = time.time() + 240.0


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build the full unitary from a gate list."""
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
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cx_matrix(qubits, control, target):
    """Return the CNOT matrix for given qubit positions."""
    n = 2 ** qubits
    mat = np.zeros((n, n), dtype=complex)
    for i in range(n):
        c_bit = (i >> (qubits - 1 - control)) & 1
        t_bit = (i >> (qubits - 1 - target)) & 1
        j = i
        if c_bit == 1:
            j = i ^ (1 << (qubits - 1 - target))
        mat[i, j] = 1.0
    return mat


def _solve_2qubit(target, budget):
    """
    Solve a 2-qubit unitary using CNOT + u3 decomposition.
    Standard decomposition: U = A CNOT B CNOT C CNOT D
    where A,B,C,D are single-qubit rotations.
    """
    U = target
    # For 2 qubits, we can use the standard decomposition
    # U = (A0 ⊗ A1) CNOT (B0 ⊗ B1) CNOT (C0 ⊗ C1) CNOT (D0 ⊗ D1)
    # This requires 3 CNOTs minimum.
    
    # Use scipy's unitary_to_rotations or manual approach
    # Let's use the standard 2-qubit decomposition
    
    # First, factor out global phase
    phase = np.angle(np.linalg.det(U))
    U_no_phase = U * np.exp(-1j * phase)
    
    # Standard 2-qubit decomposition using CNOTs
    # We'll use the approach: U = (A0⊗A1) · CNOT · (B0⊗B1) · CNOT · (C0⊗C1) · CNOT · (D0⊗D1)
    
    # This is complex; let's use a simpler approach for known gates
    # and gradient descent for generic ones
    
    # For now, use gradient descent with a fixed CNOT template
    # Template: u3 on all qubits, CNOT, u3 on all qubits, CNOT, u3 on all qubits
    # Number of CNOTs = min(budget, 4) for 2 qubits
    
    n_cnots = min(budget, 4)
    if n_cnots < 3:
        n_cnots = 3  # minimum for 2-qubit universal
    
    # Build template
    layout = []
    for layer in range(n_cnots):
        layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]
        layout.append(["cx", 0, 1])
    layout += [["u3", 0, 0.0, 0.0, 0.0], ["u3", 1, 0.0, 0.0, 0.0]]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    # Initialize parameters
    params = np.zeros(n_params)
    
    # Use gradient descent
    def circuit_from_params(p):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], p[k], p[k+1], p[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(p):
        gates = circuit_from_params(p)
        U_circuit = _build(gates, 2)
        return _distance(U, U_circuit)
    
    # Adam optimizer
    lr = 0.05
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    t = 0
    best_params = params.copy()
    best_val = objective(params)
    
    max_iters = 2000
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        t += 1
        # Compute gradient numerically
        grad = np.zeros(n_params)
        for i in range(n_params):
            h = 1e-5
            p_plus = params.copy()
            p_plus[i] += h
            p_minus = params.copy()
            p_minus[i] -= h
            grad[i] = (objective(p_plus) - objective(p_minus)) / (2 * h)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        val = objective(params)
        if val < best_val:
            best_val = val
            best_params = params.copy()
    
    gates = circuit_from_params(best_params)
    return gates


def _solve_3qubit(target, budget):
    """
    Solve a 3-qubit unitary using CNOT + u3 decomposition.
    Use a fixed CNOT template with gradient descent.
    """
    U = target
    
    # Build template with CNOTs
    # For 3 qubits, use a sequence of CNOTs
    # Template: layers of u3 + CNOT pairs
    n_cnots = min(budget, 12)  # reasonable upper bound
    
    # CNOT pairs for 3 qubits
    pairs = [(0,1), (0,2), (1,2)]
    
    layout = []
    for layer in range(n_cnots):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(3)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit_from_params(p):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], p[k], p[k+1], p[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def objective(p):
        gates = circuit_from_params(p)
        U_circuit = _build(gates, 3)
        return _distance(U, U_circuit)
    
    # Adam optimizer
    lr = 0.03
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    t = 0
    best_params = np.zeros(n_params)
    best_val = objective(best_params)
    
    max_iters = 1500
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        t += 1
        # Compute gradient numerically (sparse)
        grad = np.zeros(n_params)
        # Sample a subset of parameters for gradient if too many
        if n_params > 100:
            indices = np.random.choice(n_params, size=50, replace=False)
        else:
            indices = range(n_params)
        
        for i in indices:
            h = 1e-5
            p_plus = best_params.copy()
            p_plus[i] += h
            p_minus = best_params.copy()
            p_minus[i] -= h
            grad[i] = (objective(p_plus) - objective(p_minus)) / (2 * h)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        best_params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        val = objective(best_params)
    
    gates = circuit_from_params(best_params)
    return gates


def _known_gate_solution(name, qubits, budget):
    """Return exact gate lists for known gates."""
    if name == "CNOT":
        if qubits == 2:
            return [["u3", 0, 0.0, 0.0, 0.0],
                     ["u3", 1, 0.0, 0.0, 0.0],
                     ["cx", 0, 1],
                     ["u3", 0, 0.0, 0.0, 0.0],
                     ["u3", 1, 0.0, 0.0, 0.0]]
        else:
            # For 3 qubits, CNOT on qubits 0,1
            return [["u3", 0, 0.0, 0.0, 0.0],
                     ["u3", 1, 0.0, 0.0, 0.0],
                     ["u3", 2, 0.0, 0.0, 0.0],
                     ["cx", 0, 1],
                     ["u3", 0, 0.0, 0.0, 0.0],
                     ["u3", 1, 0.0, 0.0, 0.0],
                     ["u3", 2, 0.0, 0.0, 0.0]]
    elif name == "SWAP":
        if qubits == 2:
            # SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)
            return [["cx", 0, 1],
                    ["cx", 1, 0],
                    ["cx", 0, 1]]
        else:
            # SWAP on qubits 0,1 in 3-qubit system
            return [["u3", 0, 0.0, 0.0, 0.0],
                     ["u3", 1, 0.0, 0.0, 0.0],
                     ["u3", 2, 0.0, 0.0, 0.0],
                     ["cx", 0, 1],
                     ["u3", 0, 0.0, 0.0, 0.0],
                     ["u3", 1, 0.0, 0.0, 0.0],
                     ["u3", 2, 0.0, 0.0, 0.0],
                     ["cx", 1, 0],
                     ["u3", 0, 0.0, 0.0, 0.0],
                     ["u3", 1, 0.0, 0.0, 0.0],
                     ["u3", 2, 0.0, 0.0, 0.0],
                     ["cx", 0, 1],
                     ["u3", 0, 0.0, 0.0, 0.0],
                     ["u3", 1, 0.0, 0.0, 0.0],
                     ["u3", 2, 0.0, 0.0, 0.0]]
    elif name == "iSWAP":
        # iSWAP: |01> -> i|10>, |10> -> i|01>
        if qubits == 2:
            # iSWAP = (H⊗H) · SWAP · (H⊗H) with phase
            # Actually iSWAP = exp(iπ/4) · (|00><00| + i|01><10| + i|10><01| + |11><11|)
            # Let's use: iSWAP = (Rz(π/2)⊗Rz(π/2)) · CNOT · (Rx(π/2)⊗Rx(π/2)) · CNOT · (Rz(π/2)⊗Rz(π/2))
            # Simplified: use known decomposition
            pi2 = math.pi / 2
            return [["u3", 0, pi2, 0.0, 0.0],
                     ["u3", 1, pi2, 0.0, 0.0],
                     ["cx", 0, 1],
                     ["u3", 0, pi2, 0.0, 0.0],
                     ["u3", 1, pi2, 0.0, 0.0],
                     ["cx", 1, 0],
                     ["u3", 0, pi2, 0.0, 0.0],
                     ["u3", 1, pi2, 0.0, 0.0],
                     ["cx", 0, 1],
                     ["u3", 0, pi2, 0.0, 0.0],
                     ["u3", 1, pi2, 0.0, 0.0]]
        else:
            return _solve_3qubit(_target({"real": [], "imaginary": []}), budget)  # fallback
    elif name == "Toffoli":
        # Toffoli: |110> -> |111>
        if qubits == 3:
            # Standard Toffoli decomposition with 6 CNOTs
            # Toffoli = (I⊗I⊗T) · CNOT(1,2) · (I⊗I⊗T†) · CNOT(0,2) · (I⊗T⊗I) · CNOT(1,2) · (I⊗T†⊗I) · CNOT(0,2) · (I⊗I⊗T)
            # T = diag(1, e^{iπ/4}), T† = diag(1, e^{-iπ/4})
            # In u3: T = u3(π/2, 0, π/2), T† = u3(π/2, 0, -π/2)
            # Wait, let me use the standard 6-CNOT Toffoli
            pi4 = math.pi / 4
            gates = [
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, 0.0, 0.0, 0.0],
                ["u3", 2, pi4, 0.0, 0.0],  # T on qubit 2
                ["cx", 1, 2],
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, 0.0, 0.0, 0.0],
                ["u3", 2, -pi4, 0.0, 0.0],  # T† on qubit 2
                ["cx", 0, 2],
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, pi4, 0.0, 0.0],  # T on qubit 1
                ["u3", 2, 0.0, 0.0, 0.0],
                ["cx", 1, 2],
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, -pi4, 0.0, 0.0],  # T† on qubit 1
                ["u3", 2, 0.0, 0.0, 0.0],
                ["cx", 0, 2],
                ["u3", 0, 0.0, 0.0, 0.0],
                ["u3", 1, 0.0, 0.0, 0.0],
                ["u3", 2, pi4, 0.0, 0.0],  # T on qubit 2
            ]
            return gates
        else:
            return _solve_2qubit(_target({"real": [], "imaginary": []}), budget)  # fallback
    return None


def synthesize(problem):
    """Synthesize a circuit for the given problem."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known gate solutions first
    known = _known_gate_solution(name, qubits, budget)
    if known is not None:
        # Verify the known solution
        U_circuit = _build(known, qubits)
        dist = _distance(target, U_circuit)
        if dist < 1e-6:
            return known
    
    # Use gradient-based optimization
    if qubits == 2:
        return _solve_2qubit(target, budget)
    else:
        return _solve_3qubit(target, budget)
# EVOLVE-BLOCK-END
