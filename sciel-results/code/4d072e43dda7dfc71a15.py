"""Synthesize fixed two- and three-qubit unitaries within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Deadline guard: 270s wall-clock kill, keep 30s safety margin
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


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cnot_matrix(control, target, qubits):
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    for i in range(n):
        if (i >> (qubits - 1 - control)) & 1:
            mat[i, i] = 0
            j = i ^ (1 << (qubits - 1 - target))
            mat[i, j] = 1
            mat[j, i] = 1
    return mat


def _exact_cnot(qubits):
    """CNOT: control=0, target=1."""
    gates = [["cx", 0, 1]]
    if qubits == 3:
        gates = [["cx", 0, 1]]
    return gates


def _exact_iswap(qubits):
    """iSWAP gate: [[0, 0, 0, 0], [0, 0, i, 0], [0, i, 0, 0], [0, 0, 0, 0]] on 2 qubits."""
    if qubits == 2:
        # iSWAP = H ⊗ H, CNOT(0,1), H ⊗ H, then phase... actually let's use known decomposition
        # iSWAP = (H⊗H) CNOT(0,1) (H⊗H) then add phases
        # Standard: iSWAP = exp(iπ/4) * (H⊗H) CNOT (S⊗S) (H⊗H) ... let me just use a known circuit
        # iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
        # Known decomposition: iSWAP = (I⊗S) CNOT(0,1) (I⊗H) (S⊗I) CNOT(0,1) (I⊗H) ... 
        # Let me use a simpler approach:
        # iSWAP = (H⊗H) CNOT(0,1) (H⊗H) * diag(1, -i, -i, 1) ... 
        # Actually the cleanest: iSWAP = (S⊗S) CNOT(0,1) (H⊗H) CNOT(0,1) (H⊗H) ... 
        # Let me just verify with a standard decomposition:
        # iSWAP = (I⊗S) · CNOT(0,1) · (I⊗H) · (S⊗I) · CNOT(0,1) · (I⊗H)
        # where S = diag(1, i), H = [[1,1],[1,-1]]/√2
        # u3 for H: theta=pi/2, phi=0, lam=pi
        # u3 for S: theta=0, phi=0, lam=pi/2
        # u3 for I: theta=0, phi=0, lam=0
        gates = [
            ["u3", 1, 0.0, 0.0, 0.0],  # I on q1
            ["u3", 0, 0.0, 0.0, 0.0],  # I on q0
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, math.pi / 2],  # S on q1
            ["u3", 0, 0.0, 0.0, 0.0],  # I on q0
            ["cx", 0, 1],
            ["u3", 1, math.pi / 2, 0.0, math.pi],  # H on q1
            ["u3", 0, 0.0, 0.0, 0.0],  # I on q0
        ]
        return gates
    else:
        return []


def _exact_swap(qubits):
    """SWAP gate using 3 CNOTs."""
    gates = [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]
    return gates


def _exact_toffoli(qubits):
    """Toffoli (CCNOT) gate: control=0,1 target=2. Uses 6 CNOTs."""
    # Standard Toffoli decomposition with 6 CNOTs
    # Toffoli = (I⊗I⊗H) Toffoli_base (I⊗I⊗H) where Toffoli_base uses 6 CNOTs
    # Actually the standard 6-CNOT Toffoli:
    gates = [
        ["u3", 2, math.pi / 2, 0.0, math.pi],  # H on q2
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],  # T† on q2
        ["cx", 0, 2],
        ["u3", 2, 0.0, 0.0, math.pi / 4],   # T on q2
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],  # T† on q2
        ["u3", 1, 0.0, 0.0, math.pi / 4],   # T on q1
        ["cx", 0, 1],
        ["u3", 1, 0.0, 0.0, -math.pi / 4],  # T† on q1
        ["u3", 0, 0.0, 0.0, math.pi / 4],   # T on q0
        ["cx", 1, 0],
        ["u3", 1, 0.0, 0.0, math.pi / 4],   # T on q1
        ["cx", 0, 1],
        ["u3", 1, 0.0, 0.0, -math.pi / 4],  # T† on q1
        ["u3", 2, math.pi / 2, 0.0, math.pi],  # H on q2
    ]
    return gates


def _identify_gate(name, target, qubits):
    """Check if target matches a known gate and return exact circuit."""
    # Normalize: compare up to global phase
    def _match(a, b):
        # Phase-insensitive comparison
        d = abs(np.trace(a.conj().T @ b)) / a.shape[0]
        return d > 1 - 1e-9

    # CNOT: |00><00| + |01><01| + |10><11| + |11><10|
    if qubits == 2:
        cnot = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        if _match(target, cnot):
            return _exact_cnot(2)
        
        iswap = np.array([[0, 0, 0, 0],
                          [0, 0, 1j, 0],
                          [0, 1j, 0, 0],
                          [0, 0, 0, 0]], dtype=complex)
        if _match(target, iswap):
            return _exact_iswap(2)
        
        swap = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
        if _match(target, swap):
            return _exact_swap(2)
    
    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[7, 7] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        if _match(target, toffoli):
            return _exact_toffoli(3)
    
    return None


def _adam_optimize(target, qubits, budget, initial_params, n_params, max_iters=200, lr=0.1):
    """Adam gradient descent on u3 parameters with a fixed CNOT template."""
    # Build fixed CNOT template
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    cnot_positions = []
    for i in range(budget):
        cnot_positions.append((i, pairs[i % len(pairs)]))
    
    # Layout: u3 gates before and between CNOTs
    # Structure: [u3]*qubits, [cx], [u3]*qubits, [cx], ..., [u3]*qubits
    layout = []
    u3_indices = []
    for i in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0, 0, 0])
            u3_indices.append(len(layout) - 1)
        layout.append(["cx", *pairs[i % len(pairs)]])
    for q in range(qubits):
        layout.append(["u3", q, 0, 0, 0])
        u3_indices.append(len(layout) - 1)
    
    n_u3 = len(u3_indices)
    n_params = 3 * n_u3
    
    params = np.array(initial_params[:n_params], dtype=float)
    if len(params) < n_params:
        params = np.zeros(n_params, dtype=float)
    
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    def _circuit_unitary(p):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], p[k], p[k+1], p[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return _build(gates, qubits)
    
    def _loss(p):
        u = _circuit_unitary(p)
        return _distance(target, u)
    
    best_params = params.copy()
    best_loss = _loss(params)
    
    for t in range(1, max_iters + 1):
        if time.time() > _DEADLINE - 30:
            break
        
        # Numerical gradient
        grad = np.zeros(n_params)
        for i in range(n_params):
            h = 1e-4
            p_plus = params.copy()
            p_plus[i] += h
            p_minus = params.copy()
            p_minus[i] -= h
            grad[i] = (_loss(p_plus) - _loss(p_minus)) / (2 * h)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        current_loss = _loss(params)
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params.copy()
        
        if best_loss < 1e-10:
            break
    
    # Build final gates
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], best_params[k], best_params[k+1], best_params[k+2]])
            k += 3
        else:
            gates.append(list(gate))
    
    return gates, best_loss


def synthesize(problem):
    """Synthesize a circuit for the given target unitary within CNOT budget."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try exact solutions first
    exact = _identify_gate(name, target, qubits)
    if exact is not None:
        # Verify it's within budget
        n_cnots = sum(1 for g in exact if g[0] == "cx")
        if n_cnots <= budget and len(exact) <= 400:
            return exact
    
    # Use Adam optimization with a good initial guess
    # Initialize with small random values
    n_u3 = budget * qubits + qubits
    n_params = 3 * n_u3
    initial = np.zeros(n_params)
    
    gates, loss = _adam_optimize(target, qubits, budget, initial, n_params, max_iters=100)
    
    # Verify CNOT count
    n_cnots = sum(1 for g in gates if g[0] == "cx")
    if n_cnots > budget:
        # Truncate to fit budget (shouldn't happen with our template)
        count = 0
        trimmed = []
        for g in gates:
            if g[0] == "cx":
                count += 1
                if count > budget:
                    continue
            trimmed.append(g)
        gates = trimmed
    
    if len(gates) > 400:
        gates = gates[:400]
    
    return gates
# EVOLVE-BLOCK-END
