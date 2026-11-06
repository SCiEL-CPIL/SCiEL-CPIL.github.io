"""
Exact decompositions for known gates + gradient-based optimization for generic unitaries.
Strategy:
1. Identify if the target is a standard gate (CNOT, iSWAP, SWAP, Toffoli) by name or structure.
2. If standard, return the exact known gate sequence.
3. If generic, use a parameterized circuit with CNOTs and U3 gates.
4. Optimize the U3 parameters using Adam gradient descent to maximize fidelity.
5. Use a deadline guard to ensure we return the best result before timeout.
"""

# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_start_time = None
_DEADLINE = 210.0  # Safety margin before 270s kill

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]

def _cx(c, t):
    return ["cx", int(c), int(t)]

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _build_fast(gates, qubits):
    """Build unitary from gate list efficiently."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([
                [c, -np.exp(1j * lam) * s],
                [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
            ])
            # Construct full operator
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            matrix = op @ matrix
        else:
            _, control, target = gate
            # CNOT implementation
            # Qubit 0 is MSB. Index = b0 b1 ... b_{n-1}
            # Control bit is at position (qubits - 1 - control)
            # Target bit is at position (qubits - 1 - target)
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            # If control is 1, flip target
            # new_index = index ^ (bit << (qubits - 1 - target))
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _is_standard_gate(name):
    return name in ["CNOT", "iSWAP", "SWAP", "Toffoli"]

def _get_known_decomposition(name, qubits):
    if name == "CNOT" and qubits == 2:
        return [_cx(0, 1)]
    elif name == "SWAP" and qubits == 2:
        return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
    elif name == "iSWAP" and qubits == 2:
        # iSWAP = exp(i*pi/4 * (X0X1 + Y0Y1))
        # Decomposition:
        # U3(0, pi/2, -pi/2, pi/2)
        # CX(0,1)
        # U3(1, pi/4, 0, pi)
        # U3(0, pi/4, 0, pi)
        # CX(0,1)
        # U3(1, pi/4, 0, -pi)
        # U3(0, -pi/4, 0, -pi)
        # U3(1, pi/2, 0, pi)
        # U3(0, pi/2, 0, pi)
        # Actually, standard iSWAP decomposition:
        # H(1), CX(0,1), RZ(pi/4)(0), RX(-pi/4)(1), CX(0,1), H(1) ... 
        # Let's use a known exact one.
        # iSWAP = [0 1; 1 0] with i on off-diagonals.
        # Standard circuit:
        # U3(1, pi/2, -pi/2, pi/2)
        # CX(0,1)
        # U3(1, pi/4, 0, pi)
        # U3(0, pi/4, 0, pi)
        # CX(0,1)
        # U3(1, pi/4, 0, -pi)
        # U3(0, -pi/4, 0, -pi)
        # U3(1, pi/2, 0, pi)
        # U3(0, pi/2, 0, pi)
        # Wait, let's just use the matrix to verify.
        # For now, return a placeholder that will be optimized if not perfect.
        # Actually, let's use the exact decomposition from literature.
        # iSWAP = (H1 CX01 RZ0(pi/4) RX1(-pi/4) CX01 H1) * diag(i,1,1,i)? No.
        # Let's use the one from Qiskit:
        # iSWAP = [
        #   0 0 0 1i
        #   0 0 1i 0
        #   0 1i 0 0
        #   1i 0 0 0
        # ]
        # Decomposition:
        # U3(0, pi/2, -pi/2, pi/2)
        # CX(0,1)
        # U3(1, pi/4, 0, pi)
        # U3(0, pi/4, 0, pi)
        # CX(0,1)
        # U3(1, pi/4, 0, -pi)
        # U3(0, -pi/4, 0, -pi)
        # U3(1, pi/2, 0, pi)
        # U3(0, pi/2, 0, pi)
        # This is 2 CNOTs.
        return [
            _u3(0, math.pi/2, -math.pi/2, math.pi/2),
            _cx(0, 1),
            _u3(1, math.pi/4, 0, math.pi),
            _u3(0, math.pi/4, 0, math.pi),
            _cx(0, 1),
            _u3(1, math.pi/4, 0, -math.pi),
            _u3(0, -math.pi/4, 0, -math.pi),
            _u3(1, math.pi/2, 0, math.pi),
            _u3(0, math.pi/2, 0, math.pi)
        ]
    elif name == "Toffoli" and qubits == 3:
        # Toffoli (CCX) decomposition with 6 CNOTs.
        # Standard decomposition:
        # H(2)
        # CX(1,2)
        # U3(2, -pi/4, 0, 0)
        # CX(0,2)
        # U3(2, pi/4, 0, 0)
        # CX(1,2)
        # H(2)
        # CX(0,1)
        # U3(1, -pi/4, 0, 0)
        # CX(0,1)
        # U3(1, pi/4, 0, 0)
        # H(2)
        # CX(1,2)
        # H(2)
        # Wait, that's 5 CNOTs? No, let's count.
        # 1. CX(1,2)
        # 2. CX(0,2)
        # 3. CX(1,2)
        # 4. CX(0,1)
        # 5. CX(0,1)
        # 6. CX(1,2)
        # 7. CX(1,2)
        # Total 7? No, the standard one is 6.
        # Let's use the known 6-CNOT decomposition.
        # It involves H, CX, T, Tdg, etc.
        # Since we only have U3, we can map T to U3(pi/2, 0, pi/4) and Tdg to U3(pi/2, 0, -pi/4).
        # H is U3(pi/2, 0, pi).
        # The circuit is:
        # H(2)
        # CX(1,2)
        # Tdg(2)
        # CX(0,2)
        # T(2)
        # CX(1,2)
        # Tdg(1)
        # CX(0,1)
        # T(1)
        # T(2)
        # CX(0,1)
        # Tdg(1)
        # H(2)
        # CX(1,2)
        # H(2)
        # Count CNOTs:
        # 1. CX(1,2)
        # 2. CX(0,2)
        # 3. CX(1,2)
        # 4. CX(0,1)
        # 5. CX(0,1)
        # 6. CX(1,2)
        # Total 6.
        return [
            _u3(2, math.pi/2, 0, math.pi), # H(2)
            _cx(1, 2),
            _u3(2, math.pi/2, 0, -math.pi/4), # Tdg(2)
            _cx(0, 2),
            _u3(2, math.pi/2, 0, math.pi/4), # T(2)
            _cx(1, 2),
            _u3(1, math.pi/2, 0, -math.pi/4), # Tdg(1)
            _cx(0, 1),
            _u3(1, math.pi/2, 0, math.pi/4), # T(1)
            _u3(2, math.pi/2, 0, math.pi/4), # T(2)
            _cx(0, 1),
            _u3(1, math.pi/2, 0, -math.pi/4), # Tdg(1)
            _u3(2, math.pi/2, 0, math.pi), # H(2)
            _cx(1, 2),
            _u3(2, math.pi/2, 0, math.pi) # H(2)
        ]
    return None

def _adam_optimize(params, objective, grad_fn, lr=0.1, iterations=1000):
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    t = 0
    best_params = params.copy()
    best_val = objective(params)
    
    for i in range(iterations):
        if _elapsed() > _DEADLINE:
            break
        t += 1
        val = objective(params)
        if val < best_val:
            best_val = val
            best_params = params.copy()
        
        grad = grad_fn(params)
        m = 0.9 * m + 0.1 * grad
        v = 0.99 * v + 0.01 * grad**2
        m_hat = m / (1 - 0.9**t)
        v_hat = v / (1 - 0.99**t)
        params = params - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        
    return best_params, best_val

def synthesize(problem):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for standard gates
    if _is_standard_gate(name):
        known = _get_known_decomposition(name, qubits)
        if known is not None:
            # Verify it fits budget and is correct
            cnot_count = sum(1 for g in known if g[0] == "cx")
            if cnot_count <= budget:
                # Quick check distance
                mat = _build_fast(known, qubits)
                dist = _distance(target, mat)
                if dist < 1e-9:
                    return known
    
    # Generic case: Gradient-based optimization
    # Construct a layout with 'budget' CNOTs and interleaved U3s
    # Layout: U3s, CX, U3s, CX, ..., U3s
    # Number of U3s per layer: qubits
    # Total layers: budget + 1
    # Total U3s: qubits * (budget + 1)
    # Total params: 3 * qubits * (budget + 1)
    
    num_layers = budget + 1
    num_u3s = qubits * num_layers
    num_params = 3 * num_u3s
    
    # Initialize parameters randomly
    np.random.seed(42)
    params = np.random.uniform(0, 2*math.pi, num_params)
    
    # Define circuit builder from params
    def build_from_params(p):
        gates = []
        idx = 0
        for layer in range(num_layers):
            for q in range(qubits):
                theta = p[idx]
                phi = p[idx+1]
                lam = p[idx+2]
                gates.append(_u3(q, theta, phi, lam))
                idx += 3
            if layer < budget:
                # Add CNOT
                # Cycle through pairs
                pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
                c, t = pairs[layer % len(pairs)]
                gates.append(_cx(c, t))
        return gates
    
    def objective(p):
        gates = build_from_params(p)
        mat = _build_fast(gates, qubits)
        return _distance(target, mat)
    
    def grad_fn(p):
        # Finite difference gradient
        grad = np.zeros_like(p)
        eps = 1e-4
        base = objective(p)
        for i in range(len(p)):
            p_plus = p.copy()
            p_plus[i] += eps
            val_plus = objective(p_plus)
            grad[i] = (val_plus - base) / eps
            p_plus[i] = p[i]
        return grad
    
    # Optimize
    best_params, best_dist = _adam_optimize(params, objective, grad_fn, lr=0.1, iterations=200)
    
    # If not good enough, try a few more restarts if time permits
    best_gates = build_from_params(best_params)
    
    return best_gates
# EVOLVE-BLOCK-END
