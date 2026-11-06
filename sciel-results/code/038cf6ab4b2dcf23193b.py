"""Exact decompositions for known gates + fast optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 210.0
_start_time = None


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
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_known_gate(target, qubits):
    """Check if target matches a known gate up to global phase."""
    dim = 2 ** qubits
    
    # Identity
    if np.allclose(target, np.eye(dim), atol=1e-10):
        return []
    
    if qubits == 2:
        # CNOT variants
        cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        cnot_10 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        iswap = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], dtype=complex) * 1j
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        
        for name, mat, gates in [
            ("cnot_01", cnot_01, [_cx(0,1)]),
            ("cnot_10", cnot_10, [_cx(1,0)]),
            ("iswap", iswap, [_u3(0, math.pi/2, 0, math.pi), _u3(1, math.pi/2, 0, math.pi), _cx(0,1), _cx(1,0), _cx(0,1), _u3(0, math.pi/2, 0, math.pi), _u3(1, math.pi/2, 0, math.pi)]),
            ("swap", swap, [_cx(0,1), _cx(1,0), _cx(0,1)]),
        ]:
            # Check up to global phase
            if abs(np.trace(target.conj().T @ mat)) / dim > 1 - 1e-10:
                return gates
    
    if qubits == 3:
        # Toffoli (CCNOT) - control 0,1 target 2
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[7, 7] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        
        # Standard Toffoli decomposition with 6 CNOTs
        toffoli_gates = [
            _u3(2, math.pi/2, 0, math.pi),
            _cx(1, 2),
            _u3(2, math.pi/4, 0, math.pi),
            _cx(0, 2),
            _u3(2, -math.pi/4, 0, math.pi),
            _cx(1, 2),
            _u3(2, math.pi/4, 0, math.pi),
            _u3(1, math.pi/2, 0, math.pi),
            _cx(0, 1),
            _u3(1, -math.pi/4, 0, math.pi),
            _cx(0, 1),
            _u3(1, math.pi/4, 0, math.pi),
            _u3(2, -math.pi/2, 0, math.pi),
            _cx(1, 2),
            _u3(2, math.pi/4, 0, math.pi),
            _cx(0, 2),
            _u3(2, -math.pi/4, 0, math.pi),
            _cx(1, 2),
            _u3(2, math.pi/4, 0, math.pi),
            _u3(1, math.pi/2, 0, math.pi),
        ]
        
        if abs(np.trace(target.conj().T @ toffoli)) / 8 > 1 - 1e-10:
            return toffoli_gates
    
    return None


def _adam_optimize(target, qubits, budget, initial_params, max_time=60.0):
    """Adam optimizer for circuit parameters."""
    dim = 2 ** qubits
    n_params = len(initial_params)
    
    # Build layout: alternating CNOTs and u3 gates
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout.append(("u3", 0))
        layout.append(("u3", 1))
        if qubits > 2:
            layout.append(("u3", 2))
        layout.append(("cx", pairs[layer % len(pairs)]))
    for q in range(qubits):
        layout.append(("u3", q))
    
    # Count u3 gates
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params_expected = 3 * n_u3
    assert n_params == n_params_expected, f"Expected {n_params_expected} params, got {n_params}"
    
    # Adam hyperparameters
    lr = 0.05
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    # Initialize gradients
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    t = 0
    
    params = initial_params.copy()
    best_params = params.copy()
    best_dist = _distance(target, _build_fast(_circuit_from_layout(layout, params), qubits))
    
    start = time.time()
    while time.time() - start < max_time and _elapsed() < _DEADLINE - 30:
        t += 1
        
        # Compute gradient via finite differences
        grad = np.zeros(n_params)
        for i in range(n_params):
            h = 1e-5
            params_plus = params.copy()
            params_plus[i] += h
            params_minus = params.copy()
            params_minus[i] -= h
            dist_plus = _distance(target, _build_fast(_circuit_from_layout(layout, params_plus), qubits))
            dist_minus = _distance(target, _build_fast(_circuit_from_layout(layout, params_minus), qubits))
            grad[i] = (dist_plus - dist_minus) / (2 * h)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Check improvement
        dist = _distance(target, _build_fast(_circuit_from_layout(layout, params), qubits))
        if dist < best_dist:
            best_dist = dist
            best_params = params.copy()
        
        # Early stop if very close
        if best_dist < 1e-10:
            break
        
        # Reduce learning rate if no improvement
        if t % 100 == 0 and best_dist > 1e-6:
            lr *= 0.9
    
    return best_params, best_dist


def _circuit_from_layout(layout, params):
    """Convert layout and parameters to gate list."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            q = gate[1]
            gates.append(_u3(q, params[k], params[k+1], params[k+2]))
            k += 3
        else:
            gates.append(_cx(gate[1], gate[2]))
    return gates


def synthesize(problem):
    """Synthesize circuit for the given problem."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    known = _is_known_gate(target, qubits)
    if known is not None:
        return known
    
    # For generic unitaries, use optimization
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout.append(("u3", 0))
        layout.append(("u3", 1))
        if qubits > 2:
            layout.append(("u3", 2))
        layout.append(("cx", pairs[layer % len(pairs)]))
    for q in range(qubits):
        layout.append(("u3", q))
    
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    # Multiple restarts with different initializations
    best_params = None
    best_dist = float('inf')
    
    n_restarts = min(5, max(1, int((_DEADLINE - _elapsed() - 30) / 10)))
    
    for restart in range(n_restarts):
        if _elapsed() > _DEADLINE - 30:
            break
        
        # Initialize parameters
        if restart == 0:
            params = np.zeros(n_params)
        else:
            params = np.random.uniform(-math.pi, math.pi, size=n_params)
        
        # Run Adam optimization
        params, dist = _adam_optimize(target, qubits, budget, params, max_time=min(40, _DEADLINE - _elapsed() - 30))
        
        if dist < best_dist:
            best_dist = dist
            best_params = params
        
        if best_dist < 1e-10:
            break
    
    if best_params is None:
        best_params = np.zeros(n_params)
    
    return _circuit_from_layout(layout, best_params)
# EVOLVE-BLOCK-END
