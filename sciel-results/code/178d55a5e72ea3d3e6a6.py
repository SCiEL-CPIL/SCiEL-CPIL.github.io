"""Quantum circuit synthesis with exact decompositions and gradient-based optimization."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_DEADLINE = 255.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    """Return the 2x2 u3 rotation matrix."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)


def _build_unitary(gates, qubits):
    """Build the full unitary from a gate list."""
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build the full operator by kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, ctrl, tgt = gate
            # CNOT: flip target when control is 1
            # Build permutation
            perm = np.arange(dim)
            ctrl_bit = 1 << (qubits - 1 - ctrl)
            tgt_bit = 1 << (qubits - 1 - tgt)
            for i in range(dim):
                if (i & ctrl_bit) != 0:
                    perm[i] = i ^ tgt_bit
            P = np.zeros((dim, dim), dtype=complex)
            P[perm, np.arange(dim)] = 1
            U = P @ U
    return U


def _distance(target, U):
    """Phase-insensitive distance: 1 - |tr(target^dagger U)| / dim"""
    dim = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / dim)


def _exact_cnot_2q():
    """Exact circuit for CNOT on 2 qubits (control=0, target=1)."""
    return [
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 1, math.pi / 2, -math.pi / 2, 0.0],
    ]


def _exact_iswap_2q():
    """Exact circuit for iSWAP on 2 qubits using 2 CNOTs."""
    # iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
    # Decomposition: H on both, CNOT, H on both, then S phases
    # Actually: iSWAP = (S⊗S)·(H⊗H)·CNOT·(H⊗H)
    # S = diag(1, i) = u3(0, 0, pi/2)
    # H = u3(pi/2, 0, pi/2)
    gates = [
        ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 0, 0.0, 0.0, math.pi / 2],
        ["u3", 1, 0.0, 0.0, math.pi / 2],
    ]
    return gates


def _exact_swap_2q():
    """Exact circuit for SWAP on 2 qubits using 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _exact_toffoli_3q():
    """Exact circuit for Toffoli on 3 qubits (controls=0,1 target=2) using 6 CNOTs."""
    # Standard 6-CNOT Toffoli decomposition
    # T = u3(0, 0, pi/4), Tdag = u3(0, 0, -pi/4)
    # H = u3(pi/2, 0, pi/2)
    # S = u3(0, 0, pi/2)
    T = ["u3", 0.0, 0.0, math.pi / 4]
    Tdag = ["u3", 0.0, 0.0, -math.pi / 4]
    H = ["u3", math.pi / 2, 0.0, math.pi / 2]
    S = ["u3", 0.0, 0.0, math.pi / 2]
    
    gates = []
    # Apply T to target (q2)
    gates.append(["u3", 2, T[1], T[2], T[3]])
    # CNOT 1->2
    gates.append(["cx", 1, 2])
    # Apply Tdag to target
    gates.append(["u3", 2, Tdag[1], Tdag[2], Tdag[3]])
    # CNOT 0->2
    gates.append(["cx", 0, 2])
    # Apply T to target
    gates.append(["u3", 2, T[1], T[2], T[3]])
    # CNOT 1->2
    gates.append(["cx", 1, 2])
    # Apply Tdag to target
    gates.append(["u3", 2, Tdag[1], Tdag[2], Tdag[3]])
    # Apply T to control1 (q1)
    gates.append(["u3", 1, T[1], T[2], T[3]])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # Apply Tdag to control1
    gates.append(["u3", 1, Tdag[1], Tdag[2], Tdag[3]])
    # CNOT 0->1
    gates.append(["cx", 0, 1])
    # Apply T to control1
    gates.append(["u3", 1, T[1], T[2], T[3]])
    # CNOT 1->2
    gates.append(["cx", 1, 2])
    # Apply Tdag to target
    gates.append(["u3", 2, Tdag[1], Tdag[2], Tdag[3]])
    # CNOT 0->2
    gates.append(["cx", 0, 2])
    # Apply T to target
    gates.append(["u3", 2, T[1], T[2], T[3]])
    # CNOT 1->2
    gates.append(["cx", 1, 2])
    # Apply Tdag to target
    gates.append(["u3", 2, Tdag[1], Tdag[2], Tdag[3]])
    
    return gates


def _is_circuit_close(target, gates, qubits, tol=1e-10):
    U = _build_unitary(gates, qubits)
    return _distance(target, U) < tol


def _gradient_descent(target, qubits, budget, initial_params, max_time=20.0):
    """Gradient-based optimization of single-qubit rotation parameters."""
    # Layout: for each CNOT layer, have u3 on each qubit before the CNOT
    # Plus final u3 layer
    n_layers = budget
    # Parameters: 3 per u3 gate
    # Structure: [u3_q0, u3_q1, ..., u3_qn, cx, u3_q0, ..., cx, ..., u3_q0, ...]
    # Number of u3 gates: (n_layers + 1) * qubits
    n_u3 = (n_layers + 1) * qubits
    n_params = 3 * n_u3
    
    # Build CNOT layout (fixed)
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    cnot_layout = []
    for layer in range(n_layers):
        cnot_layout.append(pairs[layer % len(pairs)])
    
    def build_gates(params):
        gates = []
        idx = 0
        for layer in range(n_layers + 1):
            for q in range(qubits):
                theta = params[idx]
                phi = params[idx + 1]
                lam = params[idx + 2]
                gates.append(["u3", q, theta, phi, lam])
                idx += 3
            if layer < n_layers:
                gates.append(["cx", cnot_layout[layer][0], cnot_layout[layer][1]])
        return gates
    
    def objective(params):
        gates = build_gates(params)
        U = _build_unitary(gates, qubits)
        return _distance(target, U)
    
    # Numerical gradient
    def gradient(params):
        grad = np.zeros(n_params)
        eps = 1e-6
        f0 = objective(params)
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    # Adam-like optimizer
    params = np.array(initial_params, dtype=float)
    m = np.zeros(n_params)
    v = np.zeros(n_params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    lr = 0.1
    t = 0
    start = time.time()
    best_params = params.copy()
    best_val = objective(params)
    
    while time.time() - start < max_time:
        t += 1
        grad = gradient(params)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        
        # Keep parameters in reasonable range
        params = np.clip(params, -2 * math.pi, 2 * math.pi)
        
        val = objective(params)
        if val < best_val:
            best_val = val
            best_params = params.copy()
        
        if best_val < 1e-12:
            break
    
    return build_gates(best_params), best_val


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "").lower()
    
    # Try exact decompositions for known gates
    if qubits == 2:
        if budget >= 1 and "cnot" in name:
            gates = _exact_cnot_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
        if budget >= 2 and ("iswap" in name or "i_swap" in name):
            gates = _exact_iswap_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
        if budget >= 3 and "swap" in name:
            gates = _exact_swap_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
    
    if qubits == 3 and budget >= 6 and "toffoli" in name:
        gates = _exact_toffoli_3q()
        if _is_circuit_close(target, gates, qubits):
            return gates
    
    # For generic unitaries, use gradient-based optimization
    # Initial parameters: all zeros
    n_u3 = (budget + 1) * qubits
    n_params = 3 * n_u3
    initial_params = np.zeros(n_params)
    
    # Try a few different initializations
    best_gates = None
    best_dist = 1.0
    
    # First try with zero initialization
    gates, dist = _gradient_descent(target, qubits, budget, initial_params, max_time=15.0)
    if dist < best_dist:
        best_dist = dist
        best_gates = gates
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE - 30:
        if best_gates is not None:
            return best_gates
        # Fallback: return identity-like circuit
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Try with random initialization
    rng = np.random.RandomState(42)
    for attempt in range(3):
        if time.time() - _start_time > _DEADLINE - 30:
            break
        init = rng.uniform(-math.pi, math.pi, size=n_params)
        gates, dist = _gradient_descent(target, qubits, budget, init, max_time=10.0)
        if dist < best_dist:
            best_dist = dist
            best_gates = gates
    
    if best_gates is None:
        best_gates = _gradient_descent(target, qubits, budget, initial_params, max_time=5.0)[0]
    
    return best_gates
# EVOLVE-BLOCK-END
