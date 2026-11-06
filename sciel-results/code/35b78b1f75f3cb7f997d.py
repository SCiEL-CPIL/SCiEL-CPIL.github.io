"""Quantum circuit synthesis: exact known gates + gradient-based optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 220.0
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


def _apply_u3(matrix, qubit, theta, phi, lam, qubits):
    """Apply u3 gate to a unitary matrix efficiently."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    single = np.array([[c, -np.exp(1j * lam) * s],
                       [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
    dim = 2 ** qubits
    # Build full operator via kron
    operator = np.array([[1]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    for k in range(qubits):
        operator = np.kron(operator, single if k == qubit else eye)
    return operator @ matrix


def _apply_cx(matrix, control, target, qubits):
    """Apply CNOT gate to a unitary matrix."""
    dim = 2 ** qubits
    index = np.arange(dim)
    bit = (index >> (qubits - 1 - control)) & 1
    # CNOT: if control=1, flip target
    # The mapping: for each row i, if control bit is 1, XOR target bit
    new_index = index.copy()
    mask = bit == 1
    new_index[mask] = index[mask] ^ (1 << (qubits - 1 - target))
    return matrix[new_index, :]


def _build(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            matrix = _apply_u3(matrix, q, theta, phi, lam, qubits)
        else:
            _, control, target = gate
            matrix = _apply_cx(matrix, control, target, qubits)
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target^dag U)| / dim"""
    dim = target.shape[0]
    val = abs(np.trace(target.conj().T @ matrix)) / dim
    return 1.0 - min(1.0, val)


def _fidelity(target, matrix):
    """Fidelity (what we want to maximize): |tr(target^dag U)| / dim"""
    dim = target.shape[0]
    return min(1.0, abs(np.trace(target.conj().T @ matrix)) / dim)


def _known_cnot_decomposition(name, qubits, budget):
    """Return exact gate list for known gates, or None if not a known gate."""
    if name == "CNOT" or name == "cnot":
        if qubits == 2:
            return [_cx(0, 1)]
    elif name == "iSWAP" or name == "iswap":
        if qubits == 2:
            # iSWAP: |01> -> i|10>, |10> -> i|01>, |00> -> |00>, |11> -> |11>
            # iSWAP = (H⊗H) CNOT (S⊗I) (H⊗H) CNOT (H⊗H) ... let me think more carefully
            # Actually iSWAP can be done with 3 CNOTs:
            # iSWAP = exp(i*pi/4 * (X⊗X + Y⊗Y)/2)
            # Standard decomposition: 
            # |00> -> |00>, |01> -> i|10>, |10> -> i|01>, |11> -> |11>
            # Using: H CNOT H S CNOT H CNOT H ... let me just use a known decomposition
            # iSWAP = (1/sqrt(2))[[1,0,0,i],[0,1,i,0],[0,i,1,0],[i,0,0,1]]
            # Decomposition with 3 CNOTs:
            # Step 1: Apply S on qubit 0, S on qubit 1 (i.e., phase gates)
            # Actually let me use: iSWAP = (H⊗H)(CNOT)(S⊗I)(H⊗H)(CNOT)(H⊗H)(CNOT)(H⊗H)
            # That's 3 CNOTs. Let me verify this is correct.
            # Actually a cleaner decomposition:
            # iSWAP = exp(iπ/4 (X⊗X + Y⊗Y))
            # = (H⊗H) exp(iπ/4 (Z⊗I + I⊗Z)) (H⊗H) ... no
            # Let me just use the standard 3-CNOT decomposition:
            gates = []
            # S gate on q0: diag(1, i)
            gates.append(_u3(0, math.pi, 0, math.pi))  # S gate: u3(pi, 0, pi)
            # S gate on q1
            gates.append(_u3(1, math.pi, 0, math.pi))
            # H on q0
            gates.append(_u3(0, math.pi/2, 0, math.pi))  # H gate
            # H on q1
            gates.append(_u3(1, math.pi/2, 0, math.pi))
            # CNOT 0->1
            gates.append(_cx(0, 1))
            # H on q0
            gates.append(_u3(0, math.pi/2, 0, math.pi))
            # H on q1
            gates.append(_u3(1, math.pi/2, 0, math.pi))
            # CNOT 0->1
            gates.append(_cx(0, 1))
            # H on q0
            gates.append(_u3(0, math.pi/2, 0, math.pi))
            # H on q1
            gates.append(_u3(1, math.pi/2, 0, math.pi))
            # CNOT 0->1
            gates.append(_cx(0, 1))
            # H on q0
            gates.append(_u3(0, math.pi/2, 0, math.pi))
            # H on q1
            gates.append(_u3(1, math.pi/2, 0, math.pi))
            return gates
    elif name == "SWAP" or name == "swap":
        if qubits == 2:
            # SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)
            return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
    elif name == "Toffoli" or name == "toffoli":
        if qubits == 3:
            # Toffoli with 6 CNOTs (standard decomposition)
            # This is a well-known decomposition. Let me use a standard one.
            # Toffoli = CCNOT(0,1,2)
            # Standard 6-CNOT decomposition:
            gates = []
            # H on q2
            gates.append(_u3(2, math.pi/2, 0, math.pi))
            # T† on q2: u3(pi/4, 0, -pi/2) ... T = diag(1, exp(i pi/4)), T† = diag(1, exp(-i pi/4))
            gates.append(_u3(2, math.pi/4, 0, -math.pi/2))
            # CNOT 1->2
            gates.append(_cx(1, 2))
            # T on q2
            gates.append(_u3(2, math.pi/4, 0, math.pi/2))
            # CNOT 0->2
            gates.append(_cx(0, 2))
            # T† on q2
            gates.append(_u3(2, math.pi/4, 0, -math.pi/2))
            # CNOT 1->2
            gates.append(_cx(1, 2))
            # T on q2
            gates.append(_u3(2, math.pi/4, 0, math.pi/2))
            # CNOT 0->2
            gates.append(_cx(0, 2))
            # T† on q2
            gates.append(_u3(2, math.pi/4, 0, -math.pi/2))
            # H on q2
            gates.append(_u3(2, math.pi/2, 0, math.pi))
            # This gives 4 CNOTs, but we need to check if it's correct.
            # Actually the standard Toffoli decomposition uses 6 CNOTs.
            # Let me use a more careful approach.
            # For now, let's just return None and let the optimizer handle it.
            return None
    return None


def _gradient_descent(target, qubits, cnot_budget, initial_params, initial_layout):
    """
    Gradient-based optimization of single-qubit rotation parameters.
    initial_params: list of 3*N floats (theta, phi, lam for each u3 gate)
    initial_layout: list of gate templates
    """
    dim = 2 ** qubits
    n_params = len(initial_params)
    
    def circuit_unitary(params):
        matrix = np.eye(dim, dtype=complex)
        k = 0
        for gate in initial_layout:
            if gate[0] == "u3":
                matrix = _apply_u3(matrix, gate[1], params[k], params[k+1], params[k+2], qubits)
                k += 3
            else:
                matrix = _apply_cx(matrix, gate[1], gate[2], qubits)
        return matrix
    
    def fidelity(params):
        return _fidelity(target, circuit_unitary(params))
    
    # Compute gradient using finite differences (central difference)
    def gradient(params):
        grad = np.zeros(n_params)
        h = 1e-6
        base_f = fidelity(params)
        for i in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += h
            params_minus[i] -= h
            grad[i] = (fidelity(params_plus) - fidelity(params_minus)) / (2 * h)
        return grad
    
    # L-BFGS-like optimization with momentum
    params = np.array(initial_params, dtype=float)
    best_params = params.copy()
    best_f = fidelity(params)
    
    lr = 0.1
    momentum = np.zeros(n_params)
    beta = 0.9
    
    max_iters = 2000
    no_improve_count = 0
    patience = 50
    
    for it in range(max_iters):
        if _elapsed() > _DEADLINE - 10:
            break
        
        grad = gradient(params)
        momentum = beta * momentum + (1 - beta) * grad
        params = params + lr * momentum
        
        # Clip parameters to reasonable ranges
        params[0::3] = np.clip(params[0::3], -math.pi, math.pi)  # theta
        params[1::3] = np.clip(params[1::3], -math.pi, math.pi)  # phi
        params[2::3] = np.clip(params[2::3], -math.pi, math.pi)  # lam
        
        f = fidelity(params)
        if f > best_f:
            best_f = f
            best_params = params.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > patience:
                # Reduce learning rate
                lr *= 0.5
                no_improve_count = 0
                if lr < 1e-8:
                    break
        
        if best_f > 1.0 - 1e-10:
            break
    
    return best_params, best_f


def _make_layout(qubits, cnot_budget):
    """Create a layout with CNOTs and u3 slots."""
    layout = []
    # Interleave u3 gates with CNOTs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    for layer in range(cnot_budget):
        # Add u3 gates before each CNOT
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
    
    # Add final u3 gates
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    return layout


def _params_to_gates(params, layout):
    """Convert params and layout to a gate list."""
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(_u3(gate[1], params[k], params[k+1], params[k+2]))
            k += 3
        else:
            gates.append(_cx(gate[1], gate[2]))
    return gates


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    cnot_budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known exact decompositions first
    known = _known_cnot_decomposition(name, qubits, cnot_budget)
    if known is not None:
        # Verify it's correct and within budget
        n_cnots = sum(1 for g in known if g[0] == "cx")
        if n_cnots <= cnot_budget and len(known) <= 400:
            dist = _distance(target, _build(known, qubits))
            if dist < 1e-8:
                return known
    
    # For generic unitaries, use gradient-based optimization
    layout = _make_layout(qubits, cnot_budget)
    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3
    
    # Initialize with zeros
    initial_params = [0.0] * n_params
    
    # Run gradient descent
    best_params, best_f = _gradient_descent(target, qubits, cnot_budget, initial_params, layout)
    
    # If gradient descent didn't work well, try a few random restarts
    if best_f < 0.95 and _elapsed() < _DEADLINE - 30:
        import random
        rng = random.Random(42)
        for restart in range(3):
            if _elapsed() > _DEADLINE - 20:
                break
            rand_params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
            rp, rf = _gradient_descent(target, qubits, cnot_budget, rand_params, layout)
            if rf > best_f:
                best_params, best_f = rp, rf
    
    gates = _params_to_gates(best_params, layout)
    
    # Verify CNOT count and gate count
    n_cnots = sum(1 for g in gates if g[0] == "cx")
    if n_cnots > cnot_budget or len(gates) > 400:
        # Fallback: return minimal circuit
        return [_u3(q, 0, 0, 0) for q in range(qubits)]
    
    return gates
# EVOLVE-BLOCK-END
