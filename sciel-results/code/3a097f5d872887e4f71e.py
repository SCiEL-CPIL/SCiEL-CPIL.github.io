"""Quantum circuit synthesis with known gate decompositions and optimized generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START = time.time()
_DEADLINE = 260.0  # safety margin before 270s kill


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list."""
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
    """Phase-insensitive distance."""
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _h(q):
    return _u3(q, math.pi / 2, 0, math.pi)


def _s(q):
    return _u3(q, 0, 0, math.pi / 2)


def _t(q):
    return _u3(q, 0, 0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0, 0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0, 0, -math.pi / 2)


def _known_circuits(name, qubits):
    """Return known optimal circuits for standard gates."""
    if name == "CNOT" and qubits == 2:
        return [_cx(0, 1)]

    if name == "SWAP" and qubits == 2:
        return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]

    if name == "iSWAP" and qubits == 2:
        gates = [
            _h(0), _h(1),
            _cx(0, 1),
            _s(1),
            _cx(0, 1),
            _s_dag(1),
            _h(0), _h(1)
        ]
        return gates

    if name == "Toffoli" and qubits == 3:
        gates = [
            _t(2),
            _cx(1, 2),
            _t_dag(2),
            _cx(0, 2),
            _t(1), _t(2),
            _cx(1, 2),
            _t_dag(1), _t_dag(2),
            _cx(0, 2),
            _t(0), _t(2),
            _cx(0, 1),
            _t_dag(0), _t(1),
            _cx(1, 2),
            _t_dag(2)
        ]
        return gates

    return None


def _optimize_generic(target, qubits, budget, rng_seed=42):
    """Optimize a generic unitary using gradient-based optimization."""
    if _time_left() < 5:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    import random
    rng = random.Random(rng_seed)
    
    # Build a layout with alternating CNOT patterns for good connectivity
    if qubits == 2:
        cx_pairs = [(0, 1)]
    else:
        cx_pairs = [(0, 1), (1, 2), (0, 2)]
    
    n_layers = min(budget, 10)
    layout = []
    for layer in range(n_layers):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *cx_pairs[layer % len(cx_pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def fidelity(params):
        """Return |<target|U>|^2 normalized, i.e., 1 - distance."""
        u = _build(circuit(params), qubits)
        tr = np.trace(target.conj().T @ u)
        return abs(tr) / target.shape[0]
    
    # Multiple random restarts
    best_params = None
    best_fid = 0.0
    max_restarts = 3
    if _time_left() < 20:
        max_restarts = 1
    
    for restart in range(max_restarts):
        if _time_left() < 5:
            break
        
        if restart == 0:
            params = [0.0] * n_params
        else:
            params = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        fid = fidelity(params)
        if fid > best_fid:
            best_params = params[:]
            best_fid = fid
        
        if best_fid > 1.0 - 1e-10:
            break
        
        # Gradient-based optimization with adaptive step
        lr = 0.1
        momentum = 0.5
        velocity = [0.0] * n_params
        
        steps = 80
        if _time_left() < 15:
            steps = 30
        if _time_left() < 8:
            steps = 10
        
        for step in range(steps):
            if _time_left() < 5:
                break
            
            # Compute gradient via central differences
            grad = [0.0] * n_params
            eps = 1e-4
            for i in range(n_params):
                if _time_left() < 5:
                    break
                p_plus = params[:]
                p_plus[i] += eps
                p_minus = params[:]
                p_minus[i] -= eps
                grad[i] = (fidelity(p_plus) - fidelity(p_minus)) / (2 * eps)
            
            # Update with momentum
            for i in range(n_params):
                velocity[i] = momentum * velocity[i] + lr * grad[i]
                params[i] += velocity[i]
            
            fid = fidelity(params)
            if fid > best_fid:
                best_fid = fid
                best_params = params[:]
            
            # Adaptive learning rate
            if fid < best_fid * 0.95:
                lr *= 0.5
            elif step % 10 == 0 and step > 0:
                lr = min(lr * 1.1, 0.5)
            
            if best_fid > 1.0 - 1e-10:
                break
        
        if best_fid > 1.0 - 1e-10:
            break
    
    return circuit(best_params)


def synthesize(problem):
    """Synthesize the target unitary within CNOT budget."""
    if _time_left() < 2:
        # Emergency fallback: return identity or something minimal
        qubits = problem["qubits"]
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            built = _build(known, qubits)
            dist = _distance(target, built)
            if dist < 1e-8:
                return known
    
    # Fallback to optimization if known circuit failed or doesn't exist
    return _optimize_generic(target, qubits, budget, rng_seed=42 + qubits * 100 + budget)
# EVOLVE-BLOCK-END
