"""Quantum circuit synthesis using optimized brickwork circuits."""
# EVOLVE-BLOCK-START
import math
import numpy as np
from scipy.optimize import minimize

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

def _synthesize_generic(target, qubits, budget):
    """
    Synthesize a generic unitary using a fixed CNOT topology and optimizing single-qubit gates.
    """
    if qubits == 2:
        # Use min(budget, 3) CNOTs, but 3 is standard for universality
        n_cnots = min(budget, 3)
        if n_cnots == 0:
             # Just single qubit gates? Not universal for 2q.
             n_cnots = 1 if budget >= 1 else 0
        cnot_pairs = [(0, 1), (1, 0), (0, 1)]
        layout = []
        for i in range(n_cnots):
            layout += [["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0]]
            layout.append(["cx", cnot_pairs[i][0], cnot_pairs[i][1]])
        layout += [["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0]]
        
    elif qubits == 3:
        n_cnots = budget
        cnot_pairs = [(0, 1), (1, 2)]
        layout = []
        for i in range(n_cnots):
            layout += [["u3", q, 0, 0, 0] for q in range(3)]
            layout.append(["cx", cnot_pairs[i % 2][0], cnot_pairs[i % 2][1]])
        layout += [["u3", q, 0, 0, 0] for q in range(3)]
        
    else:
        return []

    # Extract parameters
    param_indices = []
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_indices.append(i)
            
    n_params = len(param_indices) * 3
    
    def build_from_params(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                q = gate[1]
                theta, phi, lam = params[k], params[k+1], params[k+2]
                gates.append(["u3", q, theta, phi, lam])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    def objective(params):
        gates = build_from_params(params)
        U = _build(gates, qubits)
        return _distance(target, U)
        
    # Initial guess: zeros
    x0 = np.zeros(n_params)
    
    # Optimize
    bounds = [(-math.pi, math.pi)] * n_params
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        best_params = res.x
    except:
        best_params = x0
        
    best_gates = build_from_params(best_params)
    
    return best_gates

def synthesize(problem):
    """Synthesize the target unitary within CNOT budget."""
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
    
    # For generic or incorrect known circuits, use optimization
    return _synthesize_generic(target, qubits, budget)
# EVOLVE-BLOCK-END
