"""Quantum circuit synthesis with KAK decomposition and global optimization."""
# EVOLVE-BLOCK-START
import math
import random
import numpy as np
import scipy.optimize

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
    """Optimize a generic unitary using Differential Evolution."""
    # Define a layout with CNOTs and single-qubit gates
    # We will use a fixed pattern of CNOTs to reduce search space complexity
    # Pattern: 0->1, 1->0, 0->1, ... or 0->1, 1->2, 0->1, ...
    
    pairs = []
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
    elif qubits == 3:
        pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    
    # Create layout: 
    # u3s before first CNOT, u3s between CNOTs, u3s after last CNOT
    layout = []
    # Pre-CNOT u3s
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    for i in range(budget):
        c, t = pairs[i % len(pairs)]
        layout.append(["cx", c, t])
        # Post-CNOT u3s
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
    
    def objective(values):
        gates = circuit(values)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    # Bounds for parameters
    bounds = [(-math.pi, math.pi)] * n_params
    
    # Use Differential Evolution
    # maxiter and tol can be tuned. 
    # Since we have a 150s budget for all 7, we need to be efficient.
    # DE can be slow. Let's try a limited number of generations.
    try:
        result = scipy.optimize.differential_evolution(
            objective, 
            bounds, 
            seed=rng_seed, 
            maxiter=50, 
            popsize=15,
            tol=1e-6,
            polish=False
        )
        best = result.x
    except Exception:
        # Fallback to random if DE fails
        best = [random.Random(rng_seed).uniform(-math.pi, math.pi) for _ in range(n_params)]
    
    return circuit(best)

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
    return _optimize_generic(target, qubits, budget, rng_seed=42 + qubits * 100 + budget)
# EVOLVE-BLOCK-END
