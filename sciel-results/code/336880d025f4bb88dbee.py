"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

import numpy as np
from scipy.optimize import minimize


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
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
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _optimize_2qubit(target, cnot_budget):
    """Optimize a 2-qubit gate with a given CNOT budget."""
    # Use a template with the given number of CNOTs
    # For simplicity, use a fixed CNOT pattern and optimize single-qubit gates
    
    if cnot_budget <= 0:
        # No CNOTs, just single-qubit gates
        # This can't represent a general 2-qubit gate, so return identity
        return [["u3", 0, 0, 0, 0], ["u3", 1, 0, 0, 0]]
    
    # Define CNOT pattern
    cnots = []
    pairs = [(0,1), (1,0)]
    for i in range(cnot_budget):
        cnots.append(pairs[i % 2])
    
    # Template: single-qubit gates between CNOTs
    n_layers = cnot_budget + 1
    n_params = n_layers * 2 * 3  # 2 qubits, 3 params each
    
    def circuit_from_params(params):
        gates = []
        idx = 0
        for layer in range(n_layers):
            for q in range(2):
                gates.append(["u3", q, params[idx], params[idx+1], params[idx+2]])
                idx += 3
            if layer < cnot_budget:
                c, t = cnots[layer]
                gates.append(["cx", c, t])
        return gates
    
    def objective(params):
        gates = circuit_from_params(params)
        matrix = _build(gates, 2)
        return _distance(target, matrix)
    
    best_params = None
    best_obj = float('inf')
    
    rng = random.Random(42 + cnot_budget)
    n_trials = 30
    for trial in range(n_trials):
        x0 = rng.uniform(-math.pi, math.pi, size=n_params)
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    
    if best_params is not None:
        return circuit_from_params(best_params)
    
    return circuit_from_params([0.0] * n_params)


def _optimize_3qubit(target, cnot_budget):
    """Optimize a 3-qubit gate with a given CNOT budget."""
    if cnot_budget <= 0:
        return [["u3", q, 0, 0, 0] for q in range(3)]
    
    # Define CNOT pattern
    cnots = []
    pairs = [(0,1), (1,2), (0,2), (1,0), (2,1), (2,0)]
    for i in range(cnot_budget):
        cnots.append(pairs[i % len(pairs)])
    
    # Template: single-qubit gates between CNOTs
    n_layers = cnot_budget + 1
    n_params = n_layers * 3 * 3  # 3 qubits, 3 params each
    
    def circuit_from_params(params):
        gates = []
        idx = 0
        for layer in range(n_layers):
            for q in range(3):
                gates.append(["u3", q, params[idx], params[idx+1], params[idx+2]])
                idx += 3
            if layer < cnot_budget:
                c, t = cnots[layer]
                gates.append(["cx", c, t])
        return gates
    
    def objective(params):
        gates = circuit_from_params(params)
        matrix = _build(gates, 3)
        return _distance(target, matrix)
    
    best_params = None
    best_obj = float('inf')
    
    rng = random.Random(42 + cnot_budget)
    n_trials = 15
    for trial in range(n_trials):
        x0 = rng.uniform(-math.pi, math.pi, size=n_params)
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    
    if best_params is not None:
        return circuit_from_params(best_params)
    
    return circuit_from_params([0.0] * n_params)


def synthesize(problem):
    """Synthesize the target unitary into gates."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Hardcoded optimal circuits for trivially known gates
    if qubits == 2 and budget == 1:
        # Check if it's CNOT
        cnot_matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _distance(target, cnot_matrix) < 1e-10:
            return [["cx", 0, 1]]
    
    if qubits == 2 and budget == 3:
        # Check if it's SWAP
        swap_matrix = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _distance(target, swap_matrix) < 1e-10:
            return [
                ["cx", 0, 1],
                ["cx", 1, 0],
                ["cx", 0, 1]
            ]
    
    # For all other gates, use optimization
    if qubits == 2:
        return _optimize_2qubit(target, budget)
    elif qubits == 3:
        return _optimize_3qubit(target, budget)
    
    return []
# EVOLVE-BLOCK-END

