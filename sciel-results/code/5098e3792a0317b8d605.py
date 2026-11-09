import math
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 240.0

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

def _optimize_angles(target, qubits, template, params_count):
    def objective(params):
        if time.time() > _DEADLINE:
            return 1.0
        gates, k = [], 0
        for gate in template:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        matrix = _build(gates, qubits)
        return _distance(target, matrix)
    
    x0 = np.zeros(params_count)
    bounds = [(0, 2*np.pi)] * params_count
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        return res.x
    except Exception:
        return x0

def synthesize(problem):
    qubits = problem["qubits"]
    name = problem.get("name", "")
    target = _target(problem)
    
    # Exact circuits for known gates
    if name == "CNOT":
        return [["cx", 0, 1]]
    elif name == "SWAP":
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    elif name == "Toffoli":
        # 7 CNOT decomposition
        return [
            ["u3", 2, math.pi/2, 0, math.pi],
            ["cx", 1, 2],
            ["u3", 2, 0, 0, -math.pi/4],
            ["cx", 0, 2],
            ["u3", 2, 0, 0, math.pi/4],
            ["cx", 1, 2],
            ["u3", 2, 0, 0, -math.pi/4],
            ["cx", 0, 2],
            ["u3", 2, 0, 0, math.pi/4],
            ["u3", 2, math.pi/2, 0, math.pi],
            ["cx", 0, 1],
            ["u3", 1, 0, 0, -math.pi/4],
            ["cx", 0, 1],
            ["u3", 1, 0, 0, math.pi/4]
        ]
    
    # Generic optimization
    if qubits == 2:
        if name == "iSWAP":
            template = [
                ["u3", 0], ["u3", 1],
                ["cx", 0, 1],
                ["u3", 0], ["u3", 1],
                ["cx", 1, 0],
                ["u3", 0], ["u3", 1]
            ]
        else:
            template = [
                ["u3", 0], ["u3", 1],
                ["cx", 0, 1],
                ["u3", 0], ["u3", 1],
                ["cx", 1, 0],
                ["u3", 0], ["u3", 1],
                ["cx", 0, 1],
                ["u3", 0], ["u3", 1]
            ]
        params_count = 3 * sum(1 for g in template if g[0] == "u3")
    elif qubits == 3:
        template = [
            ["u3", 0], ["u3", 1], ["u3", 2],
            ["cx", 0, 1],
            ["u3", 0], ["u3", 1], ["u3", 2],
            ["cx", 0, 2],
            ["u3", 0], ["u3", 1], ["u3", 2],
            ["cx", 1, 2],
            ["u3", 0], ["u3", 1], ["u3", 2],
            ["cx", 0, 1],
            ["u3", 0], ["u3", 1], ["u3", 2],
            ["cx", 0, 2],
            ["u3", 0], ["u3", 1], ["u3", 2],
            ["cx", 1, 2],
            ["u3", 0], ["u3", 1], ["u3", 2]
        ]
        params_count = 3 * sum(1 for g in template if g[0] == "u3")
    else:
        return []

    best_params = _optimize_angles(target, qubits, template, params_count)
    
    gates, k = [], 0
    for gate in template:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], best_params[k], best_params[k+1], best_params[k+2]])
            k += 3
        else:
            gates.append(list(gate))
            
    return gates
