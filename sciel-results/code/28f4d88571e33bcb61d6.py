"""Synthesize fixed two- and three-qubit unitaries within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: 270s wall-clock kill, keep 30s safety margin
_DEADLINE = time.time() + 240.0

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)

def _cx_matrix(control, target, qubits):
    eye = np.eye(2, dtype=complex)
    cx_2q = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    
    # Build full operator
    op = np.array([[1]], dtype=complex)
    for k in range(qubits):
        if k == control:
            # Control is 0
            op = np.kron(op, np.array([[1, 0], [0, 0]], dtype=complex))
        elif k == target:
            # Target is X
            op = np.kron(op, np.array([[0, 1], [1, 0]], dtype=complex))
        else:
            op = np.kron(op, eye)
            
    # This simple kron approach for CX is wrong because CX is not a product of single qubit ops.
    # Correct way:
    mat = np.eye(2 ** qubits, dtype=complex)
    for i in range(2 ** qubits):
        c_bit = (i >> (qubits - 1 - control)) & 1
        t_bit = (i >> (qubits - 1 - target)) & 1
        if c_bit == 1:
            j = i ^ (1 << (qubits - 1 - target))
            mat[i, j] = 1
        else:
            mat[i, i] = 1
    return mat

def _build_circuit(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            operator = _cx_matrix(control, target, qubits)
            matrix = operator @ matrix
    return matrix

def _distance(target, matrix):
    d = 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])
    return d

def _analytic_cnot():
    return [['cx', 0, 1]]

def _analytic_swap():
    return [
        ['cx', 0, 1],
        ['cx', 1, 0],
        ['cx', 0, 1]
    ]

def _analytic_iswap():
    # iSWAP decomposition
    # R_y(pi/2) on 0, R_y(-pi/2) on 1
    # CX 0->1
    # R_z(pi/2) on 0, R_z(-pi/2) on 1
    # CX 0->1
    # R_y(-pi/2) on 0, R_y(pi/2) on 1
    # CX 0->1
    gates = [
        ['u3', 0, math.pi/2, 0, 0], # Ry(pi/2)
        ['u3', 1, math.pi/2, 0, math.pi], # Ry(-pi/2) is u3(pi/2, 0, pi)? 
        # Ry(theta) = u3(theta, 0, pi)
        # So Ry(-pi/2) = u3(-pi/2, 0, pi) = u3(3pi/2, 0, pi)
        # Let's use standard u3(theta, phi, lam)
        # Ry(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
        # u3(theta, 0, pi) = [[cos(theta/2), -e^{ipi}sin(theta/2)], [e^{i0}sin(theta/2), e^{ipi}cos(theta/2)]]
        # = [[cos(theta/2), sin(theta/2)], [sin(theta/2), -cos(theta/2)]]
        # This is not Ry.
        # Ry(theta) = e^{-i theta/2 sigma_y}
        # u3(theta, 0, 0) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]] = Ry(theta)
        # So Ry(pi/2) = u3(pi/2, 0, 0)
        # Ry(-pi/2) = u3(-pi/2, 0, 0)
        
        ['u3', 0, math.pi/2, 0, 0],
        ['u3', 1, -math.pi/2, 0, 0],
        ['cx', 0, 1],
        ['u3', 0, 0, 0, math.pi/2], # R_z(pi/2) = u3(0, 0, pi/2)
        ['u3', 1, 0, 0, -math.pi/2], # R_z(-pi/2) = u3(0, 0, -pi/2)
        ['cx', 0, 1],
        ['u3', 0, -math.pi/2, 0, 0],
        ['u3', 1, math.pi/2, 0, 0],
        ['cx', 0, 1]
    ]
    return gates

def _numeric_synthesize(target, qubits, budget):
    # Determine CNOT ladder
    if qubits == 2:
        # Use 3 CNOTs
        cnots = [(0, 1), (0, 1), (0, 1)]
    else:
        # Use 6 CNOTs for 3 qubits
        cnots = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
    
    # Build layout: u3 on all qubits, then cx, repeat
    layout = []
    for i, (c, t) in enumerate(cnots):
        layout.append(('u3', 0))
        layout.append(('u3', 1))
        if qubits == 3:
            layout.append(('u3', 2))
        layout.append(('cx', c, t))
        
    # Final u3s
    layout.append(('u3', 0))
    layout.append(('u3', 1))
    if qubits == 3:
        layout.append(('u3', 2))
        
    num_params = sum(1 for l in layout if l[0] == 'u3') * 3
    best_x = np.zeros(num_params)
    best_dist = 1.0
    
    def objective(x):
        gates = []
        k = 0
        for l in layout:
            if l[0] == 'u3':
                q = l[1]
                gates.append(['u3', q, x[k], x[k+1], x[k+2]])
                k += 3
            else:
                gates.append(['cx', l[1], l[2]])
        
        try:
            mat = _build_circuit(gates, qubits)
            return _distance(target, mat)
        except Exception:
            return 1.0
        
    # Optimization
    start_time = time.time()
    max_time = 20.0 # 20s per numeric target
    
    # Use Nelder-Mead
    res = minimize(objective, best_x, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'fatol': 1e-12, 'xatol': 1e-12})
    
    if res.fun < best_dist:
        best_dist = res.fun
        best_x = res.x
        
    # Construct final gates
    gates = []
    k = 0
    for l in layout:
        if l[0] == 'u3':
            q = l[1]
            gates.append(['u3', q, best_x[k], best_x[k+1], best_x[k+2]])
            k += 3
        else:
            gates.append(['cx', l[1], l[2]])
            
    return gates

def synthesize(problem):
    global _DEADLINE
    if time.time() > _DEADLINE - 10:
        # Return a dummy circuit if out of time
        return []
        
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    if name == "CNOT" and qubits == 2:
        gates = _analytic_cnot()
        dist = _distance(target, _build_circuit(gates, qubits))
        if dist < 1e-10:
            return gates
    elif name == "SWAP" and qubits == 2:
        gates = _analytic_swap()
        dist = _distance(target, _build_circuit(gates, qubits))
        if dist < 1e-10:
            return gates
    elif name == "iSWAP" and qubits == 2:
        gates = _analytic_iswap()
        dist = _distance(target, _build_circuit(gates, qubits))
        if dist < 1e-10:
            return gates
            
    # Fallback to numeric
    return _numeric_synthesize(target, qubits, budget)
# EVOLVE-BLOCK-END
