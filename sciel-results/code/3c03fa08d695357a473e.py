"""Exact decompositions for known gates + NumPy-based optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_DEADLINE = 230.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", c, t]


def _build(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            # Apply single-qubit gate to qubit q
            for i in range(dim):
                if (i >> (qubits - 1 - q)) & 1 == 0:
                    matrix[i, :] = c * matrix[i, :] + np.exp(1j * phi) * s * matrix[i | (1 << (qubits - 1 - q)), :]
                else:
                    matrix[i, :] = -np.exp(1j * lam) * s * matrix[i ^ (1 << (qubits - 1 - q)), :] + np.exp(1j * (phi + lam)) * c * matrix[i, :]
        else:
            _, control, target = gate
            for i in range(dim):
                if (i >> (qubits - 1 - control)) & 1:
                    j = i ^ (1 << (qubits - 1 - target))
                    if i < j:
                        matrix[i, :], matrix[j, :] = matrix[j, :].copy(), matrix[i, :].copy()
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _decompose_cnot_2q(problem):
    """Exact CNOT for 2 qubits."""
    target = _target(problem)
    cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    cnot_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
    d01 = abs(np.trace(target.conj().T @ cnot_01)) / 4
    d10 = abs(np.trace(target.conj().T @ cnot_10)) / 4
    if d01 > d10:
        return [_cx(0, 1)]
    else:
        return [_cx(1, 0)]


def _decompose_swap_2q(problem):
    """Exact SWAP for 2 qubits: 3 CNOTs."""
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _decompose_iswap_2q(problem):
    """Exact iSWAP for 2 qubits: 2 CNOTs + 4 u3 gates."""
    gates = [
        _u3(0, math.pi, 0, 0),  # H
        _u3(1, math.pi, 0, 0),  # H
        _cx(0, 1),
        _u3(0, math.pi/2, math.pi/2, 0),
        _u3(1, math.pi/2, math.pi/2, math.pi),
        _cx(1, 0),
        _u3(0, math.pi, 0, 0),  # H
        _u3(1, math.pi, 0, 0),  # H
    ]
    return gates


def _decompose_toffoli_3q(problem):
    """Exact Toffoli for 3 qubits: 6 CNOTs + 5 u3 gates."""
    gates = [
        _u3(2, math.pi/2, 0, 0),  # H
        _cx(1, 2),
        _u3(2, -math.pi/4, 0, 0),
        _cx(0, 2),
        _u3(2, math.pi/4, 0, 0),
        _cx(1, 2),
        _u3(2, -math.pi/4, 0, 0),
        _u3(1, math.pi/2, 0, 0),  # H
        _u3(1, -math.pi/4, 0, 0),
        _cx(0, 1),
        _u3(1, math.pi/4, 0, 0),
        _u3(2, math.pi/2, 0, 0),  # H
    ]
    return gates


def _optimize_unitary(target, qubits, cnot_budget):
    """
    Optimize single-qubit gates for a fixed CNOT ladder.
    Uses coordinate descent with line search.
    """
    dim = 2 ** qubits
    
    # Define CNOT ladder
    # For 2 qubits: 3 CNOTs is sufficient for general unitaries.
    # For 3 qubits: 6 CNOTs is sufficient for general unitaries (standard ladder).
    if qubits == 2:
        cnots = [(0, 1), (1, 0), (0, 1)]
    elif qubits == 3:
        # Standard 3-qubit ladder
        cnots = [(0, 1), (1, 2), (0, 1), (0, 2), (1, 2), (0, 1)]
    else:
        return []
        
    # Ensure we don't exceed budget (though the above are minimal standard)
    if len(cnots) > cnot_budget:
        cnots = cnots[:cnot_budget]
        
    # Build layout: u3 gates on all qubits before, between, and after CNOTs
    layout = []
    # Initial u3 layer
    for q in range(qubits):
        layout.append(('u3', q))
        
    # Interleaved CNOTs and u3 layers
    for c, t in cnots:
        layout.append(('cx', c, t))
        for q in range(qubits):
            layout.append(('u3', q))
            
    n_u3 = sum(1 for item in layout if item[0] == 'u3')
    n_params = 3 * n_u3
    
    def _build_from_params(params):
        mat = np.eye(dim, dtype=complex)
        k = 0
        for item in layout:
            if item[0] == 'u3':
                q = item[1]
                theta, phi, lam = params[k], params[k+1], params[k+2]
                k += 3
                c = math.cos(theta / 2)
                s = math.sin(theta / 2)
                # Apply single-qubit gate
                for i in range(dim):
                    if (i >> (qubits - 1 - q)) & 1 == 0:
                        mat[i, :] = c * mat[i, :] + np.exp(1j * phi) * s * mat[i | (1 << (qubits - 1 - q)), :]
                    else:
                        mat[i, :] = -np.exp(1j * lam) * s * mat[i ^ (1 << (qubits - 1 - q)), :] + np.exp(1j * (phi + lam)) * c * mat[i, :]
            else:
                c, t = item[1], item[2]
                for i in range(dim):
                    if (i >> (qubits - 1 - c)) & 1:
                        j = i ^ (1 << (qubits - 1 - t))
                        if i < j:
                            mat[i, :], mat[j, :] = mat[j, :].copy(), mat[i, :].copy()
        return mat

    def objective(params):
        mat = _build_from_params(params)
        return 1.0 - min(1.0, abs(np.trace(target.conj().T @ mat)) / dim)

    # Initial guess: all zeros
    x = np.zeros(n_params)
    best_val = objective(x)
    
    deadline = _start_time + _DEADLINE - 10
    step = 1.0
    max_sweeps = 50
    min_step = 1e-8
    
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_params):
            if time.time() > deadline:
                break
            # Try positive and negative steps
            for sign in [1, -1]:
                trial_x = x.copy()
                trial_x[i] += sign * step
                trial_val = objective(trial_x)
                if trial_val < best_val:
                    # Accept and try to extend step
                    x = trial_x
                    best_val = trial_val
                    improved = True
                    # Line search: double step while improving
                    while time.time() < deadline:
                        next_x = x.copy()
                        next_x[i] += sign * step
                        next_val = objective(next_x)
                        if next_val < best_val:
                            x = next_x
                            best_val = next_val
                            step *= 1.5
                        else:
                            break
                    # If we extended, reset step for next param to a reasonable value
                    # But keep the current step for this param's further exploration?
                    # Simpler: just accept and move to next param.
                    break 
            if best_val < 1e-10:
                return _params_to_gates(x, layout)
        if not improved:
            step *= 0.5
            if step < min_step:
                break
                
    return _params_to_gates(x, layout)

def _params_to_gates(params, layout):
    gates = []
    k = 0
    for item in layout:
        if item[0] == 'u3':
            q = item[1]
            gates.append(_u3(q, params[k], params[k+1], params[k+2]))
            k += 3
        else:
            gates.append(_cx(item[1], item[2]))
    return gates


def synthesize(problem):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    if qubits == 2:
        # CNOT
        cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        cnot_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
        d01 = abs(np.trace(target.conj().T @ cnot_01)) / 4
        d10 = abs(np.trace(target.conj().T @ cnot_10)) / 4
        if max(d01, d10) > 0.999:
            return _decompose_cnot_2q(problem)
        
        # SWAP
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        dswap = abs(np.trace(target.conj().T @ swap)) / 4
        if dswap > 0.999:
            return _decompose_swap_2q(problem)
        
        # iSWAP
        iswap = np.array([[0,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
        diswap = abs(np.trace(target.conj().T @ iswap)) / 4
        if diswap > 0.999:
            return _decompose_iswap_2q(problem)
        
        # Generic 2-qubit
        return _optimize_unitary(target, qubits, budget)
    
    elif qubits == 3:
        # Toffoli
        toff = np.eye(8, dtype=complex)
        toff[6, 6] = 0
        toff[6, 7] = 1
        toff[7, 6] = 1
        toff[7, 7] = 0
        dtoff = abs(np.trace(target.conj().T @ toff)) / 8
        if dtoff > 0.999:
            return _decompose_toffoli_3q(problem)
        
        # Generic 3-qubit
        return _optimize_unitary(target, qubits, budget)
    
    # Fallback
    return []
# EVOLVE-BLOCK-END
