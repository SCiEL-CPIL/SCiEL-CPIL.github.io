"""Exact decompositions for known gates + optimized synthesis for generic targets."""
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


def _apply_u3(mat, q, qubits, theta, phi, lam):
    dim = 2 ** qubits
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    for i in range(dim):
        bit = (i >> (qubits - 1 - q)) & 1
        if bit == 0:
            j = i | (1 << (qubits - 1 - q))
            new_row = c * mat[i, :].copy() + np.exp(1j * phi) * s * mat[j, :].copy()
            mat[i, :] = new_row
        else:
            j = i ^ (1 << (qubits - 1 - q))
            new_row = -np.exp(1j * lam) * s * mat[j, :].copy() + np.exp(1j * (phi + lam)) * c * mat[i, :].copy()
            mat[i, :] = new_row


def _apply_cx(mat, control, target, qubits):
    dim = 2 ** qubits
    for i in range(dim):
        if (i >> (qubits - 1 - control)) & 1:
            j = i ^ (1 << (qubits - 1 - target))
            if i < j:
                mat[i, :], mat[j, :] = mat[j, :].copy(), mat[i, :].copy()


def _build(gates, qubits):
    dim = 2 ** qubits
    mat = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            _apply_u3(mat, q, qubits, theta, phi, lam)
        else:
            _, c, t = gate
            _apply_cx(mat, c, t, qubits)
    return mat


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_close(target, ref):
    dim = target.shape[0]
    return abs(np.trace(target.conj().T @ ref)) / dim > 0.999


def _synth_cnot_2q(problem):
    target = _target(problem)
    cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    if _is_close(target, cnot_01):
        return [_cx(0, 1)]
    else:
        return [_cx(1, 0)]


def _synth_swap_2q(problem):
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _synth_iswap_2q(problem):
    gates = [
        _u3(0, math.pi, 0, 0),
        _u3(1, math.pi, 0, 0),
        _cx(0, 1),
        _u3(0, math.pi/2, math.pi/2, 0),
        _u3(1, math.pi/2, math.pi/2, math.pi),
        _cx(1, 0),
        _u3(0, math.pi, 0, 0),
        _u3(1, math.pi, 0, 0),
    ]
    return gates


def _synth_toffoli_3q(problem):
    gates = [
        _u3(2, math.pi/2, 0, 0),
        _cx(1, 2),
        _u3(2, -math.pi/4, 0, 0),
        _cx(0, 2),
        _u3(2, math.pi/4, 0, 0),
        _cx(1, 2),
        _u3(2, -math.pi/4, 0, 0),
        _u3(1, math.pi/2, 0, 0),
        _u3(1, -math.pi/4, 0, 0),
        _cx(0, 1),
        _u3(1, math.pi/4, 0, 0),
        _u3(2, math.pi/2, 0, 0),
    ]
    return gates


def _synth_generic_2q(target):
    layout = [
        ('u3', 0), ('u3', 1), ('cx', 0, 1),
        ('u3', 0), ('u3', 1), ('cx', 1, 0),
        ('u3', 0), ('u3', 1), ('cx', 0, 1),
        ('u3', 0), ('u3', 1),
    ]
    n_params = 24
    
    def build(params):
        mat = np.eye(4, dtype=complex)
        k = 0
        for item in layout:
            if item[0] == 'u3':
                q = item[1]
                t, p, l = params[k], params[k+1], params[k+2]
                k += 3
                _apply_u3(mat, q, 2, t, p, l)
            else:
                _apply_cx(mat, item[1], item[2], 2)
        return mat
    
    def objective(params):
        mat = build(params)
        return 1.0 - min(1.0, abs(np.trace(target.conj().T @ mat)) / 4)
    
    best_params = np.zeros(n_params)
    best_val = objective(best_params)
    
    deadline = _start_time + _DEADLINE - 10
    step = 1.0
    for sweep in range(50):
        if time.time() > deadline:
            break
        step = 1.0 / (1 + sweep * 0.3)
        improved = False
        for i in range(n_params):
            if time.time() > deadline:
                break
            for trial in (best_params[i] + step, best_params[i] - step):
                tp = best_params.copy()
                tp[i] = trial
                val = objective(tp)
                if val < best_val:
                    best_params = tp
                    best_val = val
                    improved = True
        if not improved and step < 0.005:
            break
    
    gates = []
    k = 0
    for item in layout:
        if item[0] == 'u3':
            q = item[1]
            gates.append(_u3(q, best_params[k], best_params[k+1], best_params[k+2]))
            k += 3
        else:
            gates.append(_cx(item[1], item[2]))
    return gates


def _synth_generic_3q(target, budget):
    pairs = [(0,1), (1,2), (0,2)]
    layout = []
    for layer in range(budget):
        layout.append(('u3', 0))
        layout.append(('u3', 1))
        layout.append(('u3', 2))
        layout.append(('cx', pairs[layer % 3][0], pairs[layer % 3][1]))
    layout.append(('u3', 0))
    layout.append(('u3', 1))
    layout.append(('u3', 2))
    
    n_u3 = sum(1 for item in layout if item[0] == 'u3')
    n_params = 3 * n_u3
    
    def build(params):
        mat = np.eye(8, dtype=complex)
        k = 0
        for item in layout:
            if item[0] == 'u3':
                q = item[1]
                t, p, l = params[k], params[k+1], params[k+2]
                k += 3
                _apply_u3(mat, q, 3, t, p, l)
            else:
                _apply_cx(mat, item[1], item[2], 3)
        return mat
    
    def objective(params):
        mat = build(params)
        return 1.0 - min(1.0, abs(np.trace(target.conj().T @ mat)) / 8)
    
    best_params = np.zeros(n_params)
    best_val = objective(best_params)
    
    deadline = _start_time + _DEADLINE - 10
    step = 1.0
    for sweep in range(15):
        if time.time() > deadline:
            break
        step = 1.0 / (1 + sweep * 0.3)
        improved = False
        for i in range(n_params):
            if time.time() > deadline:
                break
            for trial in (best_params[i] + step, best_params[i] - step):
                tp = best_params.copy()
                tp[i] = trial
                val = objective(tp)
                if val < best_val:
                    best_params = tp
                    best_val = val
                    improved = True
        if not improved and step < 0.005:
            break
    
    gates = []
    k = 0
    for item in layout:
        if item[0] == 'u3':
            q = item[1]
            gates.append(_u3(q, best_params[k], best_params[k+1], best_params[k+2]))
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
    
    if qubits == 2:
        cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        cnot_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
        if _is_close(target, cnot_01) or _is_close(target, cnot_10):
            return _synth_cnot_2q(problem)
        
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _is_close(target, swap):
            return _synth_swap_2q(problem)
        
        iswap = np.array([[0,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,0]], dtype=complex)
        if _is_close(target, iswap):
            return _synth_iswap_2q(problem)
        
        return _synth_generic_2q(target)
    
    elif qubits == 3:
        toff = np.eye(8, dtype=complex)
        toff[6, 6] = 0
        toff[6, 7] = 1
        toff[7, 6] = 1
        toff[7, 7] = 0
        if _is_close(target, toff):
            return _synth_toffoli_3q(problem)
        
        return _synth_generic_3q(target, budget)
    
    return []
# EVOLVE-BLOCK-END
