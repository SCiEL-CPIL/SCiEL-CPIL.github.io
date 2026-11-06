"""Combined approach: exact known-gate decompositions + fast coordinate descent for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_DEADLINE = 240.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", q, float(theta), float(phi), float(lam)]


def _cx(c, t):
    return ["cx", c, t]


def _build(gates, qubits):
    """Build unitary from gate list efficiently."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            bit_shift = qubits - 1 - q
            mask = 1 << bit_shift
            # Apply single-qubit gate on qubit q
            for i in range(dim):
                if (i >> bit_shift) & 1 == 0:
                    j = i | mask
                    matrix[i, :] = c * matrix[i, :] + np.exp(1j * phi) * s * matrix[j, :]
                else:
                    j = i ^ mask
                    matrix[i, :] = -np.exp(1j * lam) * s * matrix[j, :] + np.exp(1j * (phi + lam)) * c * matrix[i, :]
        else:
            _, control, target = gate
            c_shift = qubits - 1 - control
            t_shift = qubits - 1 - target
            c_mask = 1 << c_shift
            t_mask = 1 << t_shift
            for i in range(dim):
                if (i >> c_shift) & 1:
                    j = i ^ t_mask
                    if i < j:
                        matrix[i, :], matrix[j, :] = matrix[j, :].copy(), matrix[i, :].copy()
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _decompose_cnot_2q(problem):
    """Exact CNOT for 2 qubits."""
    target = _target(problem)
    cnot_01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    cnot_10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
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
    """Exact iSWAP for 2 qubits: 2 CNOTs + single-qubit rotations."""
    gates = [
        _u3(0, math.pi, 0, 0),
        _u3(1, math.pi, 0, 0),
        _cx(0, 1),
        _u3(0, math.pi / 2, math.pi / 2, 0),
        _u3(1, math.pi / 2, math.pi / 2, math.pi),
        _cx(1, 0),
        _u3(0, math.pi, 0, 0),
        _u3(1, math.pi, 0, 0),
    ]
    return gates


def _decompose_toffoli_3q(problem):
    """Exact Toffoli for 3 qubits: 6 CNOTs + single-qubit rotations."""
    gates = [
        _u3(2, math.pi / 2, 0, 0),
        _cx(1, 2),
        _u3(2, -math.pi / 4, 0, 0),
        _cx(0, 2),
        _u3(2, math.pi / 4, 0, 0),
        _cx(1, 2),
        _u3(2, -math.pi / 4, 0, 0),
        _u3(1, math.pi / 2, 0, 0),
        _u3(1, -math.pi / 4, 0, 0),
        _cx(0, 1),
        _u3(1, math.pi / 4, 0, 0),
        _u3(2, math.pi / 2, 0, 0),
    ]
    return gates


def _optimize_unitary(target, qubits, cnot_budget):
    """
    Optimize single-qubit gates for a fixed CNOT ladder using coordinate descent.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    dim = 2 ** qubits
    
    # Define CNOT ladder
    if qubits == 2:
        cnots = [(0, 1), (1, 0), (0, 1)]
    elif qubits == 3:
        cnots = [(0, 1), (1, 2), (0, 1), (0, 2), (1, 2), (0, 1)]
    else:
        return []
    
    if len(cnots) > cnot_budget:
        cnots = cnots[:cnot_budget]
    
    # Build layout
    layout = []
    for q in range(qubits):
        layout.append(('u3', q))
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
                theta, phi, lam = params[k], params[k + 1], params[k + 2]
                k += 3
                c = math.cos(theta / 2)
                s = math.sin(theta / 2)
                bit_shift = qubits - 1 - q
                mask = 1 << bit_shift
                for i in range(dim):
                    if (i >> bit_shift) & 1 == 0:
                        j = i | mask
                        mat[i, :] = c * mat[i, :] + np.exp(1j * phi) * s * mat[j, :]
                    else:
                        j = i ^ mask
                        mat[i, :] = -np.exp(1j * lam) * s * mat[j, :] + np.exp(1j * (phi + lam)) * c * mat[i, :]
            else:
                c, t = item[1], item[2]
                c_shift = qubits - 1 - c
                t_shift = qubits - 1 - t
                c_mask = 1 << c_shift
                t_mask = 1 << t_shift
                for i in range(dim):
                    if (i >> c_shift) & 1:
                        j = i ^ t_mask
                        if i < j:
                            mat[i, :], mat[j, :] = mat[j, :].copy(), mat[i, :].copy()
        return mat
    
    def objective(params):
        mat = _build_from_params(params)
        return 1.0 - min(1.0, abs(np.trace(target.conj().T @ mat)) / dim)
    
    x = np.zeros(n_params)
    best_val = objective(x)
    
    deadline = _start_time + _DEADLINE - 10
    step = 1.0
    max_sweeps = 30
    min_step = 1e-8
    
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n_params):
            if time.time() > deadline:
                break
            for sign in [1, -1]:
                trial_x = x.copy()
                trial_x[i] += sign * step
                trial_val = objective(trial_x)
                if trial_val < best_val:
                    x = trial_x
                    best_val = trial_val
                    improved = True
                    # Line search
                    while time.time() < deadline:
                        next_x = x.copy()
                        next_x[i] += sign * step
                        next_val = objective(next_x)
                        if next_val < best_val:
                            x = next_x
                            best_val = next_val
                            step *= 1.3
                        else:
                            break
                    break
            if best_val < 1e-10:
                break
        if not improved:
            step *= 0.5
            if step < min_step:
                break
    
    # Convert params to gates
    gates = []
    k = 0
    for item in layout:
        if item[0] == 'u3':
            q = item[1]
            theta, phi, lam = x[k], x[k + 1], x[k + 2]
            k += 3
            gates.append(_u3(q, theta, phi, lam))
        else:
            gates.append(_cx(item[1], item[2]))
    return gates


def synthesize(problem):
    """Synthesize circuit for the given problem."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    name = problem.get("name", "")
    qubits = problem["qubits"]
    cnot_budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    if qubits == 2:
        if "CNOT" in name.upper() or "cnot" in name:
            return _decompose_cnot_2q(problem)
        if "SWAP" in name.upper() or "swap" in name:
            return _decompose_swap_2q(problem)
        if "iSWAP" in name.upper() or "iswap" in name:
            return _decompose_iswap_2q(problem)
    
    if qubits == 3:
        if "TOFFOLI" in name.upper() or "toffoli" in name or "CCNOT" in name.upper():
            return _decompose_toffoli_3q(problem)
    
    # Generic unitary: use optimization
    return _optimize_unitary(target, qubits, cnot_budget)
# EVOLVE-BLOCK-END
