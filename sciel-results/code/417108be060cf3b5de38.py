"""Quantum circuit synthesis: analytic Barenco decomposition with fast evaluation."""
import time
import math
import random

import numpy as np
from scipy.optimize import minimize

_T0 = time.monotonic()
_HARD_STOP = 255.0


def _time_left():
    return _HARD_STOP - (time.monotonic() - _T0)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_matrix(params, layout, qubits):
    """Build unitary from parameters and layout efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    
    pi = 0
    for entry in layout:
        if entry[0] == "u3":
            q = entry[1]
            theta, phi, lam = params[pi], params[pi+1], params[pi+2]
            pi += 3
            single = _u3_matrix(theta, phi, lam)
            # Apply single-qubit gate
            if qubits == 2:
                if q == 0:
                    matrix = np.kron(single, eye2) @ matrix
                else:
                    matrix = np.kron(eye2, single) @ matrix
            else:
                if q == 0:
                    matrix = np.kron(np.kron(single, eye2), eye2) @ matrix
                elif q == 1:
                    matrix = np.kron(np.kron(eye2, single), eye2) @ matrix
                else:
                    matrix = np.kron(np.kron(eye2, eye2), single) @ matrix
        else:
            control, target = entry[1], entry[2]
            # Apply CNOT
            idx = np.arange(n)
            c_bit = (idx >> (qubits - 1 - control)) & 1
            t_bit_shift = (qubits - 1 - target)
            new_idx = idx ^ (c_bit << t_bit_shift)
            matrix = matrix[new_idx, :]
    
    return matrix


def _distance_from_params(params, layout, qubits, target):
    """Compute distance from parameters."""
    matrix = _build_matrix(params, layout, qubits)
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _u3(q, theta, phi, lam):
    return ["u3", int(q), float(theta), float(phi), float(lam)]


def _cx(control, target):
    return ["cx", int(control), int(target)]


def _h(q):
    return _u3(q, math.pi / 2, 0.0, math.pi)


def _s(q):
    return _u3(q, 0.0, 0.0, math.pi / 2)


def _t(q):
    return _u3(q, 0.0, 0.0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0.0, 0.0, -math.pi / 2)


def _known_circuits(name, qubits):
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


def _params_to_gates(params, layout):
    """Convert parameters and layout to gate list."""
    gates = []
    pi = 0
    for entry in layout:
        if entry[0] == "u3":
            q = entry[1]
            theta, phi, lam = params[pi], params[pi+1], params[pi+2]
            pi += 3
            gates.append(_u3(q, theta, phi, lam))
        else:
            gates.append(_cx(entry[1], entry[2]))
    return gates


def _optimize_2qubit(target, budget):
    """Optimize 2-qubit unitary with Barenco decomposition."""
    # Layout: (A⊗B) CX (C⊗D) CX (E⊗F) CX (G⊗H)
    n_cnots = min(budget, 3)
    
    layout = []
    for i in range(n_cnots + 1):
        layout.append(("u3", 0))
        layout.append(("u3", 1))
        if i < n_cnots:
            layout.append(("cx", 0, 1))
    
    n_u3 = sum(1 for e in layout if e[0] == "u3")
    n_params = 3 * n_u3
    
    best_dist = 1.0
    best_params = [0.0] * n_params
    rng = random.Random(42)
    
    n_restarts = min(10, max(3, int(_time_left() / 5)))
    
    for restart in range(n_restarts):
        if _time_left() < 15:
            break
        
        # Random initialization
        params0 = np.array([rng.uniform(-math.pi, math.pi) for _ in range(n_params)])
        
        def objective(x):
            return _distance_from_params(x, layout, 2, target)
        
        try:
            result = minimize(objective, params0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_dist:
                best_dist = result.fun
                best_params = result.x
        except Exception:
            pass
        
        if best_dist < 1e-10:
            break
    
    return _params_to_gates(best_params, layout)


def _optimize_3qubit(target, budget):
    """Optimize 3-qubit unitary."""
    # Use a fixed CNOT pattern that covers all pairs
    cnot_pairs = [(0, 1), (0, 2), (1, 2)]
    n_cnots = min(budget, 12)
    
    layout = []
    # Initial single-qubit gates
    for q in range(3):
        layout.append(("u3", q))
    
    # Add CNOTs and single-qubit gates
    for i in range(n_cnots):
        layout.append(("cx", cnot_pairs[i % 3][0], cnot_pairs[i % 3][1]))
        for q in range(3):
            layout.append(("u3", q))
    
    # Final single-qubit gates
    for q in range(3):
        layout.append(("u3", q))
    
    n_u3 = sum(1 for e in layout if e[0] == "u3")
    n_params = 3 * n_u3
    
    best_dist = 1.0
    best_params = [0.0] * n_params
    rng = random.Random(42)
    
    n_restarts = min(5, max(2, int(_time_left() / 10)))
    
    for restart in range(n_restarts):
        if _time_left() < 20:
            break
        
        # Random initialization
        params0 = np.array([rng.uniform(-math.pi, math.pi) for _ in range(n_params)])
        
        def objective(x):
            return _distance_from_params(x, layout, 3, target)
        
        try:
            result = minimize(objective, params0, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.fun < best_dist:
                best_dist = result.fun
                best_params = result.x
        except Exception:
            pass
        
        if best_dist < 1e-10:
            break
    
    return _params_to_gates(best_params, layout)


def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            # Verify it matches the target
            layout = []
            for g in known:
                if g[0] == "u3":
                    layout.append(("u3", g[1]))
                else:
                    layout.append(("cx", g[1], g[2]))
            params = []
            for g in known:
                if g[0] == "u3":
                    params.extend([g[2], g[3], g[4]])
            built = _build_matrix(params, layout, qubits)
            dist = _distance_from_params(params, layout, qubits, target)
            if dist < 1e-8:
                return known
    
    # Use optimization for generic or unknown gates
    if qubits == 2:
        result = _optimize_2qubit(target, budget)
    else:
        result = _optimize_3qubit(target, budget)
    
    # Verify budget constraints
    cnot_count = sum(1 for g in result if g[0] == "cx")
    if cnot_count > budget:
        count = 0
        truncated = []
        for g in result:
            if g[0] == "cx":
                if count < budget:
                    truncated.append(g)
                    count += 1
            else:
                truncated.append(g)
        result = truncated
    
    if len(result) > 400:
        result = result[:400]
    
    return result
