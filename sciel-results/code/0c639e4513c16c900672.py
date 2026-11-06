"""Quantum circuit synthesis with exact algebraic decompositions and optimized generic synthesis."""
import math
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 255.0
_START_TIME = time.time()

def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build_fast(gates, qubits):
    """Build the unitary matrix from a gate list using in-place operations."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Apply single-qubit gate using in-place row operations
            for i in range(n):
                if (i >> (qubits - 1 - q)) & 1:
                    # qubit is 1
                    row0 = i ^ (1 << (qubits - 1 - q))
                    # U[i] = single[1,0]*U[row0] + single[1,1]*U[i]
                    # U[row0] = single[0,0]*U[row0] + single[0,1]*U[i]
                    old_row = U[i].copy()
                    U[i] = single[1, 0] * U[row0] + single[1, 1] * old_row
                    U[row0] = single[0, 0] * U[row0] + single[0, 1] * old_row
        else:
            _, control, target = gate
            # CNOT: swap rows where control bit is 1
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            U = U[index ^ (bit << (qubits - 1 - target)), :]
    return U

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _h(q):
    return ['u3', q, math.pi / 2, 0, math.pi]

def _s(q):
    return ['u3', q, 0, 0, math.pi / 2]

def _sdag(q):
    return ['u3', q, 0, 0, -math.pi / 2]

def _t(q):
    return ['u3', q, math.pi / 4, 0, 0]

def _tdag(q):
    return ['u3', q, -math.pi / 4, 0, 0]

def _exact_cnot_01():
    return [['cx', 0, 1]]

def _exact_cnot_10():
    return [['cx', 1, 0]]

def _exact_iswap_2qubit():
    """iSWAP decomposition with 2 CNOTs."""
    gates = []
    gates.append(_h(0))
    gates.append(_h(1))
    gates.append(['cx', 0, 1])
    gates.append(_sdag(0))
    gates.append(_s(1))
    gates.append(['cx', 0, 1])
    gates.append(_h(0))
    gates.append(_h(1))
    return gates

def _exact_swap_2qubit():
    """SWAP decomposition with 3 CNOTs."""
    return [['cx', 0, 1], ['cx', 1, 0], ['cx', 0, 1]]

def _exact_toffoli_3qubit(c1, c2, t):
    """Toffoli decomposition with 6 CNOTs for controls c1,c2 and target t."""
    gates = []
    gates.append(_h(t))
    gates.append(['cx', c1, t])
    gates.append(_tdag(t))
    gates.append(_t(c2))
    gates.append(['cx', c2, t])
    gates.append(_t(t))
    gates.append(_tdag(c2))
    gates.append(['cx', c1, t])
    gates.append(_t(c2))
    gates.append(_tdag(t))
    gates.append(_h(t))
    gates.append(['cx', c1, c2])
    gates.append(_t(c1))
    gates.append(_tdag(c2))
    gates.append(['cx', c1, c2])
    return gates

def _is_known_gate(name, target, qubits):
    """Check if target matches a known gate and return exact decomposition."""
    name_lower = name.lower().replace('-', '').replace('_', '')
    
    if qubits == 2:
        if 'cnot' in name_lower or 'cx' in name_lower:
            ref = _build_fast(_exact_cnot_01(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_cnot_01()
            ref = _build_fast(_exact_cnot_10(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_cnot_10()
        
        if 'iswap' in name_lower:
            ref = _build_fast(_exact_iswap_2qubit(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_iswap_2qubit()
        
        if 'swap' in name_lower:
            ref = _build_fast(_exact_swap_2qubit(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_swap_2qubit()
    
    if qubits == 3:
        if 'toffoli' in name_lower or 'ccnot' in name_lower:
            for c1 in range(3):
                for c2 in range(3):
                    if c2 == c1:
                        continue
                    t = 3 - c1 - c2
                    ref = _build_fast(_exact_toffoli_3qubit(c1, c2, t), 3)
                    if _distance(target, ref) < 1e-10:
                        return _exact_toffoli_3qubit(c1, c2, t)
    
    return None

def _synthesize_2qubit(target, budget):
    """Synthesize a generic 2-qubit unitary."""
    if budget < 3:
        # Fallback to coordinate descent if budget is too low for KAK
        return _generic_synthesize_cd(target, 2, budget)
    
    # Layout: U3s, CNOT, U3s, CNOT, U3s, CNOT, U3s
    # 4 layers of U3s (8 gates) and 3 CNOTs
    # Total U3s: 8 * 3 = 24 parameters
    
    def build_circuit(params):
        gates = []
        # Layer 0
        for q in range(2):
            gates.append(['u3', q, params[q*3], params[q*3+1], params[q*3+2]])
        gates.append(['cx', 0, 1])
        # Layer 1
        for q in range(2):
            gates.append(['u3', q, params[6+q*3], params[6+q*3+1], params[6+q*3+2]])
        gates.append(['cx', 1, 0])
        # Layer 2
        for q in range(2):
            gates.append(['u3', q, params[12+q*3], params[12+q*3+1], params[12+q*3+2]])
        gates.append(['cx', 0, 1])
        # Layer 3
        for q in range(2):
            gates.append(['u3', q, params[18+q*3], params[18+q*3+1], params[18+q*3+2]])
        return gates

    def objective(params):
        if _check_deadline():
            return 1.0
        gates = build_circuit(params)
        U = _build_fast(gates, 2)
        return _distance(target, U)

    # Initial guess: zeros
    x0 = np.zeros(24)
    
    # Use Nelder-Mead for robustness
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        best_params = res.x
        best_dist = res.fun
    except Exception:
        best_params = x0
        best_dist = objective(x0)
    
    # Polish with coordinate descent
    best_params = list(best_params)
    step = 0.01
    for _ in range(10):
        if _check_deadline():
            break
        for i in range(24):
            for delta in [step, -step]:
                trial = best_params[:]
                trial[i] += delta
                val = objective(trial)
                if val < best_dist:
                    best_params = trial
                    best_dist = val
            step *= 0.5
            if step < 1e-6:
                break

    return build_circuit(best_params)

def _generic_synthesize_cd(target, qubits, budget):
    """Coordinate descent for generic unitaries (used for 3-qubit and fallback)."""
    n = 2 ** qubits
    
    # Build layout
    if qubits == 3:
        pairs = [(0, 1), (1, 2), (2, 0)]
    else:
        pairs = [(0, 1), (1, 0)]
        
    layout = []
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    for i in range(budget):
        layout.append(["cx", *pairs[i % len(pairs)]])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    num_params = 3 * num_u3
    
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
        if _check_deadline():
            return 1.0
        gates = circuit(values)
        U = _build_fast(gates, qubits)
        return _distance(target, U)

    best = [0.0] * num_params
    best_distance = objective(best)
    
    # Coordinate descent
    max_sweeps = 20
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(num_params):
            if _check_deadline():
                break
            trials = [best[i] + step, best[i] - step,
                      best[i] + step * 0.5, best[i] - step * 0.5]
            for trial in trials:
                values = best[:]
                values[i] = trial
                value = objective(values)
                if value < best_distance:
                    best, best_distance = values, value
    
    # Polish
    for polish in range(5):
        if _check_deadline():
            break
        step = 0.01 / (1 + polish)
        for i in range(num_params):
            if _check_deadline():
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = objective(values)
                if value < best_distance:
                    best, best_distance = values, value
    
    return circuit(best)

def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check for known gates first
    known = _is_known_gate(name, target, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == 'cx')
        if cnot_count <= budget:
            return known
    
    # Generic synthesis
    if qubits == 2:
        return _synthesize_2qubit(target, budget)
    else:
        return _generic_synthesize_cd(target, qubits, budget)
