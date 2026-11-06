"""Quantum circuit synthesis with exact algebraic decompositions and optimized generic synthesis."""
import math
import time
import numpy as np

_DEADLINE = time.time() + 255.0


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
    """Build the unitary matrix from a gate list."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build full operator using kron
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye2)
            U = operator @ U
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
    # H on both qubits
    gates.append(_h(0))
    gates.append(_h(1))
    # CX(0,1)
    gates.append(['cx', 0, 1])
    # S† on qubit 0, S on qubit 1
    gates.append(_sdag(0))
    gates.append(_s(1))
    # CX(0,1)
    gates.append(['cx', 0, 1])
    # H on both qubits
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
            ref = _build(_exact_cnot_01(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_cnot_01()
            ref = _build(_exact_cnot_10(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_cnot_10()
        
        if 'iswap' in name_lower:
            ref = _build(_exact_iswap_2qubit(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_iswap_2qubit()
        
        if 'swap' in name_lower:
            ref = _build(_exact_swap_2qubit(), 2)
            if _distance(target, ref) < 1e-10:
                return _exact_swap_2qubit()
    
    if qubits == 3:
        if 'toffoli' in name_lower or 'ccnot' in name_lower:
            for c1 in range(3):
                for c2 in range(3):
                    if c2 == c1:
                        continue
                    t = 3 - c1 - c2
                    ref = _build(_exact_toffoli_3qubit(c1, c2, t), 3)
                    if _distance(target, ref) < 1e-10:
                        return _exact_toffoli_3qubit(c1, c2, t)
    
    return None


def _generic_synthesize(target, qubits, budget, name):
    """Synthesize generic unitary using coordinate descent on a CNOT+U3 layout."""
    n = 2 ** qubits
    rng_seed = 4242 + qubits * 100 + len(name) % 10000
    
    # Build a layout with exactly budget CNOTs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
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
    
    best = [0.0] * num_params
    best_distance = _distance(target, _build(circuit(best), qubits))
    
    # Coordinate descent with adaptive step size
    max_sweeps = 15
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 10:
            break
        step = 1.5 / (1 + sweep * 0.5)
        for i in range(num_params):
            if time.time() > _DEADLINE - 10:
                break
            trials = [best[i] + step, best[i] - step,
                      best[i] + step * 0.5, best[i] - step * 0.5]
            for trial in trials:
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
    
    # Final polish
    for polish in range(3):
        if time.time() > _DEADLINE - 5:
            break
        step = 0.1 / (1 + polish)
        for i in range(num_params):
            if time.time() > _DEADLINE - 5:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
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
    return _generic_synthesize(target, qubits, budget, name)
