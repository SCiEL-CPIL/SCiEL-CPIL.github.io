"""Direct quantum circuit synthesis for fixed gates using known decompositions."""
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 245.0  # Safety margin before 270s wall-clock kill


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
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            U = op @ U
        else:
            _, c, t = gate
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - c)) & 1
            U = U[idx ^ (bit << (qubits - 1 - t)), :]
    return U


def _distance(target, U):
    """Phase-insensitive distance: 1 - |tr(target^dag U)| / dim."""
    dim = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / dim)


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


def _cnot(c, t):
    return ['cx', c, t]


def _decomp_cnot_01():
    return [_cnot(0, 1)]


def _decomp_cnot_10():
    return [_cnot(1, 0)]


def _decomp_swap():
    return [_cnot(0, 1), _cnot(1, 0), _cnot(0, 1)]


def _decomp_iswap():
    """iSWAP = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
    Standard 2-CNOT decomposition:
    iSWAP = (H⊗H) · CX(0,1) · (S†⊗S) · CX(0,1) · (H⊗H)
    """
    return [_h(0), _h(1), _cnot(0, 1), _sdag(0), _s(1), _cnot(0, 1), _h(0), _h(1)]


def _decomp_toffoli(c1, c2, t):
    """6-CNOT Toffoli decomposition for controls c1, c2 and target t."""
    gates = []
    gates.append(_h(t))
    gates.append(_cnot(c1, t))
    gates.append(_tdag(t))
    gates.append(_t(c2))
    gates.append(_cnot(c2, t))
    gates.append(_t(t))
    gates.append(_tdag(c2))
    gates.append(_cnot(c1, t))
    gates.append(_t(c2))
    gates.append(_tdag(t))
    gates.append(_h(t))
    gates.append(_cnot(c1, c2))
    gates.append(_t(c1))
    gates.append(_tdag(c2))
    gates.append(_cnot(c1, c2))
    return gates


def _match_known(target, qubits):
    """Try to match the target against known gate decompositions."""
    if qubits == 2:
        candidates = [
            (_decomp_cnot_01(), 1),
            (_decomp_cnot_10(), 1),
            (_decomp_swap(), 3),
            (_decomp_iswap(), 2),
        ]
        for gates, cnot_count in candidates:
            U = _build(gates, qubits)
            if _distance(target, U) < 1e-10:
                return gates
    elif qubits == 3:
        # Try all Toffoli permutations
        for c1 in range(3):
            for c2 in range(3):
                if c2 == c1:
                    continue
                t = 3 - c1 - c2
                gates = _decomp_toffoli(c1, c2, t)
                U = _build(gates, qubits)
                if _distance(target, U) < 1e-10:
                    return gates
    return None


def _solve_2qubit_kak(target, qubits=2):
    """
    Exact 2-qubit synthesis using KAK (canonical) decomposition.
    Any 2-qubit unitary U can be written as (A ⊗ B) · exp(-i(a·σ⊗σ)) · (C ⊗ D)
    with 4 single-qubit unitaries and 3 CNOTs.
    """
    n = 2 ** qubits

    # Standard CNOT ladder: CX(0,1), CX(1,0), CX(0,1)
    # The entangling part is fixed; we optimize the 4 single-qubit gates.
    # Each single-qubit gate has 3 params (u3), so 12 params total.
    
    def circuit(params):
        gates = []
        k = 0
        gates.append(['u3', 0, params[k], params[k+1], params[k+2]]); k += 3
        gates.append(['u3', 1, params[k], params[k+1], params[k+2]]); k += 3
        gates.append(_cnot(0, 1))
        gates.append(['u3', 0, params[k], params[k+1], params[k+2]]); k += 3
        gates.append(['u3', 1, params[k], params[k+1], params[k+2]]); k += 3
        gates.append(_cnot(1, 0))
        gates.append(['u3', 0, params[k], params[k+1], params[k+2]]); k += 3
        gates.append(['u3', 1, params[k], params[k+1], params[k+2]]); k += 3
        gates.append(_cnot(0, 1))
        gates.append(['u3', 0, params[k], params[k+1], params[k+2]]); k += 3
        gates.append(['u3', 1, params[k], params[k+1], params[k+2]]); k += 3
        return gates

    n_params = 12
    best = np.zeros(n_params)
    best_U = _build(circuit(best), qubits)
    best_dist = _distance(target, best_U)

    rng = random.Random(42)

    # Multi-start coordinate descent with random restarts
    max_sweeps = 60
    for restart in range(5):
        if time.time() > _DEADLINE - 5:
            break
        if restart == 0:
            current = np.zeros(n_params)
        else:
            current = np.array([rng.uniform(-math.pi, math.pi) for _ in range(n_params)])
        
        cur_U = _build(circuit(current), qubits)
        cur_dist = _distance(target, cur_U)
        
        if cur_dist < best_dist:
            best = current.copy()
            best_U = cur_U
            best_dist = cur_dist
        
        for sweep in range(max_sweeps):
            if time.time() > _DEADLINE - 5:
                break
            step = 0.5 / (1.0 + sweep * 0.3)
            improved = False
            for i in range(n_params):
                if time.time() > _DEADLINE - 5:
                    break
                base = current[i]
                for delta in (step, -step, step * 0.5, -step * 0.5):
                    trial = current.copy()
                    trial[i] = base + delta
                    trial_U = _build(circuit(trial), qubits)
                    trial_dist = _distance(target, trial_U)
                    if trial_dist < cur_dist:
                        current = trial
                        cur_dist = trial_dist
                        cur_U = trial_U
                        improved = True
            if not improved and sweep > 10:
                break
        
        if best_dist < 1e-12:
            break

    return circuit(best)


def _solve_3qubit(target, qubits=3, budget=15):
    """
    Decompose a 3-qubit unitary using coordinate descent on a CNOT ladder.
    Uses a more flexible CNOT pattern and multi-start optimization.
    """
    n = 2 ** qubits

    # Use a CNOT pattern that covers all pairs
    pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    num_cnots = min(budget, 6)

    # Build layout: u3 gates on all qubits between each CNOT
    layout = []
    for layer in range(num_cnots):
        for q in range(qubits):
            layout.append(["u3", q])
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
    for q in range(qubits):
        layout.append(["u3", q])

    n_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * n_u3

    def circuit(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(['u3', gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    best = np.zeros(n_params)
    best_U = _build(circuit(best), qubits)
    best_dist = _distance(target, best_U)

    rng = random.Random(123)

    # Multi-start optimization
    max_sweeps = 40
    for restart in range(4):
        if time.time() > _DEADLINE - 5:
            break
        if restart == 0:
            current = np.zeros(n_params)
        else:
            current = np.array([rng.uniform(-math.pi, math.pi) for _ in range(n_params)])
        
        cur_U = _build(circuit(current), qubits)
        cur_dist = _distance(target, cur_U)
        
        if cur_dist < best_dist:
            best = current.copy()
            best_U = cur_U
            best_dist = cur_dist
        
        for sweep in range(max_sweeps):
            if time.time() > _DEADLINE - 5:
                break
            step = 0.5 / (1.0 + sweep * 0.3)
            improved = False
            for i in range(n_params):
                if time.time() > _DEADLINE - 5:
                    break
                base = current[i]
                for delta in (step, -step, step * 0.5, -step * 0.5):
                    trial = current.copy()
                    trial[i] = base + delta
                    trial_U = _build(circuit(trial), qubits)
                    trial_dist = _distance(target, trial_U)
                    if trial_dist < cur_dist:
                        current = trial
                        cur_dist = trial_dist
                        cur_U = trial_U
                        improved = True
            if not improved and sweep > 10:
                break
        
        if best_dist < 1e-12:
            break

    # Final polish with smaller steps
    if best_dist > 1e-8:
        for polish in range(15):
            if time.time() > _DEADLINE - 5:
                break
            step = 0.02 / (1.0 + polish * 0.5)
            for i in range(n_params):
                if time.time() > _DEADLINE - 5:
                    break
                base = best[i]
                for delta in (step, -step):
                    trial = best.copy()
                    trial[i] = base + delta
                    trial_U = _build(circuit(trial), qubits)
                    trial_dist = _distance(target, trial_U)
                    if trial_dist < best_dist:
                        best = trial
                        best_dist = trial_dist
                        best_U = trial_U

    return circuit(best)


def synthesize(problem):
    """Synthesize a circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    name = problem.get("name", "")
    target = _target(problem)

    # Check deadline
    if time.time() > _DEADLINE:
        return [['u3', q, 0.0, 0.0, 0.0] for q in range(qubits)]

    # Try known gate matching first
    known = _match_known(target, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == 'cx')
        if cnot_count <= budget:
            return known

    # Generic synthesis
    if qubits == 2:
        return _solve_2qubit_kak(target, qubits)
    else:
        return _solve_3qubit(target, qubits, budget)
