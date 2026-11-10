"""Quantum circuit synthesis with CNOT budget constraints."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = None
_TIME_MARGIN = 30.0  # seconds safety margin before 270s kill


def _time_remaining():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 270.0 - _TIME_MARGIN
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list (kept for known-gate verification)."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _build_fast(params, qubits, cnot_positions, cnot_controls, cnot_targets, u3_qubits):
    """Fast circuit builder. params is a flat array of [theta,phi,lam] per u3 slot."""
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    
    n_u3 = len(u3_qubits)
    n_cnot = len(cnot_positions)
    
    # Pre-compute CNOT permutation indices
    cnot_perms = []
    for ctrl, tgt in zip(cnot_controls, cnot_targets):
        index = np.arange(n)
        bit = (index >> (qubits - 1 - ctrl)) & 1
        perm = index ^ (bit << (qubits - 1 - tgt))
        cnot_perms.append(perm)
    
    # Pre-compute identity matrices for kron
    eye2 = np.eye(2, dtype=complex)
    
    u3_idx = 0
    for ci in range(n_cnot + 1):
        if ci < n_cnot:
            n_u3_before = cnot_positions[ci] - (cnot_positions[ci-1] if ci > 0 else 0)
            for _ in range(n_u3_before):
                q = u3_qubits[u3_idx]
                theta = params[3 * u3_idx]
                phi = params[3 * u3_idx + 1]
                lam = params[3 * u3_idx + 2]
                c = math.cos(theta / 2)
                s = math.sin(theta / 2)
                e_lam = np.exp(1j * lam)
                e_phi = np.exp(1j * phi)
                e_philam = np.exp(1j * (phi + lam))
                single = np.array([[c, -e_lam * s],
                                   [e_phi * s, e_philam * c]])
                op = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    op = np.kron(op, single if k == q else eye2)
                mat = op @ mat
                u3_idx += 1
            mat = mat[cnot_perms[ci], :]
        else:
            while u3_idx < n_u3:
                q = u3_qubits[u3_idx]
                theta = params[3 * u3_idx]
                phi = params[3 * u3_idx + 1]
                lam = params[3 * u3_idx + 2]
                c = math.cos(theta / 2)
                s = math.sin(theta / 2)
                e_lam = np.exp(1j * lam)
                e_phi = np.exp(1j * phi)
                e_philam = np.exp(1j * (phi + lam))
                single = np.array([[c, -e_lam * s],
                                   [e_phi * s, e_philam * c]])
                op = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    op = np.kron(op, single if k == q else eye2)
                mat = op @ mat
                u3_idx += 1
    
    return mat


def _known_decompositions(problem):
    """Return exact gate lists for known gates, or None."""
    name = problem["name"]
    qubits = problem["qubits"]
    target = _target(problem)
    
    # CNOT gate
    if name == "CNOT" and qubits == 2:
        return [["cx", 0, 1]]
    
    # iSWAP gate
    if name == "iSWAP" and qubits == 2:
        # iSWAP = |0><0| ⊗ |0><0| + |1><1| ⊗ |1><1| + i|0><1| ⊗ |1><0| + i|1><0| ⊗ |0><1|
        # Standard decomposition:
        # H(0) H(1)
        # CNOT(0,1)
        # u3(0, pi/2, -pi/2, pi/2)  # = i*H*Z*H type rotation
        # u3(1, pi/2, -pi/2, pi/2)
        # CNOT(1,0)
        # H(0) H(1)
        # Actually, let's use the standard:
        # iSWAP = (I⊗H) CNOT (I⊗S†) (I⊗H) (H⊗I) CNOT (S⊗I) (H⊗I)
        # Simpler: use known decomposition
        # iSWAP = diag(1,1,i,i) * SWAP * diag(1,1,1,-1) ... let me think
        # Standard: iSWAP = (H⊗H) CNOT (H⊗I) (S†⊗I) (H⊗H) CNOT (I⊗S) (H⊗H)
        # Actually the cleanest: 
        # iSWAP = (I⊗H) CNOT (I⊗S) CNOT (I⊗H) with some phase
        # Let me just verify: the standard decomposition is
        # q0: H --×-- S† --×-- H
        # q1: H --×-- S  --×-- H
        # That gives iSWAP up to global phase
        gates = [
            ["u3", 0, math.pi, 0.0, 0.0],  # H = u3(pi, 0, 0)
            ["u3", 1, math.pi, 0.0, 0.0],  # H
            ["cx", 0, 1],
            ["u3", 1, math.pi, 0.0, math.pi],  # S = u3(pi, 0, pi) -- wait S is diag(1,i)
            # S = u3(0, 0, pi) actually: u3(theta,phi,lambda) with theta=0 gives diag(1, e^{i lambda})
            # So S = u3(0, 0, pi/2)? No: u3(0,phi,lambda) = diag(1, e^{i lambda})
            # S = diag(1, i) = u3(0, 0, pi/2)
            ["u3", 1, 0.0, 0.0, math.pi / 2],  # S
            ["cx", 1, 0],
            ["u3", 0, math.pi, 0.0, 0.0],  # H
            ["u3", 1, math.pi, 0.0, 0.0],  # H
        ]
        # Verify this gives iSWAP up to global phase
        built = _build(gates, 2)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        # Try alternative decomposition
        gates2 = [
            ["u3", 0, math.pi, 0.0, 0.0],  # H
            ["u3", 1, math.pi, 0.0, 0.0],  # H
            ["cx", 0, 1],
            ["u3", 0, 0.0, 0.0, math.pi / 2],  # S on q0
            ["cx", 1, 0],
            ["u3", 0, math.pi, 0.0, 0.0],  # H
            ["u3", 1, math.pi, 0.0, 0.0],  # H
        ]
        built2 = _build(gates2, 2)
        dist2 = _distance(target, built2)
        if dist2 < 1e-10:
            return gates2
        return None
    
    # SWAP gate
    if name == "SWAP" and qubits == 2:
        gates = [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
        built = _build(gates, 2)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        return None
    
    # Toffoli gate
    if name == "Toffoli" and qubits == 3:
        # Standard Toffoli decomposition with 6 CNOTs
        # q2 (MSB) -- control, q1 -- control, q0 -- target
        # Standard: 
        # q2: H --×------×-- H
        # q1: H --×--×--×-- H  
        # q0: H --×--×--×-- H  ... no that's not right
        # The standard 6-CNOT Toffoli:
        # Let me use the known decomposition:
        # q2: --×--×--×--
        # q1: --×--×--×--  
        # q0: --×--×--×--
        # With single-qubit gates in between
        # Actually the standard is:
        # q2: H --×------×-- H
        # q1: H --×--×--×-- H
        # q0: H --×--×--×-- H
        # No, let me just use a known 6-CNOT decomposition
        gates = [
            ["u3", 0, math.pi, 0.0, 0.0],  # H on q0
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, -math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, 0.0, 0.0, math.pi / 4],
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, math.pi / 4],
            ["cx", 1, 2],
            ["u3", 2, 0.0, 0.0, -math.pi / 4],
            ["cx", 0, 1],
            ["u3", 0, math.pi, 0.0, 0.0],  # H on q0
        ]
        built = _build(gates, 3)
        dist = _distance(target, built)
        if dist < 1e-10:
            return gates
        return None
    
    return None


def _optimize_generic(problem, max_time=20.0):
    """Optimize a generic unitary using scipy L-BFGS-B with a structured ansatz."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 2 ** qubits
    
    # Build a structured ansatz: alternating layers of u3 on all qubits and one CNOT
    # CNOT pattern: cycle through pairs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Layout: for each CNOT layer, put u3 on all qubits before, then the CNOT
    # Final layer: u3 on all qubits
    u3_qubits = []  # qubit index for each u3 slot
    cnot_positions = []  # how many u3 slots before each cnot
    cnot_controls = []
    cnot_targets = []
    
    u3_count = 0
    for layer in range(budget):
        for q in range(qubits):
            u3_qubits.append(q)
            u3_count += 1
        cnot_positions.append(u3_count)
        cnot_controls.append(pairs[layer % len(pairs)][0])
        cnot_targets.append(pairs[layer % len(pairs)][1])
    
    # Final u3 layer
    for q in range(qubits):
        u3_qubits.append(q)
        u3_count += 1
    
    n_u3 = len(u3_qubits)
    n_params = 3 * n_u3
    
    # Objective function for scipy
    def objective(params):
        mat = _build_fast(params, qubits, cnot_positions, cnot_controls, cnot_targets, u3_qubits)
        return 1.0 - min(1.0, abs(np.trace(target.conj().T @ mat)) / n)
    
    # Try multiple random initializations
    best_params = np.zeros(n_params)
    best_dist = objective(best_params)
    
    start = time.time()
    max_time = min(max_time, _time_remaining() - 5.0)
    
    # Try several initializations
    n_starts = 3
    for s in range(n_starts):
        if time.time() - start > max_time or _time_remaining() < 5.0:
            break
        if s == 0:
            x0 = np.zeros(n_params)
        else:
            x0 = np.random.uniform(-math.pi, math.pi, size=n_params)
        
        try:
            from scipy.optimize import minimize
            result = minimize(objective, x0, method='L-BFGS-B',
                            options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12})
            if result.fun < best_dist:
                best_dist = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Build final gate list
    gates = []
    u3_idx = 0
    for ci in range(budget):
        n_u3_before = cnot_positions[ci] - (cnot_positions[ci-1] if ci > 0 else 0)
        for _ in range(n_u3_before):
            q = u3_qubits[u3_idx]
            gates.append(["u3", q, float(best_params[3*u3_idx]), float(best_params[3*u3_idx+1]), float(best_params[3*u3_idx+2])])
            u3_idx += 1
        gates.append(["cx", cnot_controls[ci], cnot_targets[ci]])
    
    # Remaining u3 slots
    while u3_idx < n_u3:
        q = u3_qubits[u3_idx]
        gates.append(["u3", q, float(best_params[3*u3_idx]), float(best_params[3*u3_idx+1]), float(best_params[3*u3_idx+2])])
        u3_idx += 1
    
    return gates


def synthesize(problem):
    """Synthesize a quantum circuit within the CNOT budget."""
    # First try known exact decompositions
    known = _known_decompositions(problem)
    if known is not None:
        return known
    
    # For generic unitaries, use optimization
    return _optimize_generic(problem)
# EVOLVE-BLOCK-END
