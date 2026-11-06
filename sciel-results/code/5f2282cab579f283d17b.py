"""Quantum circuit synthesis using KAK decomposition for 2-qubit and
structured decomposition for 3-qubit gates, with exact known-gate shortcuts."""
import math
import time
import numpy as np

_DEADLINE = 240.0
_start_time = None


def _target(problem):
    """Extract the target unitary as a complex numpy array."""
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    """Build the 2x2 u3 rotation matrix."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _u3_from_matrix(M):
    """Decompose a 2x2 unitary into u3 parameters (theta, phi, lam)."""
    if M.shape != (2, 2):
        raise ValueError("Expected 2x2 matrix")
    # Handle the case where M[0,0] is near zero
    if abs(M[0, 0]) < 1e-12:
        theta = math.pi
        phi = np.angle(M[1, 0])
        lam = -np.angle(M[0, 1]) - math.pi
        phi = (phi + math.pi) % (2 * math.pi)
        lam = (lam + math.pi) % (2 * math.pi)
        return theta, phi, lam
    else:
        theta = 2.0 * math.acos(max(-1.0, min(1.0, abs(M[0, 0]))))
        phi = np.angle(M[1, 0]) - np.angle(M[0, 0])
        lam = -np.angle(M[0, 1]) - np.angle(M[0, 0])
        phi = (phi + math.pi) % (2 * math.pi)
        lam = (lam + math.pi) % (2 * math.pi)
        return theta, phi, lam


def _build_unitary(gates, qubits):
    """Rebuild the unitary from a gate list."""
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            U = operator @ U
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            U = U[index ^ (bit << (qubits - 1 - target)), :]
    return U


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target^dagger U)| / dim."""
    dim = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / dim)


def _is_circuit_close(target, gates, qubits, tol=1e-10):
    """Check if a circuit is close to the target."""
    matrix = _build_unitary(gates, qubits)
    return _distance(target, matrix) < tol


def _cnot_gate(c, t):
    return ["cx", c, t]


def _u3_gate(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _exact_cnot_2q():
    """Exact CNOT on 2 qubits: control=0, target=1."""
    return [
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 1, math.pi / 2, -math.pi / 2, 0.0],
    ]


def _exact_swap_2q():
    """Exact SWAP on 2 qubits."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _exact_iswap_2q():
    """
    iSWAP on 2 qubits: [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
    Uses 2 CNOTs.
    """
    # iSWAP = (S ⊗ S) (H ⊗ H) CNOT(0,1) (H ⊗ H)
    # H = u3(pi/2, 0, pi/2)
    # S = u3(0, 0, pi/2)
    gates = [
        ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["cx", 0, 1],
        ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
        ["u3", 0, 0.0, 0.0, math.pi / 2],
        ["u3", 1, 0.0, 0.0, math.pi / 2],
    ]
    return gates


def _exact_toffoli_3q():
    """
    Toffoli on 3 qubits: controls 0,1 target 2.
    Uses 6 CNOTs.
    """
    # Standard decomposition using 6 CNOTs
    # H on q2
    # CNOT(1,2)
    # T† on q2
    # CNOT(0,2)
    # T on q1
    # T on q2
    # CNOT(1,2)
    # T† on q1
    # T† on q2
    # CNOT(0,2)
    # T on q1
    # H on q2
    # (T and T† need to be decomposed into u3)
    # T = u3(0, 0, pi/4)
    # T† = u3(0, 0, -pi/4)
    gates = [
        ["u3", 2, math.pi / 2, 0.0, math.pi / 2],  # H
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, -math.pi / 4],  # T†
        ["cx", 0, 2],
        ["u3", 1, 0.0, 0.0, math.pi / 4],  # T
        ["u3", 2, 0.0, 0.0, math.pi / 4],  # T
        ["cx", 1, 2],
        ["u3", 1, 0.0, 0.0, -math.pi / 4],  # T†
        ["u3", 2, 0.0, 0.0, -math.pi / 4],  # T†
        ["cx", 0, 2],
        ["u3", 1, 0.0, 0.0, math.pi / 4],  # T
        ["u3", 2, math.pi / 2, 0.0, math.pi / 2],  # H
    ]
    return gates


def _kak_2qubit(U, budget=3):
    """
    Decompose a 2-qubit unitary using KAK/standard decomposition.
    Uses up to 3 CNOTs.
    U = (A ⊗ B) CNOT(0,1) (C ⊗ D) CNOT(0,1) (E ⊗ F) CNOT(0,1) (G ⊗ H)
    We'll use a 3-CNOT circuit and optimize the single-qubit parameters.
    """
    # For 2-qubit gates, the standard decomposition uses 3 CNOTs:
    # U = (L1 ⊗ L2) CNOT CNOT CNOT (R1 ⊗ R2)
    # But the canonical form is:
    # U = (A ⊗ B) (H⊗H) CNOT (H⊗H) (C ⊗ D) (H⊗H) CNOT (H⊗H) (E ⊗ F) (H⊗H) CNOT (H⊗H) (G ⊗ H)
    # This is complex. Let's use a simpler approach:
    # U = (A ⊗ B) CNOT(0,1) (C ⊗ D) CNOT(0,1) (E ⊗ F) CNOT(0,1) (G ⊗ H)
    # We'll optimize the 24 single-qubit parameters.
    
    # Actually, for efficiency, let's use the fact that we can peel off single-qubit gates
    # and use a known structure.
    
    # Simpler: use the 3-CNOT decomposition with coordinate descent on the single-qubit params.
    # Layout: u3 on both qubits, CNOT, u3 on both, CNOT, u3 on both, CNOT, u3 on both
    layout = [
        ["u3", 0, 0.0, 0.0, 0.0],
        ["u3", 1, 0.0, 0.0, 0.0],
        ["cx", 0, 1],
        ["u3", 0, 0.0, 0.0, 0.0],
        ["u3", 1, 0.0, 0.0, 0.0],
        ["cx", 0, 1],
        ["u3", 0, 0.0, 0.0, 0.0],
        ["u3", 1, 0.0, 0.0, 0.0],
        ["cx", 0, 1],
        ["u3", 0, 0.0, 0.0, 0.0],
        ["u3", 1, 0.0, 0.0, 0.0],
    ]
    
    n_params = 24  # 8 u3 gates * 3 params
    rng = np.random.RandomState(42)
    
    def build_circuit(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k + 1], params[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initial guess: all zeros
    best_params = np.zeros(n_params)
    best_dist = _distance(U, _build_unitary(build_circuit(best_params), 2))
    
    # Coordinate descent
    for sweep in range(50):
        step = 0.5 / (1 + sweep * 0.1)
        improved = False
        for i in range(n_params):
            for trial in (best_params[i] + step, best_params[i] - step, 
                          best_params[i] + rng.uniform(-1, 1) * step * 2):
                trial_params = best_params.copy()
                trial_params[i] = trial
                trial_dist = _distance(U, _build_unitary(build_circuit(trial_params), 2))
                if trial_dist < best_dist:
                    best_params, best_dist = trial_params, trial_dist
                    improved = True
        if not improved and sweep > 10:
            break
    
    return build_circuit(best_params), best_dist


def _optimize_3qubit(U, budget=6):
    """
    Optimize a 3-qubit circuit using coordinate descent.
    Uses a fixed CNOT topology and optimizes single-qubit parameters.
    """
    # Layout for 3 qubits with 6 CNOTs
    # CNOT topology: (0,1), (1,2), (0,1), (1,2), (0,1), (1,2)
    cnot_pairs = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
    
    layout = []
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    for c, t in cnot_pairs:
        layout.append(["cx", c, t])
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    n_params = 3 * 3 * 7  # 7 layers of u3 * 3 qubits * 3 params = 63
    rng = np.random.RandomState(43)
    
    def build_circuit(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k + 1], params[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best_params = np.zeros(n_params)
    best_dist = _distance(U, _build_unitary(build_circuit(best_params), 3))
    
    max_sweeps = 30
    for sweep in range(max_sweeps):
        step = 0.5 / (1 + sweep * 0.1)
        improved = False
        for i in range(n_params):
            for trial in (best_params[i] + step, best_params[i] - step,
                          best_params[i] + rng.uniform(-1, 1) * step * 2):
                trial_params = best_params.copy()
                trial_params[i] = trial
                trial_dist = _distance(U, _build_unitary(build_circuit(trial_params), 3))
                if trial_dist < best_dist:
                    best_params, best_dist = trial_params, trial_dist
                    improved = True
        if not improved and sweep > 10:
            break
    
    return build_circuit(best_params), best_dist


def synthesize(problem):
    """
    Synthesize a quantum circuit for the given target unitary.
    Returns a list of gates.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "").lower()
    
    # Check for known gates
    if qubits == 2:
        if "cnot" in name and budget >= 1:
            gates = _exact_cnot_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
        if "iswap" in name or "i_swp" in name and budget >= 2:
            gates = _exact_iswap_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
        if "swap" in name and budget >= 3:
            gates = _exact_swap_2q()
            if _is_circuit_close(target, gates, qubits):
                return gates
    
    if qubits == 3:
        if "toffoli" in name or "ccx" in name or "ccnot" in name and budget >= 6:
            gates = _exact_toffoli_3q()
            if _is_circuit_close(target, gates, qubits):
                return gates
    
    # Generic synthesis
    if qubits == 2:
        gates, dist = _kak_2qubit(target, budget)
    else:
        gates, dist = _optimize_3qubit(target, budget)
    
    return gates
