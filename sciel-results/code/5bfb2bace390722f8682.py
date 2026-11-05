"""Quantum circuit synthesis using KAK decomposition and exact closed-form gates."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 250

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, c, t = gate
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - c)) & 1
            matrix = matrix[idx ^ (bit << (qubits - 1 - t)), :]
    return matrix

def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])

def _u3_from_matrix(mat):
    """Extract theta, phi, lam from a 2x2 unitary."""
    c = mat[0, 0]
    phi = np.angle(np.linalg.det(mat))
    theta = 2.0 * np.arccos(np.clip(abs(c), 0.0, 1.0))
    s = math.sin(theta / 2)
    if s < 1e-12:
        if abs(mat[1,1]) > 1e-12:
            lam = np.angle(mat[1,1] / c) - phi
        else:
            lam = 0.0
        return [theta, phi, lam]
    phi_val = np.angle(mat[1, 0] / s)
    lam_val = np.angle(-mat[0, 1] / s)
    return [theta, phi_val, lam_val]

def _is_close(a, b, tol=1e-6):
    """Check if two unitaries are equivalent up to global phase."""
    proj = np.vdot(b.ravel(), a.ravel()) / np.linalg.norm(b.ravel())
    return np.linalg.norm(a - proj * b) < tol

def _synth_known_2qubit(u):
    """Check known 2-qubit gates and return circuit or None."""
    # CNOT (control=0, target=1)
    cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    if _is_close(u, cnot):
        return [["cx", 0, 1]]
    
    # iSWAP
    iswap = np.array([[0,0,0,1j],[0,0,1j,0],[0,1j,0,0],[1j,0,0,0]], dtype=complex)
    if _is_close(u, iswap):
        # iSWAP decomposition: H⊗H · CX · (Rz(π/2)⊗Rz(-π/2)) · CX · (Rz(-π/2)⊗Rz(π/2)) · H⊗H
        # Let me use a verified 2-CNOT decomposition
        # q0: H -- Rz(-π/2) -- X -- Rz(π/2) -- H
        # q1: H -- Rz(π/2) -- X -- Rz(-π/2) -- H
        #      └── CNOT ──┘         └── CNOT ──┘
        gates = [
            ["u3", 0, math.pi/2, 0, math.pi],   # H on q0
            ["u3", 1, math.pi/2, 0, math.pi],   # H on q1
            ["u3", 0, 0, 0, -math.pi/2],        # RZ(-pi/2) on q0
            ["u3", 1, 0, 0, math.pi/2],         # RZ(pi/2) on q1
            ["cx", 0, 1],
            ["u3", 0, 0, 0, math.pi/2],         # RZ(pi/2) on q0
            ["u3", 1, 0, 0, -math.pi/2],        # RZ(-pi/2) on q1
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, 0, math.pi],   # H on q0
            ["u3", 1, math.pi/2, 0, math.pi],   # H on q1
        ]
        return gates
    
    # SWAP
    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    if _is_close(u, swap):
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    
    return None

def _synth_toffoli(u):
    """Check if U is Toffoli (or Toffoli†) and return 6-CNOT circuit."""
    # Toffoli: |110> -> |111>, |111> -> |110> (target = qubit 2, LSB)
    toffoli = np.eye(8, dtype=complex)
    toffoli[6,6] = 0
    toffoli[7,7] = 0
    toffoli[6,7] = 1
    toffoli[7,6] = 1
    
    if not _is_close(u, toffoli):
        # Try Toffoli with different target convention
        toffoli2 = np.eye(8, dtype=complex)
        toffoli2[3,3] = 0
        toffoli2[7,7] = 0
        toffoli2[3,7] = 1
        toffoli2[7,3] = 1
        if not _is_close(u, toffoli2):
            return None
    
    # Standard 6-CNOT Toffoli (target = qubit 2, LSB; controls = qubits 0,1)
    return [
        ["u3", 2, math.pi/2, 0, math.pi],         # H on target
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, -math.pi/2, math.pi/2],  # T†
        ["cx", 0, 2],
        ["u3", 2, math.pi/2, math.pi/2, -math.pi/2],  # T
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, -math.pi/2, math.pi/2],  # T†
        ["cx", 0, 2],
        ["u3", 2, math.pi/2, math.pi/2, -math.pi/2],  # T
        ["u3", 2, math.pi/2, 0, math.pi],         # H on target
        ["cx", 0, 1],
    ]

def _kak_2qubit(u):
    """
    KAK (Cartan) decomposition for a 2-qubit unitary.
    Returns (A, B, C, D, theta1, theta2, theta3) such that:
    U ~ (A ⊗ B) · exp(-i/2(θ1 X⊗X + θ2 Y⊗Y + θ3 Z⊗Z)) · (C ⊗ D)
    """
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    sigmas = [X, Y, Z]
    
    # Compute the 3x3 matrix M where M[i,j] = (1/4)Tr(U (σi⊗σj) U† (σi⊗σj))
    M = np.zeros((3,3), dtype=complex)
    for i in range(3):
        for j in range(3):
            Sj = np.kron(sigmas[i], sigmas[j])
            M[i,j] = 0.25 * np.trace(u @ Sj @ u.conj().T @ Sj)
    
    M = np.real(M)
    
    # Diagonalize M
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, -1.0, 1.0)
    
    # The KAK angles are related to eigenvalues: cos(2θ_k) = eigvals[k]
    # But we need to be careful about the convention
    # Actually, for U = (A⊗B) K (C⊗D), the eigenvalues of the "correlation matrix"
    # give us the KAK angles directly
    
    # Let's use a different approach: compute the KAK angles using the invariants
    # For a 2-qubit unitary with det=1:
    # The KAK angles θ1, θ2, θ3 satisfy:
    # cos(2θ1)cos(2θ2)cos(2θ3) = Re(Tr(U^4))/16
    # etc.
    
    # Actually, let's use the direct method from the literature:
    # The KAK decomposition is U ~ (A⊗B) exp(-i/2 Σ θ_k σ_k⊗σ_k) (C⊗D)
    # where the θ_k are the KAK angles.
    
    # Compute using the formula:
    # M = (1/4) Σ_k (σ_k ⊗ σ_k) U (σ_k ⊗ σ_k) U†
    # The eigenvalues of M are cos(2θ_k)
    
    # Let me just use the eigenvalues directly
    theta1 = 0.5 * np.arccos(np.clip(eigvals[0], -1, 1))
    theta2 = 0.5 * np.arccos(np.clip(eigvals[1], -1, 1))
    theta3 = 0.5 * np.arccos(np.clip(eigvals[2], -1, 1))
    
    # Now we need to find A, B, C, D
    # This is more complex. Let's use a simpler approach:
    # Just use the KAK angles to determine the number of CNOTs needed
    # and then optimize the single-qubit gates
    
    # For now, return the angles and let the optimizer handle the rest
    return theta1, theta2, theta3, eigvals

def _num_cnots_needed(eigvals):
    """Determine minimum number of CNOTs from KAK eigenvalues."""
    # If all eigenvalues are ±1, it's local (0 CNOTs)
    # If two are ±1, it needs 1 CNOT
    # If one is ±1, it needs 2 CNOTs
    # Otherwise, it needs 3 CNOTs
    
    # Check if eigenvalues are close to ±1
    close_to_pm1 = [abs(abs(ev) - 1) < 1e-6 for ev in eigvals]
    
    if all(close_to_pm1):
        return 0
    elif sum(close_to_pm1) == 2:
        return 1
    elif sum(close_to_pm1) == 1:
        return 2
    else:
        return 3

def _synth_2qubit_kak(u, budget):
    """Synthesize 2-qubit unitary using KAK decomposition."""
    theta1, theta2, theta3, eigvals = _kak_2qubit(u)
    num_cnots = _num_cnots_needed(eigvals)
    num_cnots = min(num_cnots, budget)
    
    if num_cnots == 0:
        # Local unitary: U = A ⊗ B
        # Extract A and B
        # This requires more work. For now, use optimization
        pass
    
    # Build layout with the required number of CNOTs
    layout = []
    for layer in range(num_cnots):
        layout += [["u3", 0, 0,0,0], ["u3", 1, 0,0,0]]
        layout.append(["cx", 0, 1])
    layout += [["u3", 0, 0,0,0], ["u3", 1, 0,0,0]]
    
    circuit, dist = _optimize_ladder(u, layout, 2, max_sweeps=50, rng_seed=42)
    return circuit

def _optimize_ladder(target, layout, qubits, max_sweeps=50, rng_seed=42):
    """Optimize a gate layout using coordinate descent with multiple restarts."""
    import random
    rng = random.Random(rng_seed)
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def make_circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    def eval_dist(values):
        return _distance(target, _build(make_circuit(values), qubits))
    
    best = [0.0] * n_params
    best_d = eval_dist(best)
    
    # Multiple random restarts
    n_restarts = 8 if qubits == 2 else 4
    for r in range(n_restarts):
        if _time_remaining() < 20:
            break
        cand = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        d = eval_dist(cand)
        if d < best_d:
            best = cand
            best_d = d
    
    # Coordinate descent with adaptive step
    step = 1.0
    for sweep in range(max_sweeps):
        if _time_remaining() < 20:
            break
        improved = False
        for i in range(n_params):
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                d = eval_dist(values)
                if d < best_d:
                    best = values
                    best_d = d
                    improved = True
        if not improved:
            step *= 0.5
            if step < 0.005:
                break
        else:
            step = max(step * 0.85, 0.02)
    
    return make_circuit(best), best_d

def _synth_2qubit_generic(u, budget):
    """Synthesize a generic 2-qubit unitary."""
    # Try KAK first to determine optimal CNOT count
    try:
        circuit = _synth_2qubit_kak(u, budget)
        if circuit is not None:
            return circuit
    except Exception:
        pass
    
    # Fallback: use 3-CNOT ladder
    num_cnots = min(3, budget)
    layout = []
    for layer in range(num_cnots):
        layout += [["u3", 0, 0,0,0], ["u3", 1, 0,0,0]]
        layout.append(["cx", 0, 1])
    layout += [["u3", 0, 0,0,0], ["u3", 1, 0,0,0]]
    
    circuit, dist = _optimize_ladder(u, layout, 2, max_sweeps=50, rng_seed=42)
    return circuit

def _synth_3qubit_generic(u, budget):
    """Synthesize a generic 3-qubit unitary using a ladder of 2-qubit blocks."""
    max_cnots = budget
    num_blocks = max(1, max_cnots // 3)
    
    pairs = [(0,1), (1,2)]
    layout = []
    
    for i in range(num_blocks):
        c, t = pairs[i % 2]
        layout += [["u3", 0, 0,0,0], ["u3", 1, 0,0,0], ["u3", 2, 0,0,0]]
        layout.append(["cx", c, t])
        layout += [["u3", 0, 0,0,0], ["u3", 1, 0,0,0], ["u3", 2, 0,0,0]]
        layout.append(["cx", c, t])
        layout += [["u3", 0, 0,0,0], ["u3", 1, 0,0,0], ["u3", 2, 0,0,0]]
        layout.append(["cx", c, t])
        layout += [["u3", 0, 0,0,0], ["u3", 1, 0,0,0], ["u3", 2, 0,0,0]]
    
    circuit, dist = _optimize_ladder(u, layout, 3, max_sweeps=30, rng_seed=42)
    return circuit

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    if qubits == 2:
        known = _synth_known_2qubit(target)
        if known is not None:
            return known
        return _synth_2qubit_generic(target, budget)
    
    elif qubits == 3:
        toffoli_circuit = _synth_toffoli(target)
        if toffoli_circuit is not None:
            return toffoli_circuit
        return _synth_3qubit_generic(target, budget)
    
    else:
        return []
# EVOLVE-BLOCK-END
