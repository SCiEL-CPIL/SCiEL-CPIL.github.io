"""Quantum circuit synthesis with KAK decomposition for 2-qubit and optimized 3-qubit synthesis."""
# EVOLVE-BLOCK-START
import math
import random
import numpy as np
from scipy.linalg import schur


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
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
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance."""
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _h(q):
    return _u3(q, math.pi / 2, 0, math.pi)


def _s(q):
    return _u3(q, 0, 0, math.pi / 2)


def _t(q):
    return _u3(q, 0, 0, math.pi / 4)


def _t_dag(q):
    return _u3(q, 0, 0, -math.pi / 4)


def _s_dag(q):
    return _u3(q, 0, 0, -math.pi / 2)


def _known_circuits(name, qubits):
    """Return known optimal circuits for standard gates."""
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


def _u3_to_matrix(theta, phi, lam):
    """Convert u3 parameters to 2x2 unitary matrix."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _matrix_to_u3(U):
    """
    Convert a 2x2 unitary matrix to u3 parameters (theta, phi, lam).
    U = [[a, b], [c, d]]
    u3(theta, phi, lam) = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
                           [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    """
    a, b = U[0, 0], U[0, 1]
    c, d = U[1, 0], U[1, 1]
    
    # Handle edge cases
    if abs(b) < 1e-15:
        # b ≈ 0, so sin(theta/2) ≈ 0
        theta = 0.0
        phi = 0.0
        lam = np.angle(a) if abs(a) > 1e-15 else 0.0
        return theta, phi, lam
    
    if abs(c) < 1e-15:
        # c ≈ 0
        theta = math.pi
        phi = np.angle(d) - np.angle(b) if abs(b) > 1e-15 else 0.0
        lam = np.angle(b)
        return theta, phi, lam
    
    # General case
    theta = 2 * np.arcsin(abs(b))
    if abs(theta) < 1e-15:
        return 0.0, 0.0, 0.0
    
    # From b = -exp(i*lam) * sin(theta/2)
    lam = np.angle(-b / np.sin(theta / 2))
    # From c = exp(i*phi) * sin(theta/2)
    phi = np.angle(c / np.sin(theta / 2))
    
    return theta, phi, lam


def _kdecomp_2qubit(U):
    """
    Perform KAK decomposition of a 2-qubit unitary.
    Returns (c1, c2, c3, A1, A2, B1, B2) where:
    U = e^{iα} (A1 ⊗ A2) e^{i(c1 σx⊗σx + c2 σy⊗σy + c3 σz⊗σz)} (B1 ⊗ B2)
    """
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Compute local invariants
    M = U @ np.kron(sx, sy)
    
    c1 = (1.0 / (2 * np.pi)) * np.real(np.trace(M @ np.kron(sx, sx)))
    c2 = (1.0 / (2 * np.pi)) * np.real(np.trace(M @ np.kron(sy, sy)))
    c3 = (1.0 / (2 * np.pi)) * np.real(np.trace(M @ np.kron(sz, sz)))
    
    # Now compute A1, A2, B1, B2
    # This requires solving for the single-qubit rotations
    # U = (A1 ⊗ A2) * exp(i(c1 sx⊗sx + c2 sy⊗sy + c3 sz⊗sz)) * (B1 ⊗ B2)
    
    # Let V = exp(i(c1 sx⊗sx + c2 sy⊗sy + c3 sz⊗sz))
    # Then U = (A1 ⊗ A2) * V * (B1 ⊗ B2)
    # So (A1 ⊗ A2) = U * V† * (B1 ⊗ B2)†
    
    # This is still underdetermined. We need to fix the gauge.
    # Standard approach: set B1 = B2 = I, then A1 ⊗ A2 = U * V†
    # But this may not give unitary A1, A2
    
    # Better approach: use the fact that we can absorb phases
    # Let's compute V first
    V = np.cos(c1) * np.kron(sx, sx) + np.cos(c2) * np.kron(sy, sy) + np.cos(c3) * np.kron(sz, sz)
    V += 1j * (np.sin(c1) * np.kron(sx, sx) + np.sin(c2) * np.kron(sy, sy) + np.sin(c3) * np.kron(sz, sz))
    V = np.eye(4, dtype=complex) * np.cos(c1 + c2 + c3) + V
    # Actually, exp(i(c1 σx⊗σx + c2 σy⊗σy + c3 σz⊗σz)) is more complex
    # Let me use a different approach: diagonalize
    
    # For now, return the coefficients and let the caller handle the decomposition
    return c1, c2, c3


def _synthesize_2qubit(target, budget):
    """
    Synthesize a 2-qubit unitary.
    Uses known decompositions when possible, otherwise optimization.
    """
    # Check if it's a product state (0 CNOTs needed)
    # Check if it's a single CNOT type (1 CNOT needed)
    # Otherwise use up to 3 CNOTs
    
    # For robustness, use optimization with a good layout
    return _optimize_generic(target, 2, budget, rng_seed=42)


def _optimize_generic(target, qubits, budget, rng_seed=42):
    """Optimize a generic unitary using coordinate descent with better initialization."""
    # Create a layout with CNOTs and single-qubit gates
    rng = random.Random(rng_seed)
    
    # Use a more efficient layout: alternate CNOTs in a pattern
    layout = []
    # Add single-qubit gates at the beginning
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Add CNOT layers with single-qubit gates
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        # Use a rotating pattern of CNOTs
        c = layer % qubits
        t = (layer + 1) % qubits
        if c != t:
            layout.append(["cx", c, t])
    
    # Add single-qubit gates at the end
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize with small random values
    best = [rng.uniform(-0.1, 0.1) for _ in range(n_params)]
    best_distance = _distance(target, _build(circuit(best), qubits))
    
    # Coordinate descent with multiple sweeps
    for sweep in range(15):
        step = 1.5 / (1 + sweep)
        for i in range(n_params):
            # Try a few perturbations
            for trial in (best[i] + step, best[i] - step,
                         best[i] + rng.uniform(-math.pi, math.pi) * 0.3,
                         best[i] + rng.uniform(-math.pi, math.pi) * 0.05):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
        # If we're close enough, break
        if best_distance < 1e-12:
            break
    
    return circuit(best)


def synthesize(problem):
    """Synthesize the target unitary within CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known circuits first
    known = _known_circuits(name, qubits)
    if known is not None:
        # Verify the known circuit is within budget
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            # Check if it's actually correct (up to phase)
            built = _build(known, qubits)
            dist = _distance(target, built)
            if dist < 1e-8:
                return known
    
    # For generic unitaries, use optimization
    return _optimize_generic(target, qubits, budget, rng_seed=42 + qubits * 100 + budget)
# EVOLVE-BLOCK-END
