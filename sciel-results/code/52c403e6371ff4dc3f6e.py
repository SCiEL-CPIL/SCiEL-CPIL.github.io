"""Quantum circuit synthesis for fixed unitaries within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

# Deadline guard: 270s wall-clock kill, keep 30s safety margin
_DEADLINE = time.time() + 240.0


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cnot_matrix(control, target, qubits):
    n = 2 ** qubits
    m = np.eye(n, dtype=complex)
    index = np.arange(n)
    bit = (index >> (qubits - 1 - control)) & 1
    perm = index ^ (bit << (qubits - 1 - target))
    m[perm, :] = np.eye(n, dtype=complex)
    return m


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            m = _cnot_matrix(control, target, qubits)
            matrix = m @ matrix
    return matrix


def _distance(target, matrix):
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _known_syntheses(problem):
    """Return a fixed gate list for well-known gates if the target matches."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    target = _target(problem)
    
    # CNOT
    if name == "CNOT" and qubits == 2:
        return [["cx", 0, 1]]
    
    # SWAP
    if name == "SWAP" and qubits == 2:
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    
    # iSWAP
    if name == "iSWAP" and qubits == 2:
        # iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
        # Standard decomposition:
        # CNOT(0,1), H(1), CNOT(0,1), Rz(pi/2)(0), Rz(pi/2)(1), CNOT(0,1), H(1), CNOT(0,1)
        # But we need to check if this matches the target exactly.
        # Actually, let's just use a known decomposition.
        # iSWAP = (I⊗H) CNOT (Rz(π/2)⊗Rz(π/2)) CNOT (I⊗H) ... this is getting complicated.
        # Let's try a simpler approach: use the known 4-CNOT decomposition or just return a heuristic.
        # For iSWAP, a common decomposition is:
        # u3(0, pi/2, 0, pi/2) on q0, u3(0, pi/2, 0, pi/2) on q1, cx(0,1), u3(0, pi/2, pi/2, pi/2) on q0, u3(0, pi/2, pi/2, pi/2) on q1, cx(0,1), u3(0, pi/2, 0, pi/2) on q0, u3(0, pi/2, 0, pi/2) on q1
        # This might not be exact. Let's just return None and let the optimizer handle it.
        pass
    
    # Toffoli
    if name == "Toffoli" and qubits == 3:
        # Toffoli = CCX(0,1,2)
        # Standard decomposition uses 6 CNOTs:
        # H(2), T†(2), CX(1,2), T(1), T†(2), CX(0,2), T(0), T(2), CX(1,2), T†(1), T(2), CX(0,2), T†(0), T†(2), H(2)
        # T = Rz(π/4) = u3(π/2, 0, π/4)
        # T† = Rz(-π/4) = u3(π/2, 0, -π/4)
        # H = u3(π/2, 0, π)
        gates = [
            ["u3", 2, math.pi/2, 0, math.pi],  # H
            ["u3", 2, math.pi/2, 0, -math.pi/4],  # T†
            ["cx", 1, 2],
            ["u3", 1, math.pi/2, 0, math.pi/4],  # T
            ["u3", 2, math.pi/2, 0, -math.pi/4],  # T†
            ["cx", 0, 2],
            ["u3", 0, math.pi/2, 0, math.pi/4],  # T
            ["u3", 2, math.pi/2, 0, math.pi/4],  # T
            ["cx", 1, 2],
            ["u3", 1, math.pi/2, 0, -math.pi/4],  # T†
            ["u3", 2, math.pi/2, 0, math.pi/4],  # T
            ["cx", 0, 2],
            ["u3", 0, math.pi/2, 0, -math.pi/4],  # T†
            ["u3", 2, math.pi/2, 0, -math.pi/4],  # T†
            ["u3", 2, math.pi/2, 0, math.pi],  # H
        ]
        return gates
    
    return None


def synthesize(problem):
    """Fixed CNOT ladder with single-qubit rotations tuned by coordinate descent."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try known syntheses first
    known = _known_syntheses(problem)
    if known is not None:
        # Verify it's correct
        U = _build(known, qubits)
        d = _distance(target, U)
        if d < 1e-9:
            return known
    
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = np.random.RandomState(4242 + qubits)
    
    # Build a layout with CNOTs and u3 slots
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Count u3 slots
    u3_count = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * u3_count
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize with random values
    best = rng.uniform(-math.pi, math.pi, size=n_params)
    best_distance = _distance(target, _build(circuit(best), qubits))
    
    # Coordinate descent with adaptive step size
    start_time = time.time()
    for sweep in range(20):
        if time.time() > _DEADLINE - 10:
            break
        step = 0.5 / (1 + sweep)
        for i in range(n_params):
            if time.time() > _DEADLINE - 10:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
    
    return circuit(best)
# EVOLVE-BLOCK-END
