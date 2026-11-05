"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 240.0  # Safety margin before 270s kill

def _time_left():
    return _DEADLINE - time.time()

def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)

def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)

def _build_fast(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            Uq = _u3_mat(theta, phi, lam)
            # Apply Uq on qubit q (qubit 0 = MSB)
            pos = q
            # Build full operator via kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                if k == q:
                    op = np.kron(op, Uq)
                else:
                    op = np.kron(op, np.eye(2, dtype=complex))
            U = op @ U
        else:
            _, c, t = gate
            # CNOT: control c, target t
            idx = np.arange(n)
            cbits = (idx >> (qubits - 1 - c)) & 1
            tshift = 1 << (qubits - 1 - t)
            perm = idx.copy()
            perm[cbits == 1] = idx[cbits == 1] ^ tshift
            U = U[perm, :]
    return U

def _distance(target, U):
    tr = np.trace(target.conj().T @ U)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])

def _u3_from_2x2(M):
    """Extract u3(theta, phi, lam) from a 2x2 unitary (up to global phase)."""
    # Remove global phase
    det = M[0,0] * M[1,1] - M[0,1] * M[1,0]
    if abs(det) > 1e-15:
        gp = det ** 0.5
        if abs(gp) > 1e-15:
            M = M / gp
    a, b = M[0,0], M[0,1]
    c, d = M[1,0], M[1,1]
    theta = 2 * math.acos(min(1.0, max(-1.0, abs(a))))
    if abs(a) < 1e-12:
        phi, lam = 0.0, 0.0
    else:
        phi = math.atan2(c.real, c.imag) if abs(c) > 1e-12 else 0.0
        # phi = arg(c) where c = e^{i phi} sin(theta/2)
        if abs(c) > 1e-12:
            phi = math.atan2(c.imag, c.real)
        else:
            phi = 0.0
        if abs(b) > 1e-12:
            lam = math.atan2(-b.imag, b.real)
            # b = -e^{i lam} sin(theta/2)
        else:
            lam = 0.0
    return theta, phi, lam

def _kak_decompose_2qubit(U2):
    """KAK decomposition of a 2-qubit unitary. Returns (K1, A, K2) where U2 = K1 @ expm(A) @ K2 up to global phase.
    A = a1*ZxI/2 + a2*IZ/2 + a3*ZZ/2 (or similar). For 3-CNOT circuit:
    K1, then 3 CNOTs with single-qubit rotations in between.
    Simplified: use the standard 3-CNOT decomposition."""
    # We'll use a simpler approach: find the optimal 3-CNOT decomposition
    # by parameterizing and optimizing. But for known gates, use exact circuits.
    pass

def _known_cnot():
    return [["u3", 1, 0.0, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, math.pi]]

def _known_swap():
    return [["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1]]

def _known_iswap():
    # iSWAP = 1/2 * [[0, i, 0, 0], [i, 0, 0, 0], [0, 0, 0, i], [0, 0, i, 0]]
    # Standard decomposition uses 2 CNOTs
    return [["u3", 0, math.pi/2, 0.0, 0.0],
            ["u3", 1, math.pi/2, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, math.pi, 0.0],
            ["u3", 1, math.pi/2, 0.0, math.pi],
            ["cx", 0, 1]]

def _known_toffoli():
    # Toffoli on 3 qubits: standard 6-CNOT circuit
    return [["u3", 2, math.pi/4, 0.0, 0.0],
            ["cx", 1, 2],
            ["u3", 2, -math.pi/4, 0.0, 0.0],
            ["cx", 0, 2],
            ["u3", 2, math.pi/4, 0.0, 0.0],
            ["cx", 1, 2],
            ["u3", 2, -math.pi/4, 0.0, 0.0],
            ["u3", 0, math.pi/2, 0.0, 0.0],
            ["u3", 1, math.pi/2, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 0, math.pi/4, 0.0, 0.0],
            ["u3", 1, -math.pi/4, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 0, math.pi/2, 0.0, math.pi],
            ["u3", 1, math.pi/2, 0.0, 0.0]]

def _identify_known_gate(target, qubits):
    """Check if target matches a known gate up to global phase."""
    def _close(M, N, tol=1e-6):
        # Check up to global phase
        tr = np.trace(M.conj().T @ N)
        if abs(tr) < 1e-10:
            return False
        gp = tr / abs(tr)
        return np.allclose(M, gp * N, atol=tol)
    
    if qubits == 2:
        # CNOT
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _close(target, cnot):
            return _known_cnot()
        # SWAP
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _close(target, swap):
            return _known_swap()
        # iSWAP
        iswap = 0.5 * np.array([[0,1j,0,0],[1j,0,0,0],[0,0,0,1j],[0,0,1j,0]], dtype=complex)
        if _close(target, iswap):
            return _known_iswap()
    elif qubits == 3:
        # Toffoli
        toff = np.eye(8, dtype=complex)
        toff[6,6] = 0; toff[6,7] = 1; toff[7,6] = 1; toff[7,7] = 0
        if _close(target, toff):
            return _known_toffoli()
    return None

def _local_search_2qubit(target, gates, qubits, max_iter=50):
    """Polish a 2-qubit circuit with local search on u3 parameters."""
    best_dist = _distance(target, _build_fast(gates, qubits))
    params = []
    for g in gates:
        if g[0] == "u3":
            params.extend([g[2], g[3], g[4]])
    
    for it in range(max_iter):
        if _time_left() < 5:
            break
        improved = False
        for i in range(len(params)):
            for delta in [0.1, -0.1, 0.01, -0.01, 0.5, -0.5]:
                new_params = params[:]
                new_params[i] += delta
                # Rebuild gates
                new_gates = []
                pi = 0
                for g in gates:
                    if g[0] == "u3":
                        new_gates.append(["u3", g[1], new_params[pi], new_params[pi+1], new_params[pi+2]])
                        pi += 3
                    else:
                        new_gates.append(list(g))
                d = _distance(target, _build_fast(new_gates, qubits))
                if d < best_dist - 1e-10:
                    best_dist = d
                    params = new_params
                    improved = True
        if not improved:
            break
    # Rebuild final gates
    final_gates = []
    pi = 0
    for g in gates:
        if g[0] == "u3":
            final_gates.append(["u3", g[1], params[pi], params[pi+1], params[pi+2]])
            pi += 3
        else:
            final_gates.append(list(g))
    return final_gates

def _local_search_3qubit(target, gates, qubits, max_iter=30):
    """Polish a 3-qubit circuit with local search on u3 parameters."""
    best_dist = _distance(target, _build_fast(gates, qubits))
    params = []
    for g in gates:
        if g[0] == "u3":
            params.extend([g[2], g[3], g[4]])
    
    for it in range(max_iter):
        if _time_left() < 5:
            break
        improved = False
        for i in range(len(params)):
            for delta in [0.1, -0.1, 0.01, -0.01]:
                new_params = params[:]
                new_params[i] += delta
                new_gates = []
                pi = 0
                for g in gates:
                    if g[0] == "u3":
                        new_gates.append(["u3", g[1], new_params[pi], new_params[pi+1], new_params[pi+2]])
                        pi += 3
                    else:
                        new_gates.append(list(g))
                d = _distance(target, _build_fast(new_gates, qubits))
                if d < best_dist - 1e-10:
                    best_dist = d
                    params = new_params
                    improved = True
        if not improved:
            break
    final_gates = []
    pi = 0
    for g in gates:
        if g[0] == "u3":
            final_gates.append(["u3", g[1], params[pi], params[pi+1], params[pi+2]])
            pi += 3
        else:
            final_gates.append(list(g))
    return final_gates

def _generic_2qubit(target, budget):
    """Generic 2-qubit synthesis using 3-CNOT template + local search."""
    # Template: u3s, CNOT(0,1), u3s, CNOT(1,0), u3s, CNOT(0,1), u3s
    gates = []
    for q in range(2):
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    gates.append(["cx", 0, 1])
    for q in range(2):
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    gates.append(["cx", 1, 0])
    for q in range(2):
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    gates.append(["cx", 0, 1])
    for q in range(2):
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    
    return _local_search_2qubit(target, gates, 2, max_iter=100)

def _generic_3qubit(target, budget):
    """Generic 3-qubit synthesis using CNOT ladder + local search."""
    # Use a dense CNOT ladder
    pairs = [(0,1), (0,2), (1,2)]
    gates = []
    for q in range(3):
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    for i in range(budget):
        c, t = pairs[i % 3]
        gates.append(["cx", c, t])
        for q in range(3):
            gates.append(["u3", q, 0.0, 0.0, 0.0])
    for q in range(3):
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    
    return _local_search_3qubit(target, gates, 3, max_iter=50)

def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try known gates first
    known = _identify_known_gate(target, qubits)
    if known is not None:
        # Verify it's within budget
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget and len(known) <= 400:
            return known
    
    if qubits == 2:
        gates = _generic_2qubit(target, budget)
    else:
        gates = _generic_3qubit(target, budget)
    
    # Final check
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        # Truncate to fit budget (shouldn't happen with our templates)
        cx_seen = 0
        trimmed = []
        for g in gates:
            if g[0] == "cx" and cx_seen >= budget:
                continue
            if g[0] == "cx":
                cx_seen += 1
            trimmed.append(g)
        gates = trimmed
    
    return gates
# EVOLVE-BLOCK-END
