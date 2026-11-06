"""Quantum circuit synthesis with exact known-gate circuits and fast optimization fallback."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0


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
    """Build unitary from gate list. qubit 0 is MSB."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            # Apply single-qubit gate on qubit q
            # qubit 0 is MSB, so for 2 qubits: q=0 is left factor, q=1 is right factor
            # We need to build the full operator
            eye = np.eye(2, dtype=complex)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, control, target = gate
            # CNOT: flip target if control is 1
            # qubit 0 is MSB
            idx = np.arange(n)
            ctrl_bit = (idx >> (qubits - 1 - control)) & 1
            tgt_shift = (qubits - 1 - target)
            # Only flip where control bit is 1
            mask = (ctrl_bit == 1)
            new_idx = idx.copy()
            new_idx[mask] = idx[mask] ^ (1 << tgt_shift)
            U = U[new_idx, :]
    return U


def _distance(target, U):
    """Phase-insensitive distance: 1 - |tr(target^dag U)| / dim."""
    dim = target.shape[0]
    val = np.trace(target.conj().T @ U)
    return 1.0 - min(1.0, abs(val) / dim)


def _check_close(A, B, tol=1e-6):
    """Check if two unitaries are equal up to global phase."""
    dim = A.shape[0]
    # Find a non-zero element to fix phase
    max_idx = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    if abs(A[max_idx]) < 1e-12:
        return False
    phase = A[max_idx] / B[max_idx]
    phase = phase / abs(phase)
    return np.allclose(A, phase * B, atol=tol)


def _known_circuit_cnot_2q():
    """CNOT(0,1) for 2 qubits: just cx(0,1)."""
    return [["cx", 0, 1]]


def _known_circuit_swap_2q():
    """SWAP for 2 qubits: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _known_circuit_iswap_2q():
    """iSWAP for 2 qubits: 2 CNOTs + single-qubit rotations.
    iSWAP = [[0,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,0]]
    """
    pi = math.pi
    # Standard 2-CNOT iSWAP decomposition
    # U0 = diag(1, i) on q0, diag(1, i) on q1  ->  S gates
    # Then CNOT(0,1), then single qubit rotations, then CNOT(1,0)
    return [
        ["u3", 0, 0, 0, pi / 2],       # S on q0
        ["u3", 1, 0, 0, pi / 2],       # S on q1
        ["cx", 0, 1],
        ["u3", 0, pi / 2, 0, 0],       # H on q0
        ["u3", 1, pi / 2, 0, 0],       # H on q1
        ["cx", 1, 0],
        ["u3", 0, pi / 2, 0, 0],       # H on q0
        ["u3", 1, pi / 2, 0, 0],       # H on q1
    ]


def _known_circuit_toffoli_3q():
    """Toffoli (CCX) for 3 qubits: 6 CNOTs standard circuit.
    q0=control1, q1=control2, q2=target
    """
    pi = math.pi
    # Standard 6-CNOT Toffoli decomposition
    return [
        ["u3", 2, pi / 2, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -pi / 4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, pi / 4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -pi / 4, 0, 0],
        ["u3", 2, pi / 2, 0, 0],
    ]


def _known_circuit_fredkin_3q():
    """Fredkin (controlled-SWAP) for 3 qubits: 6 CNOTs.
    q0=control, q1=swap_a, q2=swap_b
    """
    pi = math.pi
    return [
        ["cx", 1, 2],
        ["cx", 0, 2],
        ["cx", 1, 2],
        ["cx", 0, 1],
        ["cx", 1, 2],
        ["cx", 0, 2],
        ["cx", 1, 2],
    ]


def _known_circuit_ccz_3q():
    """CCZ for 3 qubits: 6 CNOTs."""
    pi = math.pi
    # CCZ = diag(1,1,1,1,1,1,1,-1)
    # Use H on target to convert CCX to CCZ
    return [
        ["u3", 2, 0, 0, pi],  # S gate
        ["cx", 1, 2],
        ["u3", 2, 0, 0, -pi / 2],  # S-dagger
        ["cx", 0, 2],
        ["u3", 2, 0, 0, pi / 2],  # S
        ["cx", 1, 2],
        ["u3", 2, 0, 0, -pi / 2],  # S-dagger
        ["cx", 0, 2],
    ]


def _detect_known_gate(target, qubits, budget):
    """Try to identify the target as a known gate and return exact circuit if so."""
    
    if qubits == 2:
        if budget >= 1:
            # CNOT(0,1)
            ref = _build_fast([["cx", 0, 1]], 2)
            if _check_close(target, ref):
                return _known_circuit_cnot_2q()
            # CNOT(1,0)
            ref = _build_fast([["cx", 1, 0]], 2)
            if _check_close(target, ref):
                return [["cx", 1, 0]]
        if budget >= 3:
            # SWAP
            circ = _known_circuit_swap_2q()
            ref = _build_fast(circ, 2)
            if _check_close(target, ref):
                return circ
        if budget >= 2:
            # iSWAP
            circ = _known_circuit_iswap_2q()
            ref = _build_fast(circ, 2)
            if _check_close(target, ref):
                return circ
    elif qubits == 3:
        if budget >= 6:
            # Toffoli (CCX) with q0,q1 as controls, q2 as target
            circ = _known_circuit_toffoli_3q()
            ref = _build_fast(circ, 3)
            if _check_close(target, ref):
                return circ
            # Other control/target combinations for Toffoli
            # We'll check all 6 permutations of (c1,c2,t)
            for c1, c2, t in [(0,1,2), (1,0,2), (0,2,1), (2,0,1), (1,2,0), (2,1,0)]:
                if c1 == 0 and c2 == 1 and t == 2:
                    continue  # already checked
                circ_alt = _permute_toffoli((c1, c2, t))
                if circ_alt:
                    ref = _build_fast(circ_alt, 3)
                    if _check_close(target, ref):
                        return circ_alt
            
            # Fredkin variants (3 CNOTs)
            if budget >= 3:
                for ctrl, a, b in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                    circ = _fredkin_circuit(ctrl, a, b)
                    ref = _build_fast(circ, 3)
                    if _check_close(target, ref):
                        return circ
            
            # CCZ variants (6 CNOTs)
            if budget >= 6:
                for c1, c2, t in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                    circ = _ccz_circuit(c1, c2, t)
                    ref = _build_fast(circ, 3)
                    if _check_close(target, ref):
                        return circ
    return None


def _permute_toffoli(controls_target):
    """Generate Toffoli circuit for arbitrary control/target qubits.
    controls_target = (c1, c2, tgt) where c1,c2 are controls and tgt is target.
    """
    c1, c2, tgt = controls_target
    pi = math.pi
    # Standard Toffoli: controls 0,1 target 2
    # We can use SWAP gates to permute qubits, apply standard Toffoli, then unpermute.
    # But that adds CNOTs. Instead, let's directly construct for each case.
    # For now, handle the most common cases directly.
    
    # If target is q2 and controls are {0,1}: standard
    if tgt == 2 and c1 == 0 and c2 == 1:
        return _known_circuit_toffoli_3q()
    if tgt == 2 and c1 == 1 and c2 == 0:
        return _known_circuit_toffoli_3q()  # symmetric in controls
    
    # If target is q1:
    if tgt == 1:
        # SWAP q1<->q2, apply Toffoli(0,2,1->now at position 2), SWAP back
        # This uses 3+6+3=12 CNOTs, too many.
        # Alternative: use the fact that CCX(c1,c2,t) can be built with 6 CNOTs for any assignment
        # Let me construct it directly
        # For CCX(c1,c2,t) where t != 2:
        # We need a 6-CNOT circuit. The standard one assumes t=2.
        # Let's just return None for non-standard and let the optimizer handle it.
        return None
    
    # If target is q0:
    if tgt == 0:
        return None
    
    return None


def _fredkin_circuit(ctrl, a, b):
    """Fredkin (controlled-SWAP) circuit for given control and swap qubits.
    Uses 3 CNOTs: cx(a,b), cx(ctrl,b), cx(a,b)
    """
    return [
        ["cx", a, b],
        ["cx", ctrl, b],
        ["cx", a, b],
    ]


def _ccz_circuit(c1, c2, t):
    """CCZ circuit for given controls and target."""
    pi = math.pi
    return [
        ["u3", t, 0, 0, pi],
        ["cx", c2, t],
        ["u3", t, 0, 0, -pi / 2],
        ["cx", c1, t],
        ["u3", t, 0, 0, pi / 2],
        ["cx", c2, t],
        ["u3", t, 0, 0, -pi / 2],
        ["cx", c1, t],
    ]


def _optimize(gates_template, target, qubits, budget, max_sweeps=12):
    """Optimize parameters using coordinate descent with adaptive step size."""
    n_params = 3 * sum(1 for g in gates_template if g[0] == "u3")
    
    def circuit(params):
        gates, k = [], 0
        for gate in gates_template:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k + 1], params[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build_fast(circuit(best), qubits))
    
    for sweep in range(max_sweeps):
        if _time_left() < 15:
            break
        step = 0.5 / (1 + sweep)
        for i in range(n_params):
            if _time_left() < 5:
                break
            improved = False
            for trial in (best[i] + step, best[i] - step):
                params = best[:]
                params[i] = trial
                dist = _distance(target, _build_fast(circuit(params), qubits))
                if dist < best_dist:
                    best, best_dist = params, dist
                    improved = True
            # If no improvement, try smaller step
            if not improved and sweep > 2:
                for trial in (best[i] + step * 0.1, best[i] - step * 0.1):
                    params = best[:]
                    params[i] = trial
                    dist = _distance(target, _build_fast(circuit(params), qubits))
                    if dist < best_dist:
                        best, best_dist = params, dist
    
    # Final polish pass with very small steps
    if _time_left() > 20:
        for _ in range(4):
            if _time_left() < 10:
                break
            for i in range(n_params):
                if _time_left() < 5:
                    break
                for trial in (best[i] + 0.01, best[i] - 0.01, best[i] + 0.001, best[i] - 0.001):
                    params = best[:]
                    params[i] = trial
                    dist = _distance(target, _build_fast(circuit(params), qubits))
                    if dist < best_dist:
                        best, best_dist = params, dist
    
    return circuit(best)


def synthesize(problem):
    global problem_name
    problem_name = problem.get("name", "")
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    # Try known gate detection first
    known = _detect_known_gate(target, qubits, budget)
    if known is not None:
        # Verify it's within budget
        n_cnot = sum(1 for g in known if g[0] == "cx")
        if n_cnot <= budget:
            return known
    
    # Fallback: generic optimization
    # Build a template with up to budget CNOTs and single-qubit rotations
    # Use a structured pattern: alternate between different CNOT pairs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # For 2 qubits: use CNOT(0,1) and CNOT(1,0) alternating
    # For 3 qubits: use a set of CNOTs that can reach any 2-qubit interaction
    if qubits == 2:
        cnot_pairs = [(0, 1), (1, 0)]
    else:
        cnot_pairs = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    
    template = []
    n_cnot = 0
    for layer in range(budget):
        template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        template.append(["cx", *cnot_pairs[layer % len(cnot_pairs)]])
        n_cnot += 1
    template += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    result = _optimize(template, target, qubits, budget)
    return result
# EVOLVE-BLOCK-END
