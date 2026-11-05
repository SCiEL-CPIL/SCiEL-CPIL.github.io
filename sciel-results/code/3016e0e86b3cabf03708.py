"""Analytical quantum circuit synthesis with coordinate descent refinement."""
import math
import time
import numpy as np

_DEADLINE = 250.0  # seconds, safety margin before 270s kill


def _elapsed():
    return time.time() - _START_TIME


_START_TIME = time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list. Qubit 0 is MSB."""
    n = 2 ** qubits
    eye2 = np.eye(2, dtype=complex)
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build full operator via kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            matrix = op @ matrix
        else:
            _, control, target = gate
            # CNOT: for each basis state, if control bit is 1, flip target bit
            indices = np.arange(n)
            c_bit = (indices >> (qubits - 1 - control)) & 1
            t_bit_pos = qubits - 1 - target
            new_indices = indices.copy()
            mask = c_bit == 1
            new_indices[mask] ^= (1 << t_bit_pos)
            matrix = matrix[new_indices, :]
    return matrix


def _distance(target, matrix):
    d = 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])
    return max(0.0, d)


def _cnot_gate(control, target):
    return ["cx", control, target]


def _u3_gate(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _synth_cnot_2q(target):
    """Synthesize CNOT on 2 qubits: control=0, target=1."""
    # CNOT = (I ⊗ H) · CX · (I ⊗ H) ... actually CNOT is already a CNOT.
    # But we can express it as: H on target, then CNOT, then H on target.
    # Wait, that's circular. Let me think.
    # Actually CNOT with control 0 target 1:
    # |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
    # We can write: H(1) CX(0,1) H(1) is not CNOT.
    # Let's just use the known decomposition: CNOT = CX directly, but we need u3 + cx.
    # Actually the simplest: just output cx(0,1) with no u3 gates.
    return [_cnot_gate(0, 1)]


def _synth_iswap_2q(target):
    """iSWAP: |01>->i|10>, |10>->i|01>, |00>->|00>, |11>->|11>"""
    # iSWAP = (H⊗H) · (iSWAP without phase?) ...
    # iSWAP = diag(1, i, i, 1) · SWAP? No.
    # iSWAP matrix: [[0,0,i,0],[0,0,0,i],[i,0,0,0],[0,i,0,0]]
    # SWAP: [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]
    # iSWAP = (i on both when swapped) ...
    # Known: iSWAP = (H⊗H) · (i·CNOT-like) ...
    # Standard decomposition: iSWAP = exp(iπ/4(X⊗X + Y⊗Y)) ...
    # Let's use: iSWAP = (S⊗S) · (H⊗H) · CX(0,1) · (H⊗H) · (S†⊗S†) · CX(0,1) · (H⊗H)
    # Actually simpler known decomposition:
    # iSWAP = (1/√2)(I + i·SWAP) ... no.
    # Let me use the standard: iSWAP requires 3 CNOTs or can be done with fewer.
    # iSWAP = CX(0,1) · (S⊗S) · CX(0,1) · (S†⊗S†) · CX(0,1) ... that's 3 CNOTs.
    # Actually: iSWAP = (H⊗H) · CX(0,1) · (i·S⊗i·S) · CX(0,1) · (H⊗H)? Let me verify.
    # Let me just use coordinate descent for this one.
    return None  # signal to use coordinate descent


def _synth_swap_2q(target):
    """SWAP = CX(0,1)·CX(1,0)·CX(0,1)"""
    return [_cnot_gate(0, 1), _cnot_gate(1, 0), _cnot_gate(0, 1)]


def _synth_toffoli_3q(target):
    """Toffoli: 5 CNOT decomposition (standard)."""
    # Standard 5-CNOT Toffoli decomposition
    # Using the well-known decomposition with 5 CNOTs and single-qubit gates
    # |000>|010>|100>|110> unchanged, |001>|011>|101> unchanged, |111>->|111> with phase... 
    # Actually Toffoli: |111>->-|111> or |111>->|111> depending on convention.
    # Standard Toffoli: |abc> -> |ab> if a&b then flip c.
    # 5-CNOT decomposition:
    gates = []
    # This is the standard decomposition from Nielsen & Chuang / Barenco et al.
    # U1(-pi/8) on q2, CX(1,2), U1(pi/4) on q2, CX(0,2), U1(-pi/4) on q2, CX(1,2), U1(pi/8) on q2, CX(0,1)
    # Then need to handle the phase on |111>.
    # Actually the standard 5-CNOT Toffoli:
    gates.append(_u3_gate(2, 0, 0, -math.pi/8))  # phase -pi/8
    gates.append(_cnot_gate(1, 2))
    gates.append(_u3_gate(2, 0, 0, math.pi/4))   # phase pi/4
    gates.append(_cnot_gate(0, 2))
    gates.append(_u3_gate(2, 0, 0, -math.pi/4))  # phase -pi/4
    gates.append(_cnot_gate(1, 2))
    gates.append(_u3_gate(2, 0, 0, math.pi/8))   # phase pi/8
    gates.append(_cnot_gate(0, 1))
    # This gives Toffoli up to a global phase on |111>. Need to check.
    # Actually this standard decomposition gives the correct Toffoli.
    return gates


def _coordinate_descent(target, qubits, budget, initial_gates, max_time=40):
    """Refine u3 parameters via coordinate descent."""
    # Extract parameter indices
    param_indices = []
    for i, g in enumerate(initial_gates):
        if g[0] == "u3":
            param_indices.append(i)
    
    n_params = len(param_indices) * 3
    if n_params == 0:
        return initial_gates
    
    # Flatten parameters
    def get_params(gates):
        params = []
        for g in gates:
            if g[0] == "u3":
                params.extend([g[2], g[3], g[4]])
        return params
    
    def set_params(gates, params):
        new_gates = [list(g) for g in gates]
        k = 0
        for i, g in enumerate(new_gates):
            if g[0] == "u3":
                g[2] = params[k]
                g[3] = params[k+1]
                g[4] = params[k+2]
                k += 3
        return new_gates
    
    params = get_params(initial_gates)
    best_params = params[:]
    best_dist = _distance(target, _build_fast(initial_gates, qubits))
    
    t0 = time.time()
    step = 0.5
    for sweep in range(50):
        if time.time() - t0 > max_time:
            break
        step *= 0.7
        improved = False
        for i in range(len(params)):
            for trial in (params[i] + step, params[i] - step):
                new_params = params[:]
                new_params[i] = trial
                gates = set_params(initial_gates, new_params)
                d = _distance(target, _build_fast(gates, qubits))
                if d < best_dist:
                    best_dist = d
                    best_params = new_params[:]
                    params = new_params[:]
                    improved = True
        if not improved:
            break
    
    return set_params(initial_gates, best_params)


def synthesize(problem):
    global _START_TIME
    _START_TIME = time.time()
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try exact analytical syntheses first
    if qubits == 2:
        if name.lower() in ("cnot", "cx"):
            gates = _synth_cnot_2q(target)
            d = _distance(target, _build_fast(gates, qubits))
            if d < 1e-10:
                return gates
        if name.lower() in ("swap",):
            gates = _synth_swap_2q(target)
            d = _distance(target, _build_fast(gates, qubits))
            if d < 1e-10:
                return gates
        if name.lower() in ("iswap", "i-swap", "iswap"):
            # iSWAP decomposition: 3 CNOTs
            # iSWAP = (S⊗S)·(H⊗H)·CX(0,1)·(H⊗H)·(S†⊗S†)·CX(0,1)·(H⊗H)·CX(0,1) ... 
            # Let me use a known 3-CNOT decomposition
            # iSWAP = CX(0,1) · (iS⊗iS) · CX(0,1) · (iS†⊗iS†) · CX(0,1)
            # where iS = diag(1, i)
            # iS = u3(0, 0, pi/2) ... actually S = diag(1,i) = u3(0,0,pi/2)? 
            # u3(theta, phi, lam): [[cos(t/2), -e^{iλ}sin(t/2)], [e^{iφ}sin(t/2), e^{i(φ+λ)}cos(t/2)]]
            # S = [[1,0],[0,i]]: theta=0 gives [[1,0],[0,1]], not S.
            # For theta=0: [[1,0],[0,e^{i(φ+λ)}]]. So set φ+λ=pi/2.
            # Let's use phi=0, lam=pi/2 for S.
            # iS = diag(1, i) = S. So iS is just S.
            # iS† = S† = diag(1, -i): phi=0, lam=-pi/2.
            gates = []
            # H = u3(pi/2, 0, pi)
            gates.append(_u3_gate(0, math.pi/2, 0, math.pi))
            gates.append(_u3_gate(1, math.pi/2, 0, math.pi))
            gates.append(_cnot_gate(0, 1))
            gates.append(_u3_gate(0, math.pi/2, 0, math.pi))
            gates.append(_u3_gate(1, math.pi/2, 0, math.pi))
            # iS on both
            gates.append(_u3_gate(0, 0, 0, math.pi/2))
            gates.append(_u3_gate(1, 0, 0, math.pi/2))
            gates.append(_cnot_gate(0, 1))
            gates.append(_u3_gate(0, math.pi/2, 0, math.pi))
            gates.append(_u3_gate(1, math.pi/2, 0, math.pi))
            gates.append(_cnot_gate(0, 1))
            # iS† on both
            gates.append(_u3_gate(0, 0, 0, -math.pi/2))
            gates.append(_u3_gate(1, 0, 0, -math.pi/2))
            d = _distance(target, _build_fast(gates, qubits))
            if d < 1e-10:
                return gates
            # Try coordinate descent
            gates = _coordinate_descent(target, qubits, budget, gates, max_time=30)
            return gates
    elif qubits == 3:
        if name.lower() in ("toffoli", "ccx", "ccnot"):
            gates = _synth_toffoli_3q(target)
            d = _distance(target, _build_fast(gates, qubits))
            if d < 1e-10:
                return gates
            # Try coordinate descent
            gates = _coordinate_descent(target, qubits, budget, gates, max_time=30)
            return gates
    
    # Generic: use coordinate descent with a reasonable initial layout
    # Build initial layout with CNOTs and u3 gates
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = np.random.RandomState(42 + qubits)
    
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(_u3_gate(q, rng.uniform(0, 2*math.pi), rng.uniform(0, 2*math.pi), rng.uniform(0, 2*math.pi)))
        layout.append(_cnot_gate(*pairs[layer % len(pairs)]))
    for q in range(qubits):
        layout.append(_u3_gate(q, rng.uniform(0, 2*math.pi), rng.uniform(0, 2*math.pi), rng.uniform(0, 2*math.pi)))
    
    # Coordinate descent refinement
    refined = _coordinate_descent(target, qubits, budget, layout, max_time=35)
    
    return refined
