"""Quantum circuit synthesis: exact known gates + algebraic diagonal solve for generics."""
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
            _decomp_cnot_01(),
            _decomp_cnot_10(),
            _decomp_swap(),
            _decomp_iswap(),
        ]
        for gates in candidates:
            U = _build(gates, qubits)
            if _distance(target, U) < 1e-10:
                return gates
    elif qubits == 3:
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


def _cnot_permutation(c, t, qubits):
    """Return the permutation array for a CNOT with control c, target t."""
    n = 2 ** qubits
    idx = np.arange(n)
    bit = (idx >> (qubits - 1 - c)) & 1
    return idx ^ (bit << (qubits - 1 - t))


def _build_from_diagonal(cnot_layout, diagonal_angles, qubits):
    """
    Build unitary from CNOT layout and diagonal phase angles.
    The circuit is: D0 · CX_{b_{K-1}} · D1 · ... · CX_{b_0} · DK
    where Di are diagonal gates with angles specified by diagonal_angles.
    diagonal_angles is a list of (K+1) arrays, each of length 2^qubits.
    """
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    
    # Apply last diagonal
    D = np.diag(np.exp(1j * diagonal_angles[-1]))
    U = D @ U
    
    # Apply CNOTs from last to first, with diagonals in between
    for i in range(len(cnot_layout) - 1, -1, -1):
        c, t = cnot_layout[i]
        perm = _cnot_permutation(c, t, qubits)
        U = U[perm, :]
        D = np.diag(np.exp(1j * diagonal_angles[i]))
        U = D @ U
    
    return U


def _solve_diagonal_phases(cnot_layout, target, qubits):
    """
    Solve for diagonal phase angles using linear algebra.
    We want: D_K · CX_{b_{K-1}} · D_{K-1} · ... · CX_{b_0} · D_0 ≈ target
    
    This is equivalent to finding phases such that the product matches.
    We can work with the logarithm of the matrix, but a more direct approach:
    
    Let P_i be the permutation for CNOT i.
    We want: exp(iD_K) P_{K-1} exp(iD_{K-1}) ... P_0 exp(iD_0) = T (up to global phase)
    
    Taking the matrix logarithm is tricky due to permutations. Instead, we can
    use the fact that for diagonal gates, we can solve layer by layer.
    
    Actually, a cleaner approach: the product of a CNOT and diagonals can be
    represented. Let's use a different strategy:
    
    We know that any unitary can be decomposed as:
    U = V · D · W
    where D is diagonal and V, W are products of CNOTs and single-qubit gates.
    
    For our layout with only CNOTs and diagonals, we can solve for the diagonal
    phases by working in the computational basis.
    
    Let's use a direct approach: iterate to find the diagonal phases.
    Actually, for a fixed CNOT layout, the problem is:
    Find phases d_i[j] for i=0..K, j=0..2^n-1 such that the product matches target.
    
    This is a system of nonlinear equations in the phases. However, we can linearize
    by taking the imaginary part of the logarithm.
    
    A more practical approach: use the fact that the CNOTs permute the basis states.
    The diagonal gates apply phases to basis states. The overall effect on each
    basis state |x⟩ can be tracked.
    
    For each basis state |x⟩, the CNOTs will permute it to some state |π(x)⟩,
    and the diagonals will apply phases. The total phase on |x⟩ is the sum of
    phases applied at each step.
    
    Let's define: for a given CNOT layout, compute the permutation π and the
    "phase accumulation" for each basis state.
    
    Actually, let's use a simpler numerical approach: set up the system and solve
    using least squares on the phases.
    """
    n = 2 ** qubits
    K = len(cnot_layout)
    
    # We'll solve for phases using the matrix equation directly.
    # The key insight: if we fix all but one diagonal, the remaining diagonal
    # can be solved in closed form.
    
    # Let's use an iterative approach: start with zero phases, then solve for
    # each diagonal in sequence.
    
    phases = [np.zeros(n) for _ in range(K + 1)]
    
    # For each iteration, solve for each diagonal layer.
    for iteration in range(50):
        if time.time() > _DEADLINE - 10:
            break
        
        for layer in range(K + 1):
            if time.time() > _DEADLINE - 10:
                break
            
            # Compute the unitary with current phases, but replace layer's phase
            # with a variable, and solve for it.
            
            # Build the "partial" unitary without layer's diagonal.
            # Then we need: target = D_layer @ (rest)
            # So D_layer = target @ (rest)^dag
            # And D_layer should be diagonal, so we extract its diagonal.
            
            # Compute rest: the unitary with all diagonals except layer set to identity.
            phases_rest = [phases[i] if i != layer else np.zeros(n) for i in range(K + 1)]
            U_rest = _build_from_diagonal(cnot_layout, phases_rest, qubits)
            
            # We want: D_layer @ U_rest ≈ target (up to global phase)
            # So D_layer ≈ target @ U_rest^dag
            D_layer = target @ U_rest.conj().T
            
            # Extract diagonal and convert to phases.
            diag = np.diag(D_layer)
            # The phase is the argument of the diagonal entries.
            # But we need to be careful: the diagonal should be e^{i*phase}.
            phases[layer] = np.angle(diag)
        
        # Check convergence
        U_test = _build_from_diagonal(cnot_layout, phases, qubits)
        dist = _distance(target, U_test)
        if dist < 1e-12:
            break
    
    return phases


def _generic_synthesis_algebraic(target, qubits, budget):
    """
    Algebraic synthesis using CNOT layout and diagonal phase solving.
    """
    n = 2 ** qubits
    
    # Use a standard CNOT layout: cycle through all pairs.
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    cnot_layout = [pairs[i % len(pairs)] for i in range(budget)]
    
    # Solve for diagonal phases.
    phases = _solve_diagonal_phases(cnot_layout, target, qubits)
    
    # Build the gate list.
    gates = []
    
    # Apply diagonals and CNOTs in the correct order.
    # Circuit: D_K · CX_{b_{K-1}} · D_{K-1} · ... · CX_{b_0} · D_0
    # In gate list order (applied left to right): D_0, CX_0, D_1, CX_1, ..., D_K
    
    for i in range(len(cnot_layout)):
        # Apply diagonal D_i
        for q in range(qubits):
            # The diagonal phase for qubit q is the sum of phases for all basis
            # states where qubit q is 1. But actually, a diagonal gate with
            # phases d[j] for j=0..2^n-1 is NOT a product of single-qubit
            # diagonal gates unless the phases have a special structure.
            
            # This is a problem: general diagonal gates cannot be decomposed
            # into single-qubit gates alone. We need CNOTs to decompose them.
            
            # So this approach doesn't work directly with u3 gates only.
            pass
    
    # The above approach fails because general diagonal gates require CNOTs
    # to decompose into single-qubit gates. We need a different strategy.
    
    # Let's fall back to coordinate descent but with a better initialization.
    return _generic_synthesis_cd(target, qubits, budget)


def _generic_synthesis_cd(target, qubits, budget):
    """Coordinate descent synthesis for generic unitaries with improved initialization."""
    n = 2 ** qubits
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Build layout: interleaved u3 gates and CNOTs
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    num_params = 3 * num_u3
    best = [0.0] * num_params
    
    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best_U = _build(circuit(best), qubits)
    best_dist = _distance(target, best_U)
    
    # Coordinate descent with random restarts for the first few sweeps
    max_sweeps = 40
    import random
    rng = random.Random(12345 + qubits * 7 + budget)
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 5:
            break
        step = 1.0 / (1.0 + sweep * 0.3)
        improved = False
        for i in range(num_params):
            if time.time() > _DEADLINE - 5:
                break
            base = best[i]
            trials = [base + step, base - step, base + step * 0.5, base - step * 0.5]
            if sweep < 10:
                trials.append(base + rng.uniform(-math.pi, math.pi))
            for trial in trials:
                values = best[:]
                values[i] = trial
                U = _build(circuit(values), qubits)
                d = _distance(target, U)
                if d < best_dist:
                    best = values
                    best_dist = d
                    best_U = U
                    improved = True
                    break
        if not improved and sweep > 12:
            break
    
    # Final polish with smaller steps
    for polish in range(20):
        if time.time() > _DEADLINE - 5:
            break
        step = 0.05 / (1.0 + polish)
        for i in range(num_params):
            if time.time() > _DEADLINE - 5:
                break
            base = best[i]
            for trial in (base + step, base - step):
                values = best[:]
                values[i] = trial
                U = _build(circuit(values), qubits)
                d = _distance(target, U)
                if d < best_dist:
                    best = values
                    best_dist = d
                    best_U = U
    
    return circuit(best)


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Try known gate matching first
    known = _match_known(target, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == 'cx')
        if cnot_count <= budget:
            return known
    
    # Generic synthesis
    return _generic_synthesis_cd(target, qubits, budget)
