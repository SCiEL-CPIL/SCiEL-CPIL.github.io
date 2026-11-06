"""Combined quantum circuit synthesis program with fast builder and improved search."""
import math
import time
import random
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


def _build_fast(gates, qubits):
    """Faster build using precomputed kron structures for u3 gates."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    
    # Precompute kron structures for each qubit position
    kron_structs = []
    for q in range(qubits):
        parts = [np.array([[1]], dtype=complex)]
        for k in range(qubits):
            if k == q:
                parts.append(np.eye(2, dtype=complex))
            else:
                parts.append(eye2)
        # We'll compute this on the fly but cache the structure
        kron_structs.append(parts)
    
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
    """iSWAP = (H⊗H) · CX(0,1) · (S†⊗S) · CX(0,1) · (H⊗H)"""
    return [_h(0), _h(1), _cnot(0, 1), _sdag(0), _s(1), _cnot(0, 1), _h(0), _h(1)]


def _decomp_toffoli(c1, c2, t):
    """6-CNOT Toffoli decomposition."""
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


def _generic_synthesis(target, qubits, budget):
    """Coordinate descent synthesis for generic unitaries with improved search."""
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
    
    # Precompute CNOT index permutations
    cnot_indices = []
    for layer in range(budget):
        c, t = pairs[layer % len(pairs)]
        idx = np.arange(n)
        bit = (idx >> (qubits - 1 - c)) & 1
        cnot_indices.append(idx ^ (bit << (qubits - 1 - t)))
    
    # Precompute kron structures for u3 gates
    u3_kron_parts = []
    for gate in layout:
        if gate[0] == "u3":
            q = gate[1]
            parts = []
            for k in range(qubits):
                parts.append(k == q)
            u3_kron_parts.append(parts)
        else:
            u3_kron_parts.append(None)
    
    def build_from_params(values):
        """Build unitary from parameter values using precomputed structures."""
        U = np.eye(n, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        param_idx = 0
        cnot_idx = 0
        for gate in layout:
            if gate[0] == "u3":
                theta = values[param_idx]
                phi = values[param_idx + 1]
                lam = values[param_idx + 2]
                param_idx += 3
                single = _u3_matrix(theta, phi, lam)
                op = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    op = np.kron(op, single if u3_kron_parts[layout.index(gate)][k] else eye2)
                U = op @ U
            else:
                U = U[cnot_indices[cnot_idx], :]
                cnot_idx += 1
        return U
    
    # Actually, let's use a cleaner approach with explicit tracking
    def build_from_params_v2(values):
        U = np.eye(n, dtype=complex)
        eye2 = np.eye(2, dtype=complex)
        param_idx = 0
        cnot_idx = 0
        for i, gate in enumerate(layout):
            if gate[0] == "u3":
                theta = values[param_idx]
                phi = values[param_idx + 1]
                lam = values[param_idx + 2]
                param_idx += 3
                q = gate[1]
                single = _u3_matrix(theta, phi, lam)
                op = np.array([[1]], dtype=complex)
                for k in range(qubits):
                    op = np.kron(op, single if k == q else eye2)
                U = op @ U
            else:
                U = U[cnot_indices[cnot_idx], :]
                cnot_idx += 1
        return U
    
    # Initialize with random values for better exploration
    rng = random.Random(42)
    best = [rng.uniform(-math.pi, math.pi) for _ in range(num_params)]
    
    best_U = build_from_params_v2(best)
    best_dist = _distance(target, best_U)
    
    # Also try zero initialization
    zero_best = [0.0] * num_params
    zero_U = build_from_params_v2(zero_best)
    zero_dist = _distance(target, zero_U)
    if zero_dist < best_dist:
        best = zero_best
        best_dist = zero_dist
        best_U = zero_U
    
    # Coordinate descent with multiple strategies
    max_sweeps = 50
    rng_search = random.Random(12345)
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 5:
            break
        step = 1.0 / (1.0 + sweep * 0.3)
        improved = False
        
        # Randomize parameter order each sweep
        param_order = list(range(num_params))
        rng_search.shuffle(param_order)
        
        for i in param_order:
            if time.time() > _DEADLINE - 5:
                break
            base = best[i]
            
            # Try several perturbations
            trials = [base + step, base - step, base + step * 0.5, base - step * 0.5,
                      base + step * 0.25, base - step * 0.25]
            if sweep < 10:
                trials.append(base + rng_search.uniform(-math.pi, math.pi))
                trials.append(base + rng_search.uniform(-math.pi/2, math.pi/2))
            
            for trial in trials:
                values = best[:]
                values[i] = trial
                U = build_from_params_v2(values)
                d = _distance(target, U)
                if d < best_dist:
                    best = values
                    best_dist = d
                    best_U = U
                    improved = True
        
        if not improved and sweep > 15:
            break
    
    # Final polish with smaller steps
    for polish in range(20):
        if time.time() > _DEADLINE - 5:
            break
        step = 0.02 / (1.0 + polish * 0.5)
        for i in range(num_params):
            if time.time() > _DEADLINE - 5:
                break
            base = best[i]
            for trial in (base + step, base - step, base + step * 0.5, base - step * 0.5):
                values = best[:]
                values[i] = trial
                U = build_from_params_v2(values)
                d = _distance(target, U)
                if d < best_dist:
                    best = values
                    best_dist = d
                    best_U = U
    
    # Convert back to gate list
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], best[k], best[k + 1], best[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    
    return gates


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
    return _generic_synthesis(target, qubits, budget)
