"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np
from scipy.linalg import logm

_START_TIME = time.time()
_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _apply_u3(matrix, q, qubits, theta, phi, lam):
    """Apply u3 gate to qubit q on the given matrix."""
    n = 2 ** qubits
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1, e2 = np.exp(1j * lam), np.exp(1j * phi)
    bit_pos = qubits - 1 - q
    shift = 1 << bit_pos
    new_matrix = np.empty_like(matrix)
    for i in range(n):
        if (i & shift) == 0:
            j = i | shift
            new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
            new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
    return new_matrix


def _apply_cx(matrix, control, target, qubits):
    """Apply CNOT gate."""
    n = 2 ** qubits
    cshift = 1 << (qubits - 1 - control)
    tshift = 1 << (qubits - 1 - target)
    perm = np.arange(n)
    for i in range(n):
        if (i & cshift):
            perm[i] = i ^ tshift
    return matrix[perm, :]


def _build_from_gates(gates, qubits):
    """Build unitary from gate list."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            matrix = _apply_u3(matrix, q, qubits, theta, phi, lam)
        else:
            _, control, target_q = gate
            matrix = _apply_cx(matrix, control, target_q, qubits)
    return matrix


def _distance_fast(target, matrix):
    """Phase-insensitive distance."""
    n = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _build_fast(gates, qubits, target):
    """Build circuit and compute distance in one pass, efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            matrix = _apply_u3(matrix, q, qubits, theta, phi, lam)
        else:
            _, control, target_q = gate
            matrix = _apply_cx(matrix, control, target_q, qubits)
    return _distance_fast(target, matrix)


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _match_target(target, circuit, qubits, tol=1e-9):
    """Check if target matches the unitary of circuit (up to global phase)."""
    matrix = _build_from_gates(circuit, qubits)
    return _distance_fast(target, matrix) < tol


def _known_circuits():
    """Return dict mapping problem names to exact circuits."""
    circuits = {}

    # CNOT (2 qubits, 1 CNOT)
    circuits["CNOT"] = [_cx(0, 1)]

    # iSWAP (2 qubits, 1 CNOT)
    # iSWAP = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
    # Standard 1-CNOT decomposition:
    # iSWAP = (H⊗H) CNOT(0,1) (S†⊗S) (H⊗H)
    # H = u3(pi/2, 0, pi), S = u3(0, 0, pi/2), S† = u3(0, 0, -pi/2)
    circuits["iSWAP"] = [
        _u3(0, math.pi/2, 0, math.pi),
        _u3(1, math.pi/2, 0, math.pi),
        _cx(0, 1),
        _u3(0, 0, 0, -math.pi/2),
        _u3(1, 0, 0, math.pi/2),
    ]

    # SWAP (2 qubits, 3 CNOTs)
    circuits["SWAP"] = [_cx(0, 1), _cx(1, 0), _cx(0, 1)]

    # Toffoli (3 qubits, 6 CNOTs minimum)
    # Standard 6-CNOT Toffoli decomposition
    # Toffoli(0,1,2): controls 0,1, target 2
    toffoli_circuit = [
        _u3(2, 0, 0, math.pi/2),
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi/4),
        _cx(0, 2),
        _u3(2, 0, 0, math.pi/4),
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi/4),
        _cx(0, 2),
        _u3(2, 0, 0, math.pi/4),
        _u3(2, 0, 0, -math.pi/2),
        _u3(1, 0, 0, math.pi/2),
        _cx(0, 1),
        _u3(1, 0, 0, -math.pi/4),
        _cx(0, 1),
        _u3(1, 0, 0, math.pi/4),
        _u3(1, 0, 0, -math.pi/2),
    ]
    circuits["Toffoli"] = toffoli_circuit

    # CCZ (3 qubits) - similar to Toffoli
    ccz_circuit = [_u3(2, math.pi, 0, -math.pi/2)] + toffoli_circuit + [_u3(2, math.pi, 0, -math.pi/2)]
    circuits["CCZ"] = ccz_circuit

    # C-SWAP / Fredkin (3 qubits, 3 CNOTs minimum)
    fredkin_circuit = [
        _cx(1, 2),
        _cx(0, 1),
        _cx(1, 2),
    ]
    circuits["Fredkin"] = fredkin_circuit

    return circuits


def _nelder_mead(f, x0, max_iter=500, tol=1e-10, deadline=None):
    """Nelder-Mead optimization."""
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        point = x0[:]
        step = 0.1 if abs(x0[i]) < 0.1 else 0.1 * abs(x0[i])
        point[i] += step
        simplex.append(point)

    f_values = [f(x) for x in simplex]
    order = sorted(range(n + 1), key=lambda i: f_values[i])
    simplex = [simplex[i] for i in order]
    f_values = [f_values[i] for i in order]

    alpha = 1.0
    gamma = 2.0
    rho = 0.5

    for it in range(max_iter):
        if deadline is not None and time.time() - _START_TIME > deadline:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)

        if fr < f_values[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        elif fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        else:
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < f_values[-1]:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    f_values[i] = f(simplex[i])

        order = sorted(range(n + 1), key=lambda i: f_values[i])
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]

        if abs(f_values[-1] - f_values[0]) < tol:
            break

    return simplex[0], f_values[0]


def _diagonalize_generic(target, qubits, budget, rng):
    """
    Attempt to synthesize a generic unitary using diagonal phase extraction.
    This works well for unitaries that are close to being diagonal in the computational basis
    after some fixed CNOT structure.
    """
    n = 2 ** qubits
    
    # For 3 qubits, try a few standard CNOT topologies
    # The idea: U = P * D * P† where P is a permutation matrix from CNOTs
    # and D is a diagonal matrix. If we can find P such that P† U P is diagonal,
    # then we can extract the phases.
    
    if qubits != 3:
        return None, None
    
    # Try different CNOT structures
    # Structure 1: CNOT(0,1), CNOT(1,2), CNOT(0,1), CNOT(1,2) ... 
    # Actually, let's just try to see if the unitary is diagonal in some basis.
    
    # A simpler approach: if the target is diagonal (or close to it), we can just use u3 gates.
    # Check if target is diagonal
    off_diag = np.sum(np.abs(target) - np.abs(np.diag(target)))
    if off_diag < 1e-8:
        # It's diagonal! We can synthesize it with just u3 gates on each qubit
        # Actually, a diagonal 3-qubit unitary can be synthesized with single-qubit gates if it's a tensor product,
        # but a general diagonal unitary requires CNOTs.
        # However, if it's diagonal, we can use the fact that any diagonal unitary can be written as
        # a product of controlled-phase gates and single-qubit gates.
        # For now, let's just return the diagonal phases as u3 gates if it's a tensor product.
        pass
    
    # This approach is too complex for now. Let's stick to the optimizer but improve it.
    return None, None


def _optimize_generic(target, qubits, budget, rng):
    """Optimize a generic unitary using coordinate descent + Nelder-Mead."""
    # Build a layout with CNOTs and u3 gates
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    best = [0.0] * n_params
    best_distance = _build_fast(circuit(best), qubits, target)

    # Coordinate descent
    sweeps = 20
    for sweep in range(sweeps):
        if time.time() - _START_TIME > _DEADLINE:
            break
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(n_params):
            if time.time() - _START_TIME > _DEADLINE:
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value
            if rng.random() < 0.2:
                trial = best[i] + rng.uniform(-math.pi, math.pi)
                values = best[:]
                values[i] = trial
                value = _build_fast(circuit(values), qubits, target)
                if value < best_distance:
                    best, best_distance = values, value

    # Nelder-Mead refinement
    if time.time() - _START_TIME < _DEADLINE - 5:
        def f(params):
            return _build_fast(circuit(params), qubits, target)

        x0 = np.array(best)
        try:
            x_opt, f_opt = _nelder_mead(f, x0, max_iter=150, deadline=_DEADLINE - 5)
            if f_opt < best_distance:
                best = list(x_opt)
                best_distance = f_opt
        except Exception:
            pass

    return circuit(best), best_distance


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    rng = random.Random(4242 + qubits + budget)

    # Check known circuits
    known = _known_circuits()
    for key, circuit in known.items():
        if key in name or name in key:
            if _match_target(target, circuit, qubits, tol=1e-8):
                return circuit

    # Also try some standard gates with different naming
    if qubits == 2 and budget >= 1:
        cnot_circuit = [_cx(0, 1)]
        if _match_target(target, cnot_circuit, 2, tol=1e-8):
            return cnot_circuit
        cnot_circuit2 = [_cx(1, 0)]
        if _match_target(target, cnot_circuit2, 2, tol=1e-8):
            return cnot_circuit2
        swap_circuit = [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
        if _match_target(target, swap_circuit, 2, tol=1e-8):
            return swap_circuit
        iswap_candidates = [
            [_u3(0, math.pi/2, 0, math.pi/2), _u3(1, math.pi/2, 0, math.pi/2), _cx(0, 1),
             _u3(0, math.pi/2, -math.pi/2, math.pi/2), _u3(1, math.pi/2, -math.pi/2, math.pi/2)],
            [_u3(0, math.pi, 0, math.pi), _u3(1, math.pi, 0, math.pi), _cx(0, 1),
             _u3(0, math.pi, 0, 0), _u3(1, math.pi, 0, 0)],
        ]
        for cand in iswap_candidates:
            if _match_target(target, cand, 2, tol=1e-8):
                return cand

    if qubits == 3 and budget >= 6:
        toffoli_circuit = known.get("Toffoli", [])
        if toffoli_circuit and _match_target(target, toffoli_circuit, 3, tol=1e-8):
            return toffoli_circuit

    # Use numerical optimizer
    circuit, dist = _optimize_generic(target, qubits, budget, rng)
    return circuit
# EVOLVE-BLOCK-END
