"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 258.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _cnot_perm(qubits, control, target):
    n = 1 << qubits
    cshift = 1 << (qubits - 1 - control)
    tshift = 1 << (qubits - 1 - target)
    perm = np.arange(n)
    for i in range(n):
        if (i & cshift):
            perm[i] = i ^ tshift
    return perm


def _build_fast(gates, qubits, target, cnot_cache=None):
    """Build circuit and compute distance in one pass, efficiently."""
    n = 1 << qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            e1 = np.exp(1j * lam)
            e2 = np.exp(1j * phi)
            # u3 = [[c, -e1*s], [e2*s, e2*e1*c]]
            bit_pos = qubits - 1 - q
            shift = 1 << bit_pos
            new_matrix = np.empty_like(matrix)
            for i in range(n):
                if (i & shift) == 0:
                    j = i | shift
                    new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
                    new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
            matrix = new_matrix
        else:
            _, control, target_q = gate
            key = (control, target_q)
            if cnot_cache is not None:
                perm = cnot_cache.get(key)
                if perm is None:
                    perm = _cnot_perm(qubits, control, target_q)
                    cnot_cache[key] = perm
            else:
                perm = _cnot_perm(qubits, control, target_q)
            matrix = matrix[perm, :]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _fast_frobenius_dist(gates, qubits, target, cnot_cache=None):
    """Phase-insensitive distance 1 - |tr(Udagger M)|/n, computed via one matmul."""
    n = 1 << qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            e1 = np.exp(1j * lam)
            e2 = np.exp(1j * phi)
            bit_pos = qubits - 1 - q
            shift = 1 << bit_pos
            new_matrix = np.empty_like(matrix)
            for i in range(n):
                if (i & shift) == 0:
                    j = i | shift
                    new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
                    new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
            matrix = new_matrix
        else:
            _, control, target_q = gate
            key = (control, target_q)
            if cnot_cache is not None:
                perm = cnot_cache.get(key)
                if perm is None:
                    perm = _cnot_perm(qubits, control, target_q)
                    cnot_cache[key] = perm
            else:
                perm = _cnot_perm(qubits, control, target_q)
            matrix = matrix[perm, :]
    # Global-phase-free: distance = 1 - |tr(Udagger M)|/n
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _exact_match(target, circuit, qubits, cnot_cache=None):
    """Check if target matches the unitary of circuit (up to global phase), fast."""
    n = 1 << qubits
    matrix = np.eye(n, dtype=complex)
    for gate in circuit:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            e1 = np.exp(1j * lam)
            e2 = np.exp(1j * phi)
            bit_pos = qubits - 1 - q
            shift = 1 << bit_pos
            new_matrix = np.empty_like(matrix)
            for i in range(n):
                if (i & shift) == 0:
                    j = i | shift
                    new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
                    new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
            matrix = new_matrix
        else:
            _, control, target_q = gate
            key = (control, target_q)
            if cnot_cache is not None:
                perm = cnot_cache.get(key)
                if perm is None:
                    perm = _cnot_perm(qubits, control, target_q)
                    cnot_cache[key] = perm
            else:
                perm = _cnot_perm(qubits, control, target_q)
            matrix = matrix[perm, :]
    # Frobenius distance to target (phase-insensitive check via trace)
    diff = target - matrix
    frob = np.sqrt(np.vdot(diff, diff).real)
    # Also check global phase: tr(Udagger M) should be ~ n * e^{i alpha}
    tr = np.trace(target.conj().T @ matrix)
    return frob < 1e-7 or abs(tr) / n > 1.0 - 1e-7


def _try_known_circuits(target, qubits, budget, cnot_cache=None):
    """Try hardcoded exact circuits for standard gates."""
    # 2-qubit gates
    if qubits == 2:
        # CNOT
        if budget >= 1:
            for ctrl, tgt in [(0, 1), (1, 0)]:
                circuit = [_cx(ctrl, tgt)]
                if _exact_match(target, circuit, 2, cnot_cache):
                    return circuit

        # SWAP
        if budget >= 3:
            circuit = [_cx(0, 1), _cx(1, 0), _cx(0, 1)]
            if _exact_match(target, circuit, 2, cnot_cache):
                return circuit

        # iSWAP: (H⊗H) · CX(0,1) · (S†⊗S) · CX(1,0) · (H⊗H)
        if budget >= 2:
            circuit = [
                _u3(0, math.pi/2, 0, math.pi),
                _u3(1, math.pi/2, 0, math.pi),
                _cx(0, 1),
                _u3(0, 0, 0, -math.pi/2),
                _u3(1, 0, 0, math.pi/2),
                _cx(1, 0),
                _u3(0, math.pi/2, 0, math.pi),
                _u3(1, math.pi/2, 0, math.pi),
            ]
            if _exact_match(target, circuit, 2, cnot_cache):
                return circuit

    # 3-qubit gates
    if qubits == 3:
        if budget >= 6:
            toffoli_variants = [
                # Toffoli(0,1,2)
                [
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
                ],
                # Toffoli(0,2,1)
                [
                    _u3(1, 0, 0, math.pi/2),
                    _cx(2, 1),
                    _u3(1, 0, 0, -math.pi/4),
                    _cx(0, 1),
                    _u3(1, 0, 0, math.pi/4),
                    _cx(2, 1),
                    _u3(1, 0, 0, -math.pi/4),
                    _cx(0, 1),
                    _u3(1, 0, 0, math.pi/4),
                    _u3(1, 0, 0, -math.pi/2),
                    _u3(2, 0, 0, math.pi/2),
                    _cx(0, 2),
                    _u3(2, 0, 0, -math.pi/4),
                    _cx(0, 2),
                    _u3(2, 0, 0, math.pi/4),
                    _u3(2, 0, 0, -math.pi/2),
                ],
                # Toffoli(1,0,2)
                [
                    _u3(2, 0, 0, math.pi/2),
                    _cx(0, 2),
                    _u3(2, 0, 0, -math.pi/4),
                    _cx(1, 2),
                    _u3(2, 0, 0, math.pi/4),
                    _cx(0, 2),
                    _u3(2, 0, 0, -math.pi/4),
                    _cx(1, 2),
                    _u3(2, 0, 0, math.pi/4),
                    _u3(2, 0, 0, -math.pi/2),
                    _u3(0, 0, 0, math.pi/2),
                    _cx(1, 0),
                    _u3(0, 0, 0, -math.pi/4),
                    _cx(1, 0),
                    _u3(0, 0, 0, math.pi/4),
                    _u3(0, 0, 0, -math.pi/2),
                ],
                # Toffoli(1,2,0)
                [
                    _u3(0, 0, 0, math.pi/2),
                    _cx(2, 0),
                    _u3(0, 0, 0, -math.pi/4),
                    _cx(1, 0),
                    _u3(0, 0, 0, math.pi/4),
                    _cx(2, 0),
                    _u3(0, 0, 0, -math.pi/4),
                    _cx(1, 0),
                    _u3(0, 0, 0, math.pi/4),
                    _u3(0, 0, 0, -math.pi/2),
                    _u3(2, 0, 0, math.pi/2),
                    _cx(1, 2),
                    _u3(2, 0, 0, -math.pi/4),
                    _cx(1, 2),
                    _u3(2, 0, 0, math.pi/4),
                    _u3(2, 0, 0, -math.pi/2),
                ],
                # Toffoli(2,0,1)
                [
                    _u3(1, 0, 0, math.pi/2),
                    _cx(0, 1),
                    _u3(1, 0, 0, -math.pi/4),
                    _cx(2, 1),
                    _u3(1, 0, 0, math.pi/4),
                    _cx(0, 1),
                    _u3(1, 0, 0, -math.pi/4),
                    _cx(2, 1),
                    _u3(1, 0, 0, math.pi/4),
                    _u3(1, 0, 0, -math.pi/2),
                    _u3(0, 0, 0, math.pi/2),
                    _cx(2, 0),
                    _u3(0, 0, 0, -math.pi/4),
                    _cx(2, 0),
                    _u3(0, 0, 0, math.pi/4),
                    _u3(0, 0, 0, -math.pi/2),
                ],
                # Toffoli(2,1,0)
                [
                    _u3(0, 0, 0, math.pi/2),
                    _cx(1, 0),
                    _u3(0, 0, 0, -math.pi/4),
                    _cx(2, 0),
                    _u3(0, 0, 0, math.pi/4),
                    _cx(1, 0),
                    _u3(0, 0, 0, -math.pi/4),
                    _cx(2, 0),
                    _u3(0, 0, 0, math.pi/4),
                    _u3(0, 0, 0, -math.pi/2),
                    _u3(1, 0, 0, math.pi/2),
                    _cx(2, 1),
                    _u3(1, 0, 0, -math.pi/4),
                    _cx(2, 1),
                    _u3(1, 0, 0, math.pi/4),
                    _u3(1, 0, 0, -math.pi/2),
                ],
            ]
            for circuit in toffoli_variants:
                if _exact_match(target, circuit, 3, cnot_cache):
                    return circuit

    return None


def _coordinate_descent(values, layout, qubits, target, cnot_cache, n_params,
                        best, best_distance, sweeps, deadline, rng):
    """Run coordinate descent with adaptive step sizes."""
    for sweep in range(sweeps):
        if time.time() > deadline:
            break
        step = 1.0 / (1 + sweep * 0.5)
        for i in range(n_params):
            if time.time() > deadline:
                break
            for trial in (best[i] + step, best[i] - step):
                values[i] = trial
                value = _fast_frobenius_dist(circuit_from_values(layout, values), qubits, target, cnot_cache)
                if value < best_distance:
                    best_distance = value
                    best[i] = trial
                else:
                    values[i] = best[i]
            # Random perturbation
            if rng.random() < 0.25:
                trial = best[i] + rng.uniform(-math.pi, math.pi)
                values[i] = trial
                value = _fast_frobenius_dist(circuit_from_values(layout, values), qubits, target, cnot_cache)
                if value < best_distance:
                    best_distance = value
                    best[i] = trial
                else:
                    values[i] = best[i]
    return best, best_distance


def circuit_from_values(layout, values):
    gates, k = [], 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    rng = random.Random(4242 + qubits + budget)

    # Precompute CNOT permutations
    cnot_cache = {}
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    for c, t in pairs:
        cnot_cache[(c, t)] = _cnot_perm(qubits, c, t)

    # 1. Try hardcoded exact circuits for common gates
    known = _try_known_circuits(target, qubits, budget, cnot_cache)
    if known is not None:
        return known

    # 2. Build layout
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]

    n_params = 3 * sum(1 for g in layout if g[0] == "u3")

    # Multi-start coordinate descent
    best_overall = [0.0] * n_params
    best_dist_overall = _fast_frobenius_dist(circuit_from_values(layout, best_overall), qubits, target, cnot_cache)

    # Try several random initializations
    n_starts = 4 if qubits == 2 else 3
    for start in range(n_starts):
        if time.time() > _DEADLINE - 30:
            break
        if start == 0:
            init = [0.0] * n_params
        else:
            init = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        best = init[:]
        best_distance = _fast_frobenius_dist(circuit_from_values(layout, best), qubits, target, cnot_cache)

        # Coordinate descent
        sweeps = 25 if qubits == 2 else 15
        best, best_distance = _coordinate_descent(
            init, layout, qubits, target, cnot_cache, n_params,
            best, best_distance, sweeps, _DEADLINE - 30, rng
        )

        if best_distance < best_dist_overall:
            best_dist_overall = best_distance
            best_overall = best[:]

    # 3. High-precision polish with smaller steps
    if time.time() < _DEADLINE - 15:
        polish_sweeps = 10
        for sweep in range(polish_sweeps):
            if time.time() > _DEADLINE - 10:
                break
            step = 0.1 / (1 + sweep * 0.5)
            for i in range(n_params):
                if time.time() > _DEADLINE - 10:
                    break
                for trial in (best_overall[i] + step, best_overall[i] - step):
                    values = best_overall[:]
                    values[i] = trial
                    value = _fast_frobenius_dist(circuit_from_values(layout, values), qubits, target, cnot_cache)
                    if value < best_dist_overall:
                        best_dist_overall = value
                        best_overall[i] = trial
                # Fine steps
                for trial in (best_overall[i] + step / 10, best_overall[i] - step / 10):
                    values = best_overall[:]
                    values[i] = trial
                    value = _fast_frobenius_dist(circuit_from_values(layout, values), qubits, target, cnot_cache)
                    if value < best_dist_overall:
                        best_dist_overall = value
                        best_overall[i] = trial

    return circuit_from_values(layout, best_overall)
# EVOLVE-BLOCK-END
