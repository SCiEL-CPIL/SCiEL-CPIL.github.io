"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = 230.0  # safety margin before 270s kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])


def _build_fast(gates, qubits):
    """Build unitary using in-place operations for speed."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            bit_pos = qubits - 1 - q
            # Iterate over all states where qubit q is 0
            for i in range(n):
                if (i >> bit_pos) & 1 == 0:
                    j = i | (1 << bit_pos)
                    row_i = matrix[i, :].copy()
                    row_j = matrix[j, :].copy()
                    matrix[i, :] = single[0, 0] * row_i + single[0, 1] * row_j
                    matrix[j, :] = single[1, 0] * row_i + single[1, 1] * row_j
        else:
            _, control, target = gate
            c_bit = qubits - 1 - control
            t_bit = qubits - 1 - target
            for i in range(n):
                if (i >> c_bit) & 1 == 1:
                    j = i ^ (1 << t_bit)
                    if i < j:
                        temp = matrix[i, :].copy()
                        matrix[i, :] = matrix[j, :]
                        matrix[j, :] = temp
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _check_deadline():
    return time.time() > _DEADLINE


def _known_decomposition(name, qubits, budget):
    """Return exact gate list for known gates, or None if not a known gate."""
    # CNOT
    if qubits == 2 and budget == 1:
        # Check if target is CNOT (control 0, target 1)
        # CNOT matrix: [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
        return [["cx", 0, 1]]
    
    # iSWAP
    if qubits == 2 and budget == 2:
        # iSWAP = [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
        # Decomposition: (H⊗H) CNOT (S†⊗S†) CNOT (H⊗H)
        # S† = u3(0, 0, -π/2)
        # H = u3(π/2, 0, π)
        gates = [
            ["u3", 0, math.pi / 2, 0, math.pi],
            ["u3", 1, math.pi / 2, 0, math.pi],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, -math.pi / 2],
            ["u3", 1, 0, 0, -math.pi / 2],
            ["cx", 0, 1],
            ["u3", 0, math.pi / 2, 0, math.pi],
            ["u3", 1, math.pi / 2, 0, math.pi],
        ]
        return gates
    
    # SWAP
    if qubits == 2 and budget == 3:
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
    
    # Toffoli
    if qubits == 3 and budget == 6:
        # Toffoli with 6 CNOTs
        # H = u3(π/2, 0, π), T = u3(0, 0, π/4), T† = u3(0, 0, -π/4)
        gates = [
            ["u3", 2, math.pi / 2, 0, math.pi],  # H on q2
            ["u3", 2, 0, 0, math.pi / 4],        # T on q2
            ["cx", 1, 2],
            ["u3", 1, 0, 0, -math.pi / 4],       # T† on q1
            ["u3", 2, 0, 0, math.pi / 4],        # T on q2
            ["cx", 0, 2],
            ["u3", 0, 0, 0, -math.pi / 4],       # T† on q0
            ["u3", 1, 0, 0, math.pi / 4],        # T on q1
            ["cx", 1, 2],
            ["u3", 2, 0, 0, -math.pi / 4],       # T† on q2
            ["u3", 0, 0, 0, math.pi / 4],        # T on q0
            ["u3", 1, 0, 0, -math.pi / 4],       # T† on q1
            ["cx", 0, 2],
            ["u3", 2, math.pi / 2, 0, math.pi],  # H on q2
        ]
        return gates
    
    return None


def _make_gates(layout, values, qubits):
    """Convert layout and parameter values to gate list."""
    gates = []
    k = 0
    for item in layout:
        if isinstance(item, int):
            q = item
            gates.append(["u3", q, values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            _, (c, t) = item
            gates.append(["cx", c, t])
    # Add final u3 gates
    for q in range(qubits):
        gates.append(["u3", q, values[k], values[k + 1], values[k + 2]])
        k += 3
    return gates


def _optimize_generic(problem, target, qubits, budget):
    """Optimize single-qubit rotations for generic unitaries using coordinate descent."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits)
    
    # Build layout: u3 gates before and after each CNOT
    layout = []
    for layer in range(budget):
        layout += [q for q in range(qubits)]  # u3 gates for each qubit
        layout.append(("cx", pairs[layer % len(pairs)]))
    # Count u3 positions
    u3_count = sum(1 for g in layout if isinstance(g, int)) + qubits  # + final u3s
    n_params = 3 * u3_count
    
    best = [0.0] * n_params
    best_distance = _distance(target, _build_fast(_make_gates(layout, best, qubits), qubits))
    
    # Adaptive step size coordinate descent
    step = 1.0
    max_sweeps = 30
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        improved = False
        for i in range(len(best)):
            if _check_deadline():
                break
            for trial in (best[i] + step, best[i] - step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(_make_gates(layout, values, qubits), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
                    step = max(0.01, step * 0.7)
                    break
            else:
                step = min(math.pi, step * 1.3)
        if not improved and step < 0.05:
            break
    
    # Final polish with smaller steps
    for sweep in range(10):
        if _check_deadline():
            break
        small_step = 0.01 / (1 + sweep)
        for i in range(len(best)):
            if _check_deadline():
                break
            for trial in (best[i] + small_step, best[i] - small_step):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(_make_gates(layout, values, qubits), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    break
    
    return _make_gates(layout, best, qubits)


def synthesize(problem):
    """Synthesize gates for the given target unitary."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    # Try known decomposition first
    known = _known_decomposition(name, qubits, budget)
    if known is not None:
        # Verify the known decomposition is actually correct
        target = _target(problem)
        dist = _distance(target, _build_fast(known, qubits))
        if dist < 1e-10:
            return known
    
    # Fall back to numerical optimization
    target = _target(problem)
    return _optimize_generic(problem, target, qubits, budget)
# EVOLVE-BLOCK-END
