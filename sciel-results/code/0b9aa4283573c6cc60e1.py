"""
Strategy:
1. Exact hardcoded decompositions for known gates (CNOT, iSWAP, SWAP, Toffoli) — no optimization.
2. For generic unitaries, use a structured CNOT ladder with alternating control/target pairs,
   then optimize single-qubit rotations via a fast Nelder-Mead implementation with adaptive
   step sizes and a tight deadline guard.
3. The key structural change: use a more expressive gate layout (interleaved CNOTs on both
   adjacent pairs for 3-qubit, and a balanced layout for 2-qubit) and a faster local optimizer
   (Nelder-Mead with reflection/expansion/contraction) instead of pure coordinate descent,
   which can escape shallow local minima that coordinate descent gets stuck in.
"""
# EVOLVE-BLOCK-START
import math
import time

import numpy as np

_START_TIME = time.time()
_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build the unitary from a list of gates."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
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
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_same_unitary(u1, u2, tol=1e-9):
    """Check if two unitaries are the same up to a global phase."""
    if u1.shape != u2.shape:
        return False
    for i in range(u1.shape[0]):
        for j in range(u1.shape[1]):
            if abs(u1[i, j]) > tol:
                phase = u1[i, j] / u2[i, j]
                if np.allclose(u1, phase * u2, atol=tol):
                    return True
    return False


def _known_decomposition(problem):
    """Return exact hardcoded decomposition for known standard gates."""
    target = _target(problem)
    qubits = problem["qubits"]

    if qubits == 2:
        # CNOT
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _is_same_unitary(target, cnot):
            return [['cx', 0, 1]]

        # iSWAP
        iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        if _is_same_unitary(target, iswap):
            return [
                ['u3', 0, 0, 0, math.pi/2],
                ['u3', 1, 0, 0, math.pi/2],
                ['cx', 0, 1],
                ['cx', 1, 0],
                ['u3', 0, 0, -math.pi/2, 0],
                ['u3', 1, 0, -math.pi/2, 0]
            ]

        # SWAP
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _is_same_unitary(target, swap):
            return [
                ['cx', 0, 1],
                ['cx', 1, 0],
                ['cx', 0, 1]
            ]

    elif qubits == 3:
        # Toffoli
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[6, 6] = 0
        toffoli[7, 7] = 0
        if _is_same_unitary(target, toffoli):
            return [
                ['u3', 2, math.pi/2, 0, math.pi],
                ['u3', 2, 0, 0, math.pi/4],
                ['cx', 0, 2],
                ['u3', 1, 0, 0, -math.pi/4],
                ['cx', 0, 1],
                ['u3', 2, 0, 0, math.pi/4],
                ['u3', 1, 0, 0, -math.pi/4],
                ['cx', 0, 1],
                ['u3', 1, 0, 0, math.pi/4],
                ['u3', 2, math.pi/2, 0, math.pi],
                ['cx', 1, 2],
                ['u3', 2, 0, 0, -math.pi/4],
                ['cx', 0, 2],
                ['u3', 2, 0, 0, math.pi/4],
                ['u3', 2, math.pi/2, 0, math.pi],
            ]

    return None


def _nelder_mead(f, x0, max_iter=2000, tol=1e-12, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """
    Fast Nelder-Mead simplex optimizer.
    x0: initial point (list of floats)
    f: objective function
    """
    n = len(x0)
    # Initialize simplex
    simplex = [list(x0)]
    for i in range(n):
        xi = list(x0)
        xi[i] = xi[i] + 0.1 if abs(xi[i]) > 1e-8 else 0.1
        simplex.append(xi)

    vals = [f(v) for v in simplex]

    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break

        # Sort simplex by function value
        order = sorted(range(n + 1), key=lambda k: vals[k])
        simplex = [simplex[k] for k in order]
        vals = [vals[k] for k in order]

        if abs(vals[-1] - vals[0]) < tol:
            break

        # Centroid of all but worst point
        centroid = [sum(simplex[k][i] for k in range(n)) / n for i in range(n)]

        # Reflection
        xr = [centroid[i] + alpha * (centroid[i] - simplex[-1][i]) for i in range(n)]
        fr = f(xr)

        if fr < vals[0]:
            # Expansion
            xe = [centroid[i] + gamma * (xr[i] - centroid[i]) for i in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            # Contraction
            xc = [centroid[i] + rho * (simplex[-1][i] - centroid[i]) for i in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                # Shrink
                for k in range(1, n + 1):
                    simplex[k] = [simplex[0][i] + sigma * (simplex[k][i] - simplex[0][i]) for i in range(n)]
                    vals[k] = f(simplex[k])

    # Return best point
    best_idx = min(range(n + 1), key=lambda k: vals[k])
    return simplex[best_idx], vals[best_idx]


def _build_layout_2qubit(budget):
    """Build a balanced CNOT layout for 2 qubits."""
    layout = []
    u3_indices = []
    for layer in range(budget):
        for q in range(2):
            u3_indices.append(len(layout))
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", 0, 1])
    for q in range(2):
        u3_indices.append(len(layout))
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    return layout, u3_indices


def _build_layout_3qubit(budget):
    """Build an interleaved CNOT layout for 3 qubits."""
    layout = []
    u3_indices = []
    pairs = [(0, 1), (1, 2)]
    for layer in range(budget):
        for q in range(3):
            u3_indices.append(len(layout))
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", *pairs[layer % 2]])
    for q in range(3):
        u3_indices.append(len(layout))
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    return layout, u3_indices


def _optimize_generic(problem):
    """Optimize single-qubit rotations for a generic unitary using Nelder-Mead."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    if qubits == 2:
        layout, u3_indices = _build_layout_2qubit(budget)
    else:
        layout, u3_indices = _build_layout_3qubit(budget)

    num_u3 = len(u3_indices)
    num_params = 3 * num_u3

    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    def objective(values):
        return _distance(target, _build(circuit(values), qubits))

    import random
    rng = random.Random(4242 + qubits * 100 + budget)
    x0 = [rng.uniform(-math.pi, math.pi) for _ in range(num_params)]

    best_x, best_dist = _nelder_mead(objective, x0)

    # Try a few more restarts if time permits and distance is still poor
    restarts = 3
    for r in range(1, restarts + 1):
        if time.time() > _DEADLINE - 5.0:
            break
        if best_dist < 1e-8:
            break
        x0_r = [rng.uniform(-math.pi, math.pi) for _ in range(num_params)]
        x_r, dist_r = _nelder_mead(objective, x0_r)
        if dist_r < best_dist:
            best_x = x_r
            best_dist = dist_r

    return circuit(best_x)


def synthesize(problem):
    qubits, budget = problem["qubits"], problem["cnot_budget"]

    # 1. Check for known gates
    known = _known_decomposition(problem)
    if known is not None:
        return known

    # 2. Generic unitary: optimize with Nelder-Mead
    return _optimize_generic(problem)
# EVOLVE-BLOCK-END
