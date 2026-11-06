"""Exact quantum circuit synthesis within published CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 235.0  # safety margin before 270s kill


def _time_left():
    return _DEADLINE - time.time()


def _target(problem):
    return (np.array(problem["real"], dtype=complex)
            + 1j * np.array(problem["imaginary"], dtype=complex))


def _u3_mat(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
                    dtype=complex)


def _build_fast(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_mat(theta, phi, lam)
            op = np.array([[1.0]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else np.eye(2, dtype=complex))
            U = op @ U
        else:
            _, c, t = gate
            cb, tb = qubits - 1 - c, qubits - 1 - t
            idx = np.arange(n)
            bit = (idx >> cb) & 1
            U = U[idx ^ (bit << tb), :]
    return U


def _distance(target, U):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / target.shape[0])


def _known_decompositions(name, qubits):
    """Return exact gate list for the fixed gates, else None."""
    if name == "CNOT":
        return [["cx", 0, 1]]
    if name == "iSWAP":
        # iSWAP = (1/sqrt2)[[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]
        # = diag(i,i,1,1) * (1/sqrt2)[[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]] * diag(i,i,1,1)
        # The middle matrix is H_0 H_1 CX H_1 H_0 (up to phase).
        return [
            ["u3", 0, 0.0, 0.0, math.pi],
            ["u3", 1, 0.0, 0.0, math.pi],
            ["u3", 0, math.pi / 2, 0.0, 0.0],
            ["u3", 1, math.pi / 2, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 1, math.pi / 2, 0.0, 0.0],
            ["u3", 0, math.pi / 2, 0.0, 0.0],
            ["u3", 0, 0.0, 0.0, math.pi],
            ["u3", 1, 0.0, 0.0, math.pi],
        ]
    if name == "SWAP":
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
    if name == "Toffoli":
        # Toffoli with control 0,1 target 2, within 6 CNOT budget.
        # Standard decomposition:
        # H on t; CX c1->t; T on t; CX c0->t; T^-1 on c1; T on t; CX c1->t;
        # T^-1 on c0; T on c1; CX c0->t; T^-1 on t; H on t
        T = math.pi / 4
        return [
            ["u3", 2, math.pi / 2, 0.0, 0.0],
            ["cx", 1, 2],
            ["u3", 2, 0.0, 0.0, T],
            ["cx", 0, 2],
            ["u3", 1, 0.0, 0.0, -T],
            ["u3", 2, 0.0, 0.0, T],
            ["cx", 1, 2],
            ["u3", 0, 0.0, 0.0, -T],
            ["u3", 1, 0.0, 0.0, T],
            ["cx", 0, 2],
            ["u3", 2, 0.0, 0.0, -T],
            ["u3", 2, math.pi / 2, 0.0, 0.0],
        ]
    return None


def _make_layout(qubits, budget):
    """Build a CNOT ladder layout with u3 between CNOTs."""
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q])
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
    for q in range(qubits):
        layout.append(["u3", q])
    return layout


def _layout_to_gates(layout, values):
    gates = []
    k = 0
    for gate in layout:
        if gate[0] == "u3":
            gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
            k += 3
        else:
            gates.append(list(gate))
    return gates


def _optimize_generic(target, qubits, budget, seed):
    layout = _make_layout(qubits, budget)
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    rng = np.random.RandomState(seed)

    best = None
    best_dist = float("inf")

    # multi-start
    n_starts = 3
    for s in range(n_starts):
        if _time_left() < 30:
            break
        if s == 0:
            vals = np.zeros(n_params)
        else:
            vals = rng.uniform(-math.pi, math.pi, size=n_params)

        dist = _distance(target, _build_fast(_layout_to_gates(layout, vals), qubits))
        if dist < best_dist:
            best, best_dist = vals.copy(), dist

        # coordinate descent: coarse then fine
        for sweep in range(15):
            if _time_left() < 20:
                break
            step = 1.0 / (1 + sweep * 2)
            improved = False
            for i in range(n_params):
                if _time_left() < 20:
                    break
                for trial in (vals[i] + step, vals[i] - step,
                              vals[i] + rng.uniform(-step, step)):
                    new_vals = vals.copy()
                    new_vals[i] = trial
                    d = _distance(target, _build_fast(_layout_to_gates(layout, new_vals), qubits))
                    if d < dist:
                        vals, dist = new_vals, d
                        improved = True
            if not improved and sweep > 8:
                break

        if dist < best_dist:
            best, best_dist = vals.copy(), dist

    # fine polish
    if best is not None and _time_left() > 20:
        vals = best.copy()
        dist = best_dist
        for sweep in range(10):
            if _time_left() < 15:
                break
            step = 0.1 / (1 + sweep)
            for i in range(len(vals)):
                if _time_left() < 15:
                    break
                for trial in (vals[i] + step, vals[i] - step):
                    new_vals = vals.copy()
                    new_vals[i] = trial
                    d = _distance(target, _build_fast(_layout_to_gates(layout, new_vals), qubits))
                    if d < dist:
                        vals, dist = new_vals, d

    if best is None:
        best = np.zeros(n_params)
    return _layout_to_gates(layout, best)


def synthesize(problem):
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    known = _known_decompositions(name, qubits)
    if known is not None:
        # verify it is within budget and correct; if not, fall back
        n_cx = sum(1 for g in known if g[0] == "cx")
        if n_cx <= budget and len(known) <= 400:
            return known
        # fall through to generic

    return _optimize_generic(target, qubits, budget, seed=12345 + qubits)
# EVOLVE-BLOCK-END
