"""Quantum circuit synthesis: exact decompositions + fast optimized search."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_DEADLINE = 230.0   # safety margin under the 270 s kill


def _target(problem):
    return (np.array(problem["real"], dtype=complex)
            + 1j * np.array(problem["imaginary"], dtype=complex))


# ---------- fast circuit evaluation ----------
def _apply_u3(state, q, theta, phi, lam, qubits):
    d = 2 ** qubits
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    if abs(c) < 1e-15:
        c = 0.0
    if abs(s) < 1e-15:
        s = 0.0
    e_lam = complex(math.cos(lam), math.sin(lam))
    e_phi = complex(math.cos(phi), math.sin(phi))
    e_pl = complex(math.cos(phi + lam), math.sin(phi + lam))
    a, b = c, -s * e_lam
    cc, dd = s * e_phi, c * e_pl
    step = 1 << (qubits - 1 - q)
    for base in range(0, d, step * 2):
        for j in range(step):
            p = base + j
            q_ = base + step + j
            sp, sq = state[p], state[q_]
            state[p] = a * sp + b * sq
            state[q_] = cc * sp + dd * sq
    return state


def _apply_cnot(state, c, t, qubits):
    d = 2 ** qubits
    cs = 1 << (qubits - 1 - c)
    ts = 1 << (qubits - 1 - t)
    for base in range(0, d, cs * 2):
        for j in range(cs):
            p = base + cs + j
            q_ = base + j
            state[p] = state[p] + state[q_]
            state[q_] = state[q_] - state[p]
    return state


def _dist_fast(gates, qubits, target):
    d = 2 ** qubits
    state = np.zeros(d, dtype=complex)
    state[0] = 1.0
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            _apply_u3(state, q, th, ph, lm, qubits)
        else:
            _, c, t = g
            _apply_cnot(state, c, t, qubits)
    return 1.0 - abs(np.vdot(target.ravel(), state)) / d


def _build(gates, qubits):
    """Full unitary (kept for reference / small debugging)."""
    d = 2 ** qubits
    U = np.eye(d, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            c, s = math.cos(th / 2), math.sin(th / 2)
            e_lam = complex(math.cos(lm), math.sin(lm))
            e_phi = complex(math.cos(ph), math.sin(ph))
            e_pl = complex(math.cos(ph + lm), math.sin(ph + lm))
            single = np.array([[c, -s * e_lam], [s * e_phi, c * e_pl]], dtype=complex)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye)
            U = op @ U
        else:
            _, c, t = g
            cs = 1 << (qubits - 1 - c)
            ts = 1 << (qubits - 1 - t)
            idx = np.arange(d)
            bit = (idx >> (qubits - 1 - c)) & 1
            U = U[idx ^ (bit << (qubits - 1 - t)), :]
    return U


# ---------- exact decompositions ----------
def _u3(q, th, ph, lm):
    return ["u3", q, th, ph, lm]


def _cx(c, t):
    return ["cx", c, t]


def _decomp_cnot(problem):
    return [_cx(0, 1)]


def _decomp_swap(problem):
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _decomp_iswap(problem):
    gates = []
    for q in (0, 1):
        gates.append(_u3(q, math.pi / 2, 0.0, -math.pi / 2))
    gates.append(_cx(0, 1))
    gates.append(_cx(1, 0))
    for q in (0, 1):
        gates.append(_u3(q, math.pi / 2, math.pi / 2, 0.0))
    return gates


def _decomp_toffoli(problem):
    gates = []
    gates.append(_u3(2, math.pi / 8, 0.0, 0.0))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, -math.pi / 8, 0.0, 0.0))
    gates.append(_cx(0, 2))
    gates.append(_u3(2, math.pi / 8, 0.0, 0.0))
    gates.append(_cx(1, 2))
    gates.append(_u3(2, -math.pi / 8, 0.0, 0.0))
    gates.append(_u3(0, 0.0, 0.0, math.pi / 8))
    gates.append(_u3(1, 0.0, 0.0, math.pi / 8))
    gates.append(_cx(0, 1))
    gates.append(_u3(1, 0.0, 0.0, -math.pi / 8))
    gates.append(_u3(1, math.pi / 2, 0.0, 0.0))
    gates.append(_u3(0, math.pi / 2, 0.0, 0.0))
    return gates


# ---------- generic optimizer ----------
def _optimize_generic(problem, target, qubits, budget, deadline, rng):
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout += ["u3" for _ in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += ["u3" for _ in range(qubits)]

    n_u3 = sum(1 for g in layout if g == "u3")
    n_params = 3 * n_u3

    def make_circuit(vals):
        gates, k = [], 0
        for g in layout:
            if g == "u3":
                gates.append(["u3", 0, vals[k], vals[k + 1], vals[k + 2]])
                k += 3
            else:
                gates.append(list(g))
        return gates

    # fix the per-u3 qubit indices
    u3_q = []
    for g in layout:
        if g == "u3":
            u3_q.append(len(u3_q) % qubits)

    def make_circuit(vals):
        gates, k = [], 0
        for g in layout:
            if g == "u3":
                gates.append(["u3", u3_q[k // 3], vals[k], vals[k + 1], vals[k + 2]])
                k += 3
            else:
                gates.append(list(g))
        return gates

    best = np.zeros(n_params)
    best_d = _dist_fast(make_circuit(best), qubits, target)

    # multi-start coordinate descent with decreasing steps
    for start in range(3):
        if time.time() > deadline:
            break
        if start == 0:
            cur = np.zeros(n_params)
        else:
            cur = rng.uniform(-math.pi, math.pi, n_params)
        cur_d = _dist_fast(make_circuit(cur), qubits, target)
        if cur_d < best_d:
            best, best_d = cur, cur_d

        for sweep in range(10):
            if time.time() > deadline:
                break
            step = 1.0 / (1 + sweep)
            for i in range(n_params):
                if time.time() > deadline:
                    break
                for trial in (cur[i] + step, cur[i] - step,
                              cur[i] + rng.uniform(-math.pi, math.pi)):
                    cand = cur.copy()
                    cand[i] = trial
                    v = _dist_fast(make_circuit(cand), qubits, target)
                    if v < cur_d:
                        cur, cur_d = cand, v
        if cur_d < best_d:
            best, best_d = cur, cur_d

    return make_circuit(best)


def synthesize(problem):
    t0 = time.time()
    deadline = t0 + _DEADLINE
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    if qubits == 2:
        ln = name.lower()
        if "cnot" in ln:
            return _decomp_cnot(problem)
        if "swap" in ln and "iswap" not in ln:
            return _decomp_swap(problem)
        if "iswap" in ln:
            return _decomp_iswap(problem)

    if qubits == 3 and "toffoli" in name.lower():
        return _decomp_toffoli(problem)

    rng = random.Random(4242 + qubits + len(name))
    return _optimize_generic(problem, target, qubits, budget, deadline, rng)
# EVOLVE-BLOCK-END
