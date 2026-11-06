"""Rewritten synthesis: fast builder + deadline-guarded multi-start coordinate descent."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 250.0  # safety margin before the 270s wall-clock kill


def _elapsed():
    return time.time() - _START_TIME


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _make_cnot_perm(qubits, control, target):
    n = 1 << qubits
    cshift = qubits - 1 - control
    tshift = qubits - 1 - target
    idx = np.arange(n)
    bit = (idx >> cshift) & 1
    return (idx ^ (bit << tshift)).astype(np.int64)


def _build(gates, qubits, cnot_perms):
    n = 1 << qubits
    m = np.eye(n, dtype=complex)
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, la = g
            c = math.cos(th * 0.5)
            s = math.sin(th * 0.5)
            single = np.array([[c, -np.exp(1j * la) * s],
                               [np.exp(1j * ph) * s, np.exp(1j * (ph + la)) * c]], dtype=complex)
            op = m.reshape([2] * qubits + [n])
            op = np.tensordot(single, op, axes=([1], [q]))
            op = np.moveaxis(op, 0, q)
            m = op.reshape(n, n)
        else:
            _, c, t = g
            m = m[cnot_perms[(c, t)], :]
    return m


def _dist(target, m):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ m)) / target.shape[0])


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    n = 1 << qubits

    # Precompute CNOT permutations for every (control, target) pair.
    cnot_perms = {}
    for c in range(qubits):
        for t in range(qubits):
            if c != t:
                cnot_perms[(c, t)] = _make_cnot_perm(qubits, c, t)

    # Build a fixed CNOT ladder (<= budget) with single-qubit rotation slots before/after.
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    layout = []
    for layer in range(budget):
        layout.append(("u3", qubits))  # a full layer of single-qubit rotations
        layout.append(("cx", pairs[layer % len(pairs)]))
    layout.append(("u3", qubits))

    n_slots = sum(1 for kind, _ in layout if kind == "u3") * qubits
    cx_positions = [i for i, (kind, _) in enumerate(layout) if kind == "cx"]

    def circuit(vals):
        gates = []
        k = 0
        for kind, payload in layout:
            if kind == "u3":
                for q in range(qubits):
                    gates.append(["u3", q, vals[k], vals[k + 1], vals[k + 2]])
                    k += 3
            else:
                gates.append(["cx", payload[0], payload[1]])
        return gates

    def dist_vals(vals):
        return _dist(target, _build(circuit(vals), qubits, cnot_perms))

    def coordinate_descent(vals, base_dist, max_sweeps):
        best = list(vals)
        bd = base_dist
        step = 1.0
        for sweep in range(max_sweeps):
            if _elapsed() > _DEADLINE - 10.0:
                break
            improved = False
            for i in range(len(best)):
                for trial in (best[i] + step, best[i] - step,
                              best[i] + random.uniform(-step, step)):
                    cand = best[:]
                    cand[i] = trial
                    d = dist_vals(cand)
                    if d < bd:
                        best, bd, improved = cand, d, True
            if improved:
                step *= 0.7
            else:
                step *= 0.5
            if step < 1e-5:
                break
        return best, bd

    # Multi-start coordinate descent.
    rng = random.Random(12345)
    best_vals = [0.0] * n_slots
    best_dist = dist_vals(best_vals)

    max_starts = 6
    for s in range(max_starts):
        if _elapsed() > _DEADLINE - 15.0:
            break
        if s == 0:
            start = [0.0] * n_slots
            start_dist = best_dist
        else:
            start = [rng.uniform(-math.pi, math.pi) for _ in range(n_slots)]
            start_dist = dist_vals(start)
        if start_dist >= best_dist:
            continue
        vals, d = coordinate_descent(start, start_dist, max_sweeps=14)
        if d < best_dist:
            best_vals, best_dist = vals, d

    # Final polish from the best.
    if _elapsed() < _DEADLINE - 20.0:
        vals, d = coordinate_descent(best_vals, best_dist, max_sweeps=10)
        if d < best_dist:
            best_vals, best_dist = vals, d

    return circuit(best_vals)
# EVOLVE-BLOCK-END
