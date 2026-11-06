"""Quantum circuit synthesis: exact algebraic 2q decomposition + fast coordinate descent."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 258.0  # Safety margin before 270s wall-clock kill


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


# ---------- fast matrix construction ----------
def _apply_u3_to_matrix(M, q, qubits, theta, phi, lam):
    n = 2 ** qubits
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    shift = 1 << (qubits - 1 - q)
    out = np.empty_like(M)
    for i in range(n):
        if (i & shift) == 0:
            j = i | shift
            ri = M[i]
            rj = M[j]
            out[i] = c * ri - e1 * s * rj
            out[j] = e2 * s * ri + e2 * e1 * c * rj
        else:
            out[i] = M[i]
    return out


def _apply_cnot_to_matrix(M, control, target, qubits):
    n = 2 ** qubits
    cs = 1 << (qubits - 1 - control)
    ts = 1 << (qubits - 1 - target)
    perm = np.arange(n)
    for i in range(n):
        if (i & cs):
            perm[i] = i ^ ts
    return M[perm]


def _build_matrix(gates, qubits):
    n = 2 ** qubits
    M = np.eye(n, dtype=complex)
    for g in gates:
        if g[0] == "u3":
            M = _apply_u3_to_matrix(M, g[1], qubits, g[2], g[3], g[4])
        else:
            M = _apply_cnot_to_matrix(M, g[1], g[2], qubits)
    return M


def _distance(target, M):
    n = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ M)) / n)


# ---------- exact known-gate decompositions ----------
def _exact_cnot():
    return [["cx", 0, 1]]


def _exact_iswap():
    h = math.pi / 2
    return [["u3", 0, h, 0.0, 0.0],
            ["u3", 1, h, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 0, -h, 0.0, 0.0],
            ["u3", 1, -h, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 0, h, 0.0, 0.0],
            ["u3", 1, h, 0.0, 0.0]]


def _exact_swap():
    return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]


def _exact_toffoli():
    e = math.pi / 8
    return [["u3", 2, e, 0.0, 0.0],
            ["cx", 1, 2],
            ["u3", 2, -e, 0.0, 0.0],
            ["cx", 0, 2],
            ["u3", 2, e, 0.0, 0.0],
            ["cx", 1, 2],
            ["u3", 2, -e, 0.0, 0.0],
            ["u3", 1, e, 0.0, 0.0],
            ["u3", 2, e, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 1, -e, 0.0, 0.0],
            ["u3", 2, -e, 0.0, 0.0],
            ["cx", 0, 2],
            ["u3", 1, e, 0.0, 0.0],
            ["u3", 2, e, 0.0, 0.0],
            ["cx", 1, 2],
            ["u3", 2, -e, 0.0, 0.0],
            ["u3", 1, 3 * e, 0.0, math.pi / 2],
            ["u3", 2, 3 * e, 0.0, math.pi / 2],
            ["u3", 0, math.pi / 2, 0.0, 0.0]]


def _identify_known(target, qubits):
    tol = 1e-8
    if qubits == 2:
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        if np.max(np.abs(target - cnot)) < tol:
            return _exact_cnot()
        iswap = np.array([[0, 0, 0, 1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]], dtype=complex)
        if np.max(np.abs(target - iswap)) < tol:
            return _exact_iswap()
        swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        if np.max(np.abs(target - swap)) < tol:
            return _exact_swap()
    elif qubits == 3:
        t = np.eye(8, dtype=complex)
        t[6, 6] = 0.0
        t[6, 7] = 1.0
        t[7, 6] = 1.0
        t[7, 7] = 0.0
        if np.max(np.abs(target - t)) < tol:
            return _exact_toffoli()
    return None


# ---------- exact algebraic 2-qubit (3 CNOT) decomposition ----------
def _solve_u3(M):
    d = np.linalg.det(M)
    if abs(d) > 1e-12:
        M = M / np.sqrt(d)
    c = np.real(M[0, 0])
    c = max(-1.0, min(1.0, c))
    th = 2.0 * math.acos(c)
    s = math.sin(th / 2)
    if s < 1e-9:
        return [0.0, 0.0, 0.0]
    lam = np.angle(-M[0, 1] / s)
    phi = np.angle(M[1, 0] / s)
    return [th, phi, lam]


def _u3mat(p):
    th, ph, lm = p
    c, s = math.cos(th / 2), math.sin(th / 2)
    e1 = np.exp(1j * lm)
    e2 = np.exp(1j * ph)
    return np.array([[c, -e1 * s], [e2 * s, e2 * e1 * c]], dtype=complex)


def _exact_2q(target):
    CX01 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    CX10 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
    U = target
    D = np.eye(4, dtype=complex)
    for _ in range(8):
        M = U @ D
        A0 = M[0:2, 0:2]
        A1 = M[2:4, 2:4]
        pA0 = _solve_u3(A0)
        pA1 = _solve_u3(A1)
        if max(abs(A0 - _u3mat(pA0)), abs(A1 - _u3mat(pA1))) > 1e-6:
            return None
        D = np.kron(_u3mat(pA0), _u3mat(pA1))
        U = U @ CX10
        M = U @ D
        B0 = M[0:2, 0:2]
        B1 = M[2:4, 2:4]
        pB0 = _solve_u3(B0)
        pB1 = _solve_u3(B1)
        if max(abs(B0 - _u3mat(pB0)), abs(B1 - _u3mat(pB1))) > 1e-6:
            return None
        D = np.kron(_u3mat(pB0), _u3mat(pB1)) @ D
        U = U @ CX01
        M = U @ D
        C0 = M[0:2, 0:2]
        C1 = M[2:4, 2:4]
        pC0 = _solve_u3(C0)
        pC1 = _solve_u3(C1)
        if max(abs(C0 - _u3mat(pC0)), abs(C1 - _u3mat(pC1))) > 1e-6:
            return None
        D = np.kron(_u3mat(pC0), _u3mat(pC1)) @ D
        U = U @ CX01
        M = U @ D
        F0 = M[0:2, 0:2]
        F1 = M[2:4, 2:4]
        pF0 = _solve_u3(F0)
        pF1 = _solve_u3(F1)
        if max(abs(F0 - _u3mat(pF0)), abs(F1 - _u3mat(pF1))) > 1e-6:
            return None
        gates = []
        for q, p in ((0, pF0), (1, pF1)):
            gates.append(["u3", q, p[0], p[1], p[2]])
        gates.append(["cx", 0, 1])
        for q, p in ((0, pC0), (1, pC1)):
            gates.append(["u3", q, p[0], p[1], p[2]])
        gates.append(["cx", 1, 0])
        for q, p in ((0, pB0), (1, pB1)):
            gates.append(["u3", q, p[0], p[1], p[2]])
        gates.append(["cx", 0, 1])
        for q, p in ((0, pA0), (1, pA1)):
            gates.append(["u3", q, p[0], p[1], p[2]])
        return gates
    return None


# ---------- coordinate-descent fallback (2q) ----------
def _two_layer_2q(target, max_time=6.0):
    t0 = time.time()
    rng = random.Random(12345)
    cnot = [("cx", 0, 1), ("cx", 1, 0), ("cx", 0, 1)]
    order = []
    for _ in range(4):
        for q in range(2):
            order.append(("u3", q))
    for i in range(3):
        order.append(cnot[i])

    def build(vals):
        gates, k = [], 0
        for g in order:
            if g[0] == "u3":
                gates.append(["u3", g[1], vals[k], vals[k + 1], vals[k + 2]])
                k += 3
            else:
                gates.append(list(g))
        return gates

    best = [0.0] * 24
    best_d = _distance(target, _build_matrix(build(best), 2))
    for start in range(8):
        if time.time() - t0 > max_time or _check_deadline():
            break
        vals = [0.0] * 24 if start == 0 else [rng.uniform(-math.pi, math.pi) for _ in range(24)]
        d = _distance(target, _build_matrix(build(vals), 2))
        for sweep in range(12):
            if time.time() - t0 > max_time or _check_deadline():
                break
            step = math.pi / (2 ** sweep)
            imp = False
            for i in range(24):
                for trial in (vals[i] + step, vals[i] - step):
                    nv = vals[:]
                    nv[i] = trial
                    dd = _distance(target, _build_matrix(build(nv), 2))
                    if dd < d:
                        vals = nv
                        d = dd
                        imp = True
                if sweep >= 5:
                    nv = vals[:]
                    nv[i] = vals[i] + rng.uniform(-step, step)
                    dd = _distance(target, _build_matrix(build(nv), 2))
                    if dd < d:
                        vals = nv
                        d = dd
                        imp = True
            if not imp and sweep > 4:
                break
        if d < best_d:
            best_d = d
            best = vals[:]
    return build(best), best_d


# ---------- coordinate descent (3q) ----------
def _three_layer_3q(target, budget, max_time=18.0):
    t0 = time.time()
    rng = random.Random(777 + budget)
    pairs = [(0, 1), (1, 2), (0, 2), (1, 0), (2, 1), (2, 0)]
    order = []
    for layer in range(budget):
        for q in range(3):
            order.append(("u3", q))
        order.append(("cx",) + pairs[layer % len(pairs)])
    for q in range(3):
        order.append(("u3", q))
    if len(order) > 400:
        while len(order) > 400:
            for i in range(len(order) - 1, -1, -1):
                if order[i][0] == "u3":
                    order.pop(i)
                    break
            else:
                break

    def build(vals):
        gates, k = [], 0
        for g in order:
            if g[0] == "u3":
                gates.append(["u3", g[1], vals[k], vals[k + 1], vals[k + 2]])
                k += 3
            else:
                gates.append(list(g))
        return gates

    npar = 3 * sum(1 for g in order if g[0] == "u3")
    best = [0.0] * npar
    best_d = _distance(target, _build_matrix(build(best), 3))
    for start in range(5):
        if time.time() - t0 > max_time or _check_deadline():
            break
        vals = [0.0] * npar if start == 0 else [rng.uniform(-math.pi, math.pi) for _ in range(npar)]
        d = _distance(target, _build_matrix(build(vals), 3))
        for sweep in range(9):
            if time.time() - t0 > max_time or _check_deadline():
                break
            step = math.pi / (2 ** sweep)
            imp = False
            for i in range(npar):
                if time.time() - t0 > max_time or _check_deadline():
                    break
                for trial in (vals[i] + step, vals[i] - step):
                    nv = vals[:]
                    nv[i] = trial
                    dd = _distance(target, _build_matrix(build(nv), 3))
                    if dd < d:
                        vals = nv
                        d = dd
                        imp = True
                if sweep >= 4:
                    nv = vals[:]
                    nv[i] = vals[i] + rng.uniform(-step, step)
                    dd = _distance(target, _build_matrix(build(nv), 3))
                    if dd < d:
                        vals = nv
                        d = dd
                        imp = True
            if not imp and sweep > 3:
                break
        if d < best_d:
            best_d = d
            best = vals[:]
    return build(best), best_d


def _optimize_generic(target, qubits, budget):
    if qubits == 2:
        gates = _exact_2q(target)
        if gates is not None:
            d = _distance(target, _build_matrix(gates, 2))
            if d < 1e-6:
                return gates, d
        return _two_layer_2q(target)
    return _three_layer_3q(target, budget)


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    known = _identify_known(target, qubits)
    if known is not None:
        cx = sum(1 for g in known if g[0] == "cx")
        if cx <= budget and len(known) <= 400:
            return known

    gates, dist = _optimize_generic(target, qubits, budget)
    cx = sum(1 for g in gates if g[0] == "cx")
    if cx > budget or len(gates) > 400:
        return [["u3", 0, 0.0, 0.0, 0.0]]
    return gates
# EVOLVE-BLOCK-END
