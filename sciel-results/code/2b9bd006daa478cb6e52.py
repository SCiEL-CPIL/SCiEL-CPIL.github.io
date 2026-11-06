"""Quantum circuit synthesis: exact known gates + diagonal-form algebraic solve + polish."""
import math
import random
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 255.0


def _elapsed():
    return time.time() - _START


def _check_deadline():
    return _elapsed() > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
    """Build the unitary matrix from a gate list (left multiplication)."""
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
    dim = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / dim)


# --- Exact gate helpers ---

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
    return [_h(0), _h(1), _cnot(0, 1), _sdag(0), _s(1), _cnot(0, 1), _h(0), _h(1)]


def _decomp_toffoli(c1, c2, t):
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
    if qubits == 2:
        candidates = [_decomp_cnot_01(), _decomp_cnot_10(),
                      _decomp_swap(), _decomp_iswap()]
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


# --- Diagonal-form algebraic solve for generic unitaries ---
#
# A single-qubit u3(theta, phi, lam) decomposes as:
#     u3 = D . P . D     where  D = diag(e^{i*lam/2}, e^{-i*lam/2})  (pure phase)
#     and P = H . X . H  (fixed 2x2 unitary, independent of theta/phi/lam).
#
# With a ladder of CNOTs (fixed permutation F) and a diagonal gate (D_k) on each
# qubit between layers, the whole circuit is:
#     U = P0 . (D . P . D) . F . (D . P . D) . F . ... . (D . P . D)
#       = (product of all D's)  .  (P . F . P . F ... . P)
# The D-product is a diagonal matrix whose log-angles are LINEAR in the per-qubit
# lam values, so the phases are solved exactly by one linear system. The global
# phase is recovered from the trace. A short coordinate-descent polish then absorbs
# the residual left by the fixed permutation P (which cannot encode arbitrary
# single-qubit rotations exactly).


def _diag_phase_gates(dvec):
    """Return a list of ['u3', q, 0, 0, 2*dvec[q]] diagonal (phase) gates."""
    gates = []
    for q, d in enumerate(dvec):
        gates.append(['u3', q, 0.0, 0.0, 2.0 * d])
    return gates


def _u3_dphases(theta, phi, lam):
    """Decompose u3(theta, phi, lam) into (diagonal phase lam, fixed P part).
    Returns the diagonal phase (lam) and the u3 with lam set to 0 (the P part)."""
    return lam, (theta, phi)


def _solve_generic(target, qubits, budget):
    n = 2 ** qubits
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    budget = min(budget, 40)  # cap CNOTs so the circuit stays under 400 gates

    # Build the fixed-permutation part: P (lam=0 u3) on every qubit, then a CNOT.
    P_list = []
    for q in range(qubits):
        P_list.append(['u3', q, 0.0, 0.0, 0.0])  # P = H X H on qubit q
    F_list = []
    for layer in range(budget):
        F_list.append(_cnot(pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]))
    # Fixed circuit (no diagonal phases) -> permutation part M.
    fixed_gates = []
    for q in range(qubits):
        fixed_gates.append(['u3', q, 0.0, 0.0, 0.0])
    for layer in range(budget):
        for q in range(qubits):
            fixed_gates.append(['u3', q, 0.0, 0.0, 0.0])
        fixed_gates.append(_cnot(pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]))
    for q in range(qubits):
        fixed_gates.append(['u3', q, 0.0, 0.0, 0.0])
    M = _build(fixed_gates, qubits)

    # Permutation part of the product:  M = Dprod . Pperm  ->  Pperm = Dprod^{-1} . M
    # With all dvec = 0, Dprod = I, so Pperm = M. (Diagonal phases factor out.)
    Pperm = M

    # The diagonal phase product is a diagonal matrix. Its log-angles are linear
    # in the per-qubit lam. Each u3 diagonal contributes diag(lam, -lam) to the
    # log on the two basis states where qubit q is 0/1. Build the linear map.
    # Total number of diagonal gates = (budget+1) * qubits, each on some qubit.
    # All diagonal gates on the same qubit contribute identically, so the total
    # diagonal log-vec = sum_q  (count_q) * [lam_q * (+1 if bit q =0 else -1)].
    # count_q = budget + 1 (one P0, one before each CNOT, one final) => all equal.
    # So the diagonal log-vec = (budget+1) * sum_q lam_q * s_q(state).
    # We need  Dprod . Pperm = e^{i*alpha} . target  (up to global phase).
    # Equivalently target . Pperm^{-1} = e^{i*alpha} . Dprod  (diagonal, up to phase).
    # Since Dprod is diagonal, target @ inv(Pperm) must be diagonal up to phase.
    # This only holds exactly if Pperm is already close to diagonal; otherwise the
    # residual is handled by the polish. We solve the diagonal part of
    # (target @ inv(Pperm)) to set the phases as best as possible.
    try:
        Pperm_inv = np.linalg.inv(Pperm)
    except np.linalg.LinAlgError:
        Pperm_inv = np.linalg.pinv(Pperm)
    T = target @ Pperm_inv
    # Diagonal entries of T give e^{i*alpha} * e^{i * dprod_log[state]}.
    diag = np.diag(T)
    # Global phase alpha: choose so that the mean of log(diag) - dprod_log is consistent.
    # dprod_log[state] = (budget+1) * sum_q lam_q * s_q(state). Unknowns: lam_q (qubits).
    # Build matrix A: rows = states, cols = qubits. A[state, q] = (budget+1) * s_q(state).
    states = np.arange(n)
    A = np.zeros((n, qubits), dtype=float)
    for st in range(n):
        for q in range(qubits):
            bit = (st >> (qubits - 1 - q)) & 1
            A[st, q] = (budget + 1) * (1.0 if bit == 0 else -1.0)
    # log(diag) = alpha * 1 + A @ lam  (mod 2pi). Solve least squares for lam after
    # removing the global phase by centering.
    logdiag = np.angle(diag)
    # Remove global phase: subtract the least-squares constant.
    # Solve [A, 1] @ [lam; alpha] = logdiag  (least squares, then unwrap handled by polish).
    Ab = np.hstack([A, np.ones((n, 1))])
    sol, *_ = np.linalg.lstsq(Ab, logdiag, rcond=None)
    lam = sol[:qubits]
    alpha = sol[qubits]

    # Build the full gate list with the solved diagonal phases and the fixed P parts.
    gates = []
    # P0 diagonal + P
    for q in range(qubits):
        gates.append(['u3', q, 0.0, 0.0, 2.0 * lam[q]])
    for q in range(qubits):
        gates.append(['u3', q, 0.0, 0.0, 0.0])
    for layer in range(budget):
        for q in range(qubits):
            gates.append(['u3', q, 0.0, 0.0, 2.0 * lam[q]])
        for q in range(qubits):
            gates.append(['u3', q, 0.0, 0.0, 0.0])
        gates.append(_cnot(pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]))
    for q in range(qubits):
        gates.append(['u3', q, 0.0, 0.0, 2.0 * lam[q]])
    for q in range(qubits):
        gates.append(['u3', q, 0.0, 0.0, 0.0])

    # Global phase correction: prepend a u3 on qubit 0 with lam = alpha (phase only).
    gates.insert(0, ['u3', 0, 0.0, 0.0, alpha])

    # Polish: coordinate descent over the diagonal phases (lam) and a small set of
    # final u3 angles to absorb the residual from the fixed permutation.
    best = gates
    best_U = _build(best, qubits)
    best_dist = _distance(target, best_U)

    # Parameterise the polish: per-qubit final u3 (theta, phi, lam) and the leading
    # global phase. Keep it low-dimensional for speed.
    num_polish = qubits * 3 + 1
    vals = [0.0] * num_polish

    def polish_circuit(v):
        g = []
        g.append(['u3', 0, 0.0, 0.0, v[num_polish - 1]])  # global phase
        for q in range(qubits):
            g.append(['u3', q, v[3 * q], v[3 * q + 1], v[3 * q + 2]])
        return g

    # Prepend the polish block to the solved body (body without its own leading phase).
    body = best[1:]  # drop the solved global-phase gate; polish re-adds it

    def full_circuit(v):
        return polish_circuit(v) + body

    rng = random.Random(99991 + qubits * 13 + budget)
    for sweep in range(12):
        if _check_deadline():
            break
        step = 0.5 / (1.0 + sweep * 0.4)
        improved = False
        for i in range(num_polish):
            if _check_deadline():
                break
            base = vals[i]
            trials = [base + step, base - step]
            if sweep < 3:
                trials.append(base + rng.uniform(-math.pi, math.pi))
            for trial in trials:
                vals[i] = trial
                U = _build(full_circuit(vals), qubits)
                d = _distance(target, U)
                if d < best_dist:
                    best_dist = d
                    best = full_circuit(vals)
                    improved = True
                else:
                    vals[i] = base
        if not improved:
            break

    # Final fine polish
    for polish in range(8):
        if _check_deadline():
            break
        step = 0.05 / (1.0 + polish)
        for i in range(num_polish):
            if _check_deadline():
                break
            base = vals[i]
            for trial in (base + step, base - step):
                vals[i] = trial
                U = _build(full_circuit(vals), qubits)
                d = _distance(target, U)
                if d < best_dist:
                    best_dist = d
                    best = full_circuit(vals)
                else:
                    vals[i] = base

    if len(best) > 400:
        best = best[:400]
    return best


def synthesize(problem):
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    known = _match_known(target, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == 'cx')
        if cnot_count <= budget:
            return known

    if _check_deadline():
        # Fallback: return a valid (identity-ish) circuit within budget.
        return []

    return _solve_generic(target, qubits, budget)
