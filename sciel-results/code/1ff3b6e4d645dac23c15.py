"""Optimized 12-stage product formula for six-site open XXZ Heisenberg chain."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

_DEADLINE = 128.0
_T0 = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    n = 6
    delta = 0.8
    h = 0.3
    s = 1.0 / 2.0
    sx = np.array([[0.0, s], [s, 0.0]])
    sz = np.array([[s, 0.0], [0.0, -s]])
    sxp = np.array([[0.0, s], [0.0, 0.0]])
    sxm = np.array([[0.0, 0.0], [s, 0.0]])

    def kron_all(ops):
        r = np.array([[1.0]])
        for op in ops:
            r = np.kron(r, op)
        return r

    def site_op(op):
        ops = [np.eye(2)] * n
        ops[0] = op
        return kron_all(ops)

    def bond_op(op1, op2, i):
        ops = [np.eye(2)] * n
        ops[i] = op1
        ops[i + 1] = op2
        return kron_all(ops)

    A = np.zeros((2 ** n, 2 ** n), dtype=complex)
    B = np.zeros((2 ** n, 2 ** n), dtype=complex)
    # Odd bonds (0-indexed 1,3) go to A; even bonds (0,2,4) go to B
    for i in range(n - 1):
        bond = (delta / 2.0) * (bond_op(sxp, sxm, i) + bond_op(sxm, sxp, i)) + bond_op(sz, sz, i)
        if i % 2 == 1:
            A = A + bond
        else:
            B = B + bond
    # Half of uniform field on each
    for i in range(n):
        hf = 0.5 * h * site_op(sz)
        A = A + hf
        B = B + hf
    return A, B


def _symmetric_12():
    # Symmetric 12-stage composition: a_1..a_6, b_1..b_6, then reversed.
    # This is a valid 12-stage formula with sum(a)=sum(b)=1 automatically.
    a = [0.0] * 12
    b = [0.0] * 12
    for k in range(6):
        a[k] = 1.0 / 6.0
        a[11 - k] = 1.0 / 6.0
        b[k] = 1.0 / 6.0
        b[11 - k] = 1.0 / 6.0
    return a, b


def coefficients(stages):
    if stages != 12:
        # Fallback for unexpected stage counts (shouldn't happen).
        a = [1.0 / stages] * stages
        b = [1.0 / stages] * stages
        return a, b

    A, B = _build_hamiltonians()
    t_vals = (0.25, 0.5, 1.0)

    def _norm_err(U):
        return np.linalg.norm(U, 2)

    def _objective(x):
        a = x[:12]
        b = x[12:24]
        total = 0.0
        for t in t_vals:
            U = np.eye(2 ** 6, dtype=complex)
            for k in range(12):
                U = expm(-1j * b[k] * t * B) @ U
                U = expm(-1j * a[k] * t * A) @ U
            Ue = expm(-1j * t * (A + B))
            total += _norm_err(U - Ue)
        return total / 3.0

    def _jac(x):
        # Finite-difference gradient (cheap enough for 24 vars, few iters).
        g = np.zeros(24)
        f0 = _objective(x)
        for i in range(24):
            xp = x.copy()
            xp[i] += 1e-6
            g[i] = (_objective(xp) - f0) / 1e-6
        return g

    def _constraints(x):
        return [np.sum(x[:12]) - 1.0, np.sum(x[12:24]) - 1.0]

    def _constr_jac(x):
        J = np.zeros((2, 24))
        J[0, :12] = 1.0
        J[1, 12:] = 1.0
        return J

    cons = ({'type': 'eq', 'fun': _constraints, 'jac': _constr_jac},)

    # Start from the symmetric composition.
    a0, b0 = _symmetric_12()
    x0 = np.array(a0 + b0, dtype=float)

    best_x = x0.copy()
    best_f = _objective(x0)

    # A few restarts with small perturbations to escape local minima,
    # each guarded by the deadline.
    rng = np.random.default_rng(12345)
    n_restarts = 4
    for r in range(n_restarts):
        if _remaining() < 25.0:
            break
        if r == 0:
            start = x0.copy()
        else:
            start = x0 + rng.normal(0.0, 0.05, 24)
        try:
            res = minimize(
                _objective, start, jac=_jac, method='SLSQP',
                constraints=cons, bounds=[(-50.0, 50.0)] * 24,
                options={'maxiter': 40, 'ftol': 1e-9, 'disp': False},
            )
            if res.x is not None:
                # Enforce bounds strictly.
                rx = np.clip(res.x, -50.0, 50.0)
                rf = _objective(rx)
                if np.isfinite(rf) and rf < best_f:
                    best_f = rf
                    best_x = rx.copy()
        except Exception:
            pass
        if _remaining() < 25.0:
            break

    a = np.clip(best_x[:12], -50.0, 50.0)
    b = np.clip(best_x[12:24], -50.0, 50.0)
    # Re-normalize to satisfy sum constraints exactly within tolerance.
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-12:
        a = a / sa
    else:
        a = np.ones(12) / 12.0
    if abs(sb) > 1e-12:
        b = b / sb
    else:
        b = np.ones(12) / 12.0
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    return list(a), list(b)
