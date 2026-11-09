import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 135.0


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    N = 6
    delta = 0.8
    field = 0.3
    dim = 2 ** N

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_all(ops):
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def bond_op(i, j):
        ops_x = [I2] * N
        ops_y = [I2] * N
        ops_z = [I2] * N
        ops_x[i] = sx
        ops_x[j] = sx
        ops_y[i] = sy
        ops_y[j] = sy
        ops_z[i] = sz
        ops_z[j] = sz
        return (kron_all(ops_x) + kron_all(ops_y) + delta * kron_all(ops_z))

    B = bond_op(0, 1) + bond_op(2, 3) + bond_op(4, 5)
    A = bond_op(1, 2) + bond_op(3, 4)

    field_op = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        ops = [I2] * N
        ops[i] = sz
        field_op += kron_all(ops)

    A = A + (field / 2.0) * field_op
    B = B + (field / 2.0) * field_op

    return A, B


def _compute_error(a, b, A, B, times, dim, U_exacts):
    n = len(a)
    errors = []
    for idx, t in enumerate(times):
        U = np.eye(dim, dtype=complex)
        for k in range(n):
            Pk = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A)
            U = U @ Pk
        U_exact = U_exacts[idx]
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return float(np.mean(errors))


def coefficients(stages):
    if stages != 12:
        substeps = max(1, stages // 2)
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        while len(a) > stages:
            a.pop()
            b.pop()
        return a, b

    dim = 64
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]

    AB = A + B
    U_exacts = [expm(-1j * t * AB) for t in times]

    def objective(x):
        if _remaining() < 5:
            return 1e10
        a = list(x[:11])
        a.append(1.0 - sum(a))
        b = list(x[11:22])
        b.append(1.0 - sum(b))

        a = [max(-50.0, min(50.0, float(ai))) for ai in a]
        b = [max(-50.0, min(50.0, float(bi))) for bi in b]
        sa = sum(a)
        sb = sum(b)
        if abs(sa) < 1e-15:
            a = [1.0 / 12] * 12
        else:
            a = [ai / sa for ai in a]
        if abs(sb) < 1e-15:
            b = [1.0 / 12] * 12
        else:
            b = [bi / sb for bi in b]

        try:
            return _compute_error(a, b, A, B, times, dim, U_exacts)
        except Exception:
            return 1e10

    uniform = np.array([1.0 / 12] * 22)

    np.random.seed(12345)
    starts = [uniform]
    for _ in range(6):
        noise = np.random.uniform(-0.04, 0.04, 22)
        starts.append(uniform + noise)

    best_x = None
    best_val = np.inf

    for x0 in starts:
        if _remaining() < 15:
            break
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 150, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass

    if best_x is None:
        return [1.0 / 12] * 12, [1.0 / 12] * 12

    a = list(best_x[:11])
    a.append(1.0 - sum(a))
    b = list(best_x[11:22])
    b.append(1.0 - sum(b))

    a = [max(-50.0, min(50.0, float(ai))) if np.isfinite(ai) else 0.0 for ai in a]
    b = [max(-50.0, min(50.0, float(bi))) if np.isfinite(bi) else 0.0 for bi in b]

    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [ai / sa for ai in a]
    else:
        a = [1.0 / 12] * 12
    if abs(sb) > 1e-15:
        b = [bi / sb for bi in b]
    else:
        b = [1.0 / 12] * 12

    return a, b
