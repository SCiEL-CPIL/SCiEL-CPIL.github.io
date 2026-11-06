"""Optimized product-formula coefficients for Heisenberg-chain time evolution."""
import time
import numpy as np

_DEADLINE = 140.0  # safety margin before the 150 s wall-clock kill


def _build_hamiltonians():
    """Six-site open XXZ Heisenberg chain, anisotropy 0.8, field 0.3.
    A = odd bonds + half field, B = even bonds + half field."""
    N = 6
    aniso = 0.8
    field = 0.3
    # spin-1/2 operators
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    I2 = np.eye(2)

    def op_on(site, op):
        mats = [I2] * N
        mats[site] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def bond(i, j):
        # XXZ bond: Sx_i Sx_j + Sy_i Sy_j + aniso Sz_i Sz_j
        return (np.kron(op_on(i, Sx), op_on(j, I2)) @ np.kron(op_on(i, I2), op_on(j, Sx))
                + np.kron(op_on(i, Sy), op_on(j, I2)) @ np.kron(op_on(i, I2), op_on(j, Sy))
                + aniso * np.kron(op_on(i, Sz), op_on(j, I2)) @ np.kron(op_on(i, I2), op_on(j, Sz)))

    # sum of Sx over all sites (field term)
    Sx_total = sum(op_on(s, Sx) for s in range(N))

    A = 1j * (bond(0, 1) + bond(2, 3) + bond(4, 5)) + 1j * (field / 2.0) * Sx_total
    B = 1j * (bond(1, 2) + bond(3, 4)) + 1j * (field / 2.0) * Sx_total
    return A, B


def _mat_exp(M):
    """Matrix exponential via eigendecomposition (M is Hermitian/normal)."""
    w, V = np.linalg.eig(M)
    return (V @ np.diag(np.exp(w))) @ np.linalg.inv(V)


def _precompute(A, B, a, b, t):
    """Return list of (expB, expA) factors for the 12 stages at time t."""
    expB = [_mat_exp(-1j * bk * t * B) for bk in b]
    expA = [_mat_exp(-1j * ak * t * A) for ak in a]
    return expB, expA


def _product_formula(expB, expA):
    U = np.eye(len(expB[0]))
    for k in range(len(expB)):
        U = expB[k] @ U
        U = expA[k] @ U
    return U


def _spectral_norm(M):
    # spectral norm = largest singular value
    try:
        return float(np.linalg.norm(M, 2))
    except np.linalg.LinAlgError:
        # fallback via eig of M^H M
        w = np.linalg.eigvalsh(M.conj().T @ M)
        return float(np.sqrt(max(0.0, w.max())))


def _objective(a12, b12, A, B, expAB_cache):
    """Mean spectral-norm error over the three evaluation times.
    a12, b12 are the 22 free coefficients (first 11 of each); the 12th is 1 - sum."""
    a = list(a12) + [1.0 - sum(a12)]
    b = list(b12) + [1.0 - sum(b12)]
    # consistency already enforced by construction
    total = 0.0
    for t in (0.25, 0.5, 1.0):
        expB, expA = _precompute(A, B, a, b, t)
        U = _product_formula(expB, expA)
        total += _spectral_norm(U - expAB_cache[t])
    return total / 3.0


def _nelder_mead(f, x0, max_iter=400, tol=1e-10, deadline=_DEADLINE):
    """Manual Nelder-Mead with a deadline guard. Returns best x, best f."""
    n = len(x0)
    # simplex
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        step = 0.05 if x0[i] != 0.0 else 0.05
        pt[i] += step
        simplex.append(pt)
    fvals = [f(np.array(s)) for s in simplex]
    best_x = simplex[int(np.argmin(fvals))]
    best_f = min(fvals)

    for it in range(max_iter):
        if time.time() > deadline:
            break
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if it % 50 == 0:
            best_x = simplex[0]
            best_f = fvals[0]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        centroid = [sum(simplex[i][j] for i in range(n - 1)) / (n - 1) for j in range(n)]
        # reflection
        xr = [centroid[j] + 1.0 * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(np.array(xr))
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < fvals[0]:
            # expansion
            xe = [centroid[j] + 2.0 * (xr[j] - centroid[j]) for j in range(n)]
            fe = f(np.array(xe))
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        else:
            # contraction
            xc = [centroid[j] + 0.5 * (simplex[-1][j] - centroid[j]) for j in range(n)]
            fc = f(np.array(xc))
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    fvals[i] = f(np.array(simplex[i]))
        # track best
        m = int(np.argmin(fvals))
        if fvals[m] < best_f:
            best_f = fvals[m]
            best_x = simplex[m]
    return np.array(best_x), best_f


def coefficients(stages):
    """Optimized 12-stage product-formula coefficients.
    Returns (a, b): two lists of `stages` finite numbers in [-50,50], sum(a)=sum(b)=1."""
    stages = int(stages)
    A, B = _build_hamiltonians()
    # exact propagator exp(-i t (A+B)) for the three times
    AB = A + B
    expAB_cache = {t: _mat_exp(-1j * t * AB) for t in (0.25, 0.5, 1.0)}

    # start from a reasonable Strang-like split: 6 substeps of (a=1/12, b=1/12) pattern
    # a: 12 entries, b: 12 entries; use symmetric pattern
    a0 = [1.0 / 12.0] * 12
    b0 = [1.0 / 12.0] * 12
    x0 = np.array(a0[:11] + b0[:11])

    deadline = _DEADLINE
    best_x, best_f = _nelder_mead(lambda x: _objective(x[:11], x[11:], A, B, expAB_cache),
                                  x0, max_iter=600, deadline=deadline)

    a = list(best_x[:11]) + [1.0 - sum(best_x[:11])]
    b = list(best_x[11:]) + [1.0 - sum(best_x[11:])]

    # pad / trim to `stages`
    def fix(v):
        v = list(v)
        while len(v) < stages:
            v.append(0.0)
        v = v[:stages]
        # clamp to [-50, 50]
        v = [max(-50.0, min(50.0, float(x))) for x in v]
        return v

    a = fix(a)
    b = fix(b)
    return a, b
