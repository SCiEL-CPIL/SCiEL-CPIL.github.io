"""Product-formula coefficient optimizer for 6-site XXZ Heisenberg chain."""
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 128.0


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build_hamiltonians():
    """Return (A, B): odd-bond+half-field and even-bond+half-field for 6-site open XXZ."""
    N = 6
    delta = 0.8
    field = 0.3

    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    def kron_all(ops):
        r = np.array([[1.0 + 0j]])
        for op in ops:
            r = np.kron(r, op)
        return r

    sx_ops = [sx] + [I2] * (N - 1)
    sy_ops = [sy] + [I2] * (N - 1)
    sz_ops = [sz] + [I2] * (N - 1)
    Sx_list, Sy_list, Sz_list = [], [], []
    for i in range(N):
        Sx_list.append(kron_all(sx_ops[:i] + [sx] + sx_ops[i + 1:]))
        Sy_list.append(kron_all(sy_ops[:i] + [sy] + sy_ops[i + 1:]))
        Sz_list.append(kron_all(sz_ops[:i] + [sz] + sz_ops[i + 1:]))

    A = np.zeros((2 ** N, 2 ** N), dtype=complex)
    B = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        J = Sx_list[i] @ Sx_list[i + 1] + Sy_list[i] @ Sy_list[i + 1] + delta * Sz_list[i] @ Sz_list[i + 1]
        if i % 2 == 0:
            A += J
        else:
            B += J
    for i in range(N):
        h = field * Sz_list[i] * 0.5
        if i % 2 == 0:
            A += h
        else:
            B += h
    return A, B


def _norm_err(a, b, A, B, Aexp, Bexp, tvals, cache):
    err_sum = 0.0
    for t in tvals:
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            bk, ak = b[k], a[k]
            if abs(bk) > 1e-15:
                key = round(bk * t, 12)
                if key not in cache:
                    cache[key] = Bexp[key] @ Bexp[key] if False else None
                U = U @ Bexp_cache(bk * t, B, cache)
            if abs(ak) > 1e-15:
                U = U @ Aexp_cache(ak * t, A, cache)
        err_sum += np.linalg.norm(U - Aexp_cache(t, A + B, cache))
    return err_sum / len(tvals)


def _expm(M):
    # Use scipy if available, else numpy's built-in via eigendecomp fallback
    try:
        from scipy.linalg import expm
        return expm(M)
    except Exception:
        w, V = np.linalg.eig(M)
        return (V @ np.diag(np.exp(w)) @ np.linalg.inv(V))


_exp_cache = {}


def _expm_cached(M, key):
    if key not in _exp_cache:
        _exp_cache[key] = _expm(M)
    return _exp_cache[key]


def _Aexp_cache(s, A, cache):
    key = ("A", round(s, 12))
    if key not in cache:
        cache[key] = _expm_cached(-1j * s * A, key)
    return cache[key]


def _Bexp_cache(s, B, cache):
    key = ("B", round(s, 12))
    if key not in cache:
        cache[key] = _expm_cached(-1j * s * B, key)
    return cache[key]


def _eval(a, b, A, B, Aexp, Bexp, tvals, cache):
    err_sum = 0.0
    for t in tvals:
        U = np.eye(A.shape[0], dtype=complex)
        for k in range(len(a)):
            bk = b[k]
            if abs(bk) > 1e-15:
                U = U @ _Bexp_cache(bk * t, B, cache)
            ak = a[k]
            if abs(ak) > 1e-15:
                U = U @ _Aexp_cache(ak * t, A, cache)
        err_sum += np.linalg.norm(U - _Aexp_cache(t, A + B, cache))
    return err_sum / len(tvals)


def coefficients(stages):
    global _T0
    stages = int(stages)
    if stages < 2:
        stages = 2
    tvals = [0.25, 0.5, 1.0]

    A, B = _build_hamiltonians()
    Aexp = _expm_cached(-1j * (A + B), ("AB", 1.0))  # placeholder; real exact per t below
    Bexp = None

    cache = {}
    # exact propagator per t
    exact = {t: _expm_cached(-1j * t * (A + B), ("AB", round(t, 12))) for t in tvals}

    def score(a, b):
        err_sum = 0.0
        for t in tvals:
            U = np.eye(A.shape[0], dtype=complex)
            for k in range(len(a)):
                bk = b[k]
                if abs(bk) > 1e-15:
                    U = U @ _Bexp_cache(bk * t, B, cache)
                ak = a[k]
                if abs(ak) > 1e-15:
                    U = U @ _Aexp_cache(ak * t, A, cache)
            err_sum += np.linalg.norm(U - exact[t])
        return err_sum / len(tvals)

    # Initial guess: symmetric Strang-like split with 12 stages
    # Pattern: a,b alternating with endpoints zeroed for symmetry
    a = [0.0] * stages
    b = [0.0] * stages
    # Use 6 substeps of Strang: each substep a=1/12, b=1/12, plus symmetric padding
    # Simpler: uniform a=1/12, b=1/12 for all 12 (product of exp(-ib t B)exp(-ia t A))
    a = [1.0 / stages] * stages
    b = [1.0 / stages] * stages

    best_a, best_b, best_s = a[:], b[:], score(a, b)

    # Local search: coordinate descent + small random perturbations
    rng = np.random.default_rng(42)
    iters = 0
    while _remaining() > 15.0:
        iters += 1
        # pick a random coefficient to perturb
        which = rng.integers(0, 2)  # 0=a, 1=b
        idx = rng.integers(0, stages)
        cur_list = best_a if which == 0 else best_b
        other_list = best_b if which == 0 else best_a

        # try a few step sizes
        improved = False
        for step in [0.01, 0.005, 0.002, -0.01, -0.005, -0.002]:
            trial = cur_list[:]
            trial[idx] += step
            # enforce sum constraint by adjusting another index
            s_diff = 1.0 - sum(trial)
            if abs(s_diff) > 1e-15:
                # spread correction over a few indices
                for j in range(stages):
                    trial[j] += s_diff / stages
            # clamp to [-50,50]
            trial = [min(50.0, max(-50.0, x)) for x in trial]
            # re-normalize if clamping broke sum
            s2 = sum(trial)
            if abs(s2 - 1.0) > 1e-12:
                for j in range(stages):
                    trial[j] += (1.0 - s2) / stages

            if which == 0:
                s = score(trial, other_list)
                if s < best_s - 1e-10:
                    best_a, best_s = trial[:], s
                    improved = True
            else:
                s = score(other_list, trial)
                if s < best_s - 1e-10:
                    best_b, best_s = trial[:], s
                    improved = True
        if not improved:
            # try a small random move across a few coordinates
            n_move = int(rng.integers(1, 4))
            idxs = rng.choice(stages, size=n_move, replace=False)
            for _ in range(3):
                trial = (best_a if which == 0 else best_b)[:].copy()
                for idx in idxs:
                    trial[idx] += float(rng.normal(0, 0.005))
                s_diff = 1.0 - sum(trial)
                for j in range(stages):
                    trial[j] += s_diff / stages
                trial = [min(50.0, max(-50.0, x)) for x in trial]
                s2 = sum(trial)
                if abs(s2 - 1.0) > 1e-12:
                    for j in range(stages):
                        trial[j] += (1.0 - s2) / stages
                if which == 0:
                    s = score(trial, other_list)
                else:
                    s = score(other_list, trial)
                if s < best_s - 1e-10:
                    if which == 0:
                        best_a, best_s = trial[:], s
                    else:
                        best_b, best_s = trial[:], s
                    break

    # Final cleanup: ensure exact sums
    for lst in (best_a, best_b):
        s = sum(lst)
        if abs(s - 1.0) > 1e-12:
            for j in range(stages):
                lst[j] += (1.0 - s) / stages
        for j in range(stages):
            lst[j] = min(50.0, max(-50.0, lst[j]))
        # re-fix sum after clamp
        s = sum(lst)
        if abs(s - 1.0) > 1e-12:
            for j in range(stages):
                lst[j] += (1.0 - s) / stages

    return best_a, best_b
