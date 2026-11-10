# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import minimize

_DEADLINE = 630.0
_START = time.time()


def _remaining():
    return _DEADLINE - 30.0 - (time.time() - _START)


def _build_pooling(n, r):
    """Return a sparse matrix A (k x n^2) such that A @ p (p = P.ravel()) = pooled probs.
    Pooling: value = x + r*y for cell (y, x). P is indexed P[y, x].
    """
    rows = []
    cols = []
    # map each cell to a bin index
    bin_of_cell = {}
    for y in range(n):
        for x in range(n):
            v = x + r * y
            if v not in bin_of_cell:
                bin_of_cell[v] = len(bin_of_cell)
            rows.append(bin_of_cell[v])
            cols.append(y * n + x)
    k = len(bin_of_cell)
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(k, n * n))
    return A.tocsr()


def _entropy_and_grad(pooled):
    """Shannon entropy (bits) of a probability vector, ignoring zeros.
    Returns (H, grad) where grad = dH/dpooled.
    """
    pooled = np.asarray(pooled, dtype=np.float64)
    mask = pooled > 0.0
    pm = pooled[mask]
    # H = -sum p log2 p ; dH/dp = -1 - log2(p)  (for p>0), else 0
    g = np.zeros_like(pooled)
    g[mask] = -1.0 - np.log2(pm)
    H = float(-(pm * np.log2(pm)).sum())
    return H, g


def _objective_and_grad(theta, As, n):
    """theta: logit vector of length n^2. Returns (-ratio, grad) for minimization.
    As: list of 4 sparse matrices [A_{-1}, A_0, A_1, A_2].
    """
    # softmax over logits (stable)
    t = theta - theta.max()
    e = np.exp(t)
    p = e / e.sum()  # length n^2

    D_H, D_g = _entropy_and_grad(As[0] @ p)
    H0, g0 = _entropy_and_grad(As[1] @ p)
    H1, g1 = _entropy_and_grad(As[2] @ p)
    H2, g2 = _entropy_and_grad(As[3] @ p)

    # M = max(H0, H1, H2); pick the argmax for gradient
    Hs = np.array([H0, H1, H2])
    gs = [g0, g1, g2]
    idx = int(np.argmax(Hs))
    M = Hs[idx]

    if M <= 1e-9:
        return (1e18, np.zeros_like(theta))

    ratio = D_H / M

    # grad of ratio wrt p:
    # d(ratio)/dp = (dD/dp)/M - D*(dM/dp)/M^2
    dD_dp = As[0].T @ D_g
    dM_dp = As[idx].T @ gs[idx]
    dp_ratio = dD_dp / M - D_H * dM_dp / (M * M)

    # chain rule: dp/dtheta = diag(p) - outer(p, p)
    # grad_theta = (dp_ratio * p) - p * sum(dp_ratio * p)
    s = float((dp_ratio * p).sum())
    grad_theta = dp_ratio * p - p * s

    return (-ratio, -grad_theta)


def _run():
    n = 10
    As = [_build_pooling(n, r) for r in (-1.0, 0.0, 1.0, 2.0)]
    N = n * n

    best_P = np.ones(N) / N
    best_val = -np.inf

    def finalize(p):
        nonlocal best_P, best_val
        p = np.clip(p, 0.0, None)
        s = p.sum()
        if s <= 1e-9:
            return
        p = p / s
        val = _value_of(p, As)
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_P = p.copy()

    def _value_of(p, As):
        D_H, _ = _entropy_and_grad(As[0] @ p)
        Hs = [_entropy_and_grad(A @ p)[0] for A in As[1:]]
        M = max(Hs)
        if M <= 1e-9:
            return -np.inf
        return D_H / M

    # baseline uniform
    finalize(best_P)

    rng = np.random.default_rng(12345)

    # restarts: uniform + random dirichlet + random logits
    n_restarts = 60
    for it in range(n_restarts):
        if _remaining() < 5.0:
            break

        # choose starting point
        if it == 0:
            theta = np.zeros(N)
        elif it % 3 == 1:
            # dirichlet-ish
            w = rng.dirichlet(np.ones(N))
            theta = np.log(w + 1e-12)
        else:
            # random logits, small scale to keep softmax not too peaked initially
            theta = rng.normal(0.0, 1.0, size=N)

        try:
            res = minimize(
                _objective_and_grad,
                theta,
                args=(As, n),
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 400, "maxfun": 800, "ftol": 1e-12, "gtol": 1e-8},
            )
        except Exception:
            continue

        # recover p from theta
        t = res.x - res.x.max()
        e = np.exp(t)
        p = e / e.sum()
        finalize(p)

    return best_P.reshape(n, n)


def arithmetic_kakeya_distribution() -> np.ndarray:
    P = _run()
    # final safety clip + normalize
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s <= 1e-9:
        P = np.ones((10, 10)) / 100.0
    else:
        P = P / s
    return P


# EVOLVE-BLOCK-END
