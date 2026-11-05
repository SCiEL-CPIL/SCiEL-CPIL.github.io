# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 140.0  # safety margin below 150s kill
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

def _expected_cut(n, edges, gamma, beta):
    """Exact expected cut for p=2 QAOA via state-vector simulation."""
    N = 1 << n
    # initial |+>^n
    psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)
    # precompute cut values for all masks
    cut_vals = np.zeros(N, dtype=np.float64)
    for a, b in edges:
        ab = (1 << a) | (1 << b)
        for m in range(N):
            if (m & ab) != 0 and (m & ab) != ab:
                cut_vals[m] += 1.0
    for g, b in zip(gamma, beta):
        # cost phase
        psi *= np.exp(-1j * g * cut_vals)
        # mixer: exp(-i b sum X) = product of single-qubit rotations
        for q in range(n):
            bit = 1 << q
            c = np.cos(b)
            s = np.sin(b)
            new = np.empty_like(psi)
            for m in range(N):
                m2 = m ^ bit
                new[m] = c * psi[m] - 1j * s * psi[m2]
            psi = new
    prob = np.abs(psi) ** 2
    return float(np.dot(prob, cut_vals))

def _exact_optimum(n, edges):
    """Exact MaxCut via brute force over all 2^n assignments."""
    best = 0
    for m in range(1 << n):
        c = 0
        for a, b in edges:
            if ((m >> a) ^ (m >> b)) & 1:
                c += 1
        if c > best:
            best = c
    return float(best)

def _optimize(n, edges, opt):
    def neg_obj(x):
        g1, b1, g2, b2 = x
        return -_expected_cut(n, edges, (g1, g2), (b1, b2))
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    bounds = [(0.0, np.pi)] * 4
    best_x, best_v = x0, neg_obj(x0)
    rng = np.random.default_rng(713)
    starts = [x0]
    for _ in range(11):
        starts.append(rng.uniform(0.0, np.pi, size=4))
    for s in starts:
        if _remaining() < 0.6:
            break
        res = minimize(neg_obj, s, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "maxls": 40})
        if res.fun < best_v:
            best_v, best_x = res.fun, res.x
    return best_x

def angles(n, edges, p):
    edges = [(int(a), int(b)) for a, b in edges]
    if not edges:
        return np.zeros((p, 2))
    # exact optimum (cheap for n<=10)
    opt = _exact_optimum(n, edges)
    # optimize 4 angles
    x = _optimize(n, edges, opt)
    g1, b1, g2, b2 = x
    return np.array([[g1, b1], [g2, b2]])
# EVOLVE-BLOCK-END
