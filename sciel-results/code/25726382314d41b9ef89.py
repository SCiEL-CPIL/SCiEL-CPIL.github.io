# EVOLVE-BLOCK-START
import time
import numpy as np


def _keich(N):
    """Keich binary construction for N = 2^k base positions (1D)."""
    k = int(round(np.log2(N)))
    j_terms = np.arange(1, k + 1)
    out = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(bin(idx)[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        out[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / k)
    return out


def _grid1d(p, N):
    """Interval endpoints (left, right) for each of N sheared 1D intervals.

    Interval i at height c: [p_i + c*i/N, p_i + (i+1)/N + c*i/N].
    left_i = p_i + c*i/N, right_i = p_i + (i+1)/N + c*i/N.
    """
    left = p + c * (np.arange(N) / N)
    right = left + (np.arange(1, N + 1) / N)
    return left, right


def _obj1d(p, N, quad):
    """Exact 1D objective: S(p) * I(p), where
    S = span(p) = max_i(p_i + (i+1)/N) - min_i p_i
    I = integral_0^1 (union length of sheared intervals) dc.
    Returns (value, grad)."""
    L = np.arange(1, N + 1) / N
    S = np.max(p + L) - np.min(p)
    dS = np.zeros(N)
    dS[np.argmax(p + L)] = 1.0
    dS[np.argmin(p)] = -1.0

    # Integrate union length over c via Gauss-Legendre.
    c_nodes, c_wts = np.polynomial.legendre.leggauss(quad)
    c_nodes = 0.5 * (c_nodes + 1.0)
    c_wts = 0.5 * c_wts
    I = 0.0
    dI = np.zeros(N)
    for c, w in zip(c_nodes, c_wts):
        left, right = _grid1d(p, N)
        order = np.argsort(left, kind="stable")
        ol = left[order]
        orr = right[order]
        # union length of intervals
        cov = 0.0
        cur_r = -np.inf
        for a, b in zip(ol, orr):
            if a > cur_r:
                cov += b - a
                cur_r = b
            else:
                if b > cur_r:
                    cov += b - cur_r
                    cur_r = b
        I += w * cov
        # gradient of union length wrt p (p_i shifts interval i by 1 in both
        # left and right, so its contribution to cov is 1 if i is the active
        # rightmost endpoint of its covered component, else 0).
        # Determine active right endpoints: walk the union, an interval is
        # "active" (extends the union) iff its right endpoint is the max right
        # among intervals overlapping the current covered region at that step.
        # Simpler: recompute by checking for each interval whether removing its
        # right endpoint reduces coverage. Use the sweep: an interval i is a
        # "max-right" interval if right_i >= right_j for all j whose interval
        # reaches the start of i's new contribution. We approximate exactly by
        # attributing each covered segment to the interval that set cur_r.
        # Redo sweep attributing contribution to the index that extends cur_r.
        contrib = np.zeros(N)
        cur_r = -np.inf
        for a, b, idx in zip(ol, orr, order):
            if a > cur_r:
                contrib[idx] += b - a
                cur_r = b
            else:
                if b > cur_r:
                    contrib[idx] += b - cur_r
                    cur_r = b
        dI += w * contrib
    return S * I, S * dI + I * dS


def _optimize_1d(p0, N, deadline):
    """Optimize a single 1D coordinate set with analytic-gradient L-BFGS-B."""
    from scipy.optimize import minimize
    N = int(N)
    p = np.array(p0, dtype=np.float64)

    def f_and_g(x):
        v, g = _obj1d(x, N, 40)
        return v, g

    def f(x):
        return f_and_g(x)[0]

    def fg(x):
        return f_and_g(x)

    best = p.copy()
    best_v = f(p)

    # Stage 1: coarse optimization
    res = minimize(fg, p, jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-12, "ftol": 1e-15,
                            "maxls": 40})
    if res.x is not None and np.all(np.isfinite(res.x)) and res.fun < best_v:
        best = res.x
        best_v = res.fun

    # Stage 2: high-precision polish with finer quadrature
    if time.time() < deadline - 20:
        def f_and_g2(x):
            v, g = _obj1d(x, N, 80)
            return v, g

        res2 = minimize(f_and_g2, best, jac=True, method="L-BFGS-B",
                        options={"maxiter": 1000, "gtol": 1e-14, "ftol": 1e-16,
                                 "maxls": 60})
        if res2.x is not None and np.all(np.isfinite(res2.x)) and res2.fun < best_v:
            best = res2.x
            best_v = res2.fun
    return best, best_v


def kakeya_needle_3d8() -> np.ndarray:
    """3D discrete Kakeya set (N=8). Returns (64,2) base positions (x,y).

    Row k = tube (i,j) with i=k//8, j=k%8. We exploit the product
    decomposition V = (X1-X0)(Y1-Y0) * int_0^1 A_x(c) A_y(c) dc and optimize
    the x and y coordinate sets independently with an exact analytic objective
    and gradient (L-BFGS-B), starting from the Keich construction.
    """
    N = 8
    deadline = time.time() + 600.0 - 30.0  # safety margin
    keich = _keich(N)

    # Optimize x and y independently.
    px, _ = _optimize_1d(keich, N, deadline)
    py, _ = _optimize_1d(keich, N, deadline)

    # Build (64,2) in raster order: i outer, j inner.
    i_idx = np.repeat(np.arange(N), N)
    j_idx = np.tile(np.arange(N), N)
    positions = np.stack([px[i_idx], py[j_idx]], axis=1)
    return positions


# EVOLVE-BLOCK-END
