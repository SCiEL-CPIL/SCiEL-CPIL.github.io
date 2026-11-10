# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_MARGIN = 5.0  # seconds before the 630s kill


def _build_data():
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    cw = np.zeros((D, n), dtype=float)
    ccw = np.zeros((D, n), dtype=float)
    for i, (s, t, w) in enumerate(demands):
        # clockwise from s to t uses edges s, s+1, ..., t-1 (mod n)
        # if s == t, uses all edges
        if s == t:
            edges = list(range(n))
        else:
            edges = [(s + k) % n for k in range((t - s) % n)]
        for e in edges:
            cw[i, e] = w
        # counter-clockwise uses all other edges
        for e in range(n):
            ccw[i, e] = w - cw[i, e]
    return demands, cw, ccw


def _max_load(cw, ccw, routing):
    # routing: array of 0/1 (0=CW, 1=CCW)
    loads = np.zeros(cw.shape[1], dtype=float)
    for i in range(cw.shape[0]):
        if routing[i] >= 0.5:
            loads += ccw[i]
        else:
            loads += cw[i]
    return loads


def _smooth_max(loads, beta):
    # log-sum-exp based smooth max
    m = loads.max()
    if m == 0:
        return 0.0
    s = np.exp(beta * (loads - m))
    return m + np.log(s.sum()) / beta


def _smooth_grad(loads, beta):
    # gradient of smooth max w.r.t. loads
    m = loads.max()
    s = np.exp(beta * (loads - m))
    p = s / s.sum()
    return p


def _local_search(cw, ccw, routing, max_iters=200000):
    D, n = cw.shape
    routing = routing.copy()
    loads = np.zeros(n, dtype=float)
    for i in range(D):
        if routing[i] >= 0.5:
            loads += ccw[i]
        else:
            loads += cw[i]
    best_max = loads.max()
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(D):
            # try flipping i
            if routing[i] >= 0.5:
                # currently CCW, try CW
                new_loads = loads - ccw[i] + cw[i]
            else:
                # currently CW, try CCW
                new_loads = loads - cw[i] + ccw[i]
            nm = new_loads.max()
            if nm < best_max - 1e-12:
                # accept
                if routing[i] >= 0.5:
                    loads = new_loads
                    routing[i] = 0
                else:
                    loads = new_loads
                    routing[i] = 1
                best_max = nm
                improved = True
                iters = 0  # reset to allow more passes
    return routing, best_max


def _gradient_descent(cw, ccw, D, n, beta, lr, max_iter):
    # x in [0,1], x=0 means CW, x=1 means CCW
    # load = sum_i ( (1-x_i)*cw_i + x_i*ccw_i ) = base + sum_i x_i*(ccw_i - cw_i)
    base = np.zeros(n, dtype=float)
    diff = np.zeros((D, n), dtype=float)
    for i in range(D):
        base += cw[i]
        diff[i] = ccw[i] - cw[i]

    x = np.full(D, 0.5)
    best_x = x.copy()
    best_val = np.inf
    for it in range(max_iter):
        loads = base + diff.T @ x  # shape (n,)
        val = _smooth_max(loads, beta)
        if val < best_val - 1e-10:
            best_val = val
            best_x = x.copy()
        # gradient of smooth_max w.r.t. loads
        p = _smooth_grad(loads, beta)  # shape (n,)
        # d loads / d x_i = diff[i]
        # grad_x_i = dot(p, diff[i])
        grad = diff @ p  # shape (D,)
        x -= lr * grad
        x = np.clip(x, 0, 1)
    return best_x


def ring_routing() -> np.ndarray:
    start = time.time()
    demands, cw, ccw = _build_data()
    D, n = cw.shape

    # Try multiple beta values for smooth surrogate
    best_routing = None
    best_max = np.inf

    for beta in [5.0, 10.0, 20.0, 40.0, 80.0]:
        if time.time() - start > 630 - _DEADLINE_MARGIN:
            break
        # gradient descent
        max_iter = 3000
        lr = 0.05
        x = _gradient_descent(cw, ccw, D, n, beta, lr, max_iter)
        # round
        routing = (x >= 0.5).astype(float)
        loads = _max_load(cw, ccw, routing)
        cur_max = loads.max()
        if cur_max < best_max:
            best_max = cur_max
            best_routing = routing.copy()

    # Also try all-CW and all-CCW as baselines
    for init in [np.zeros(D), np.ones(D)]:
        routing = (init >= 0.5).astype(float)
        loads = _max_load(cw, ccw, routing)
        cur_max = loads.max()
        if cur_max < best_max:
            best_max = cur_max
            best_routing = routing.copy()

    # Local search polish on best
    if best_routing is not None:
        if time.time() - start > 630 - _DEADLINE_MARGIN:
            return best_routing
        routing, cur_max = _local_search(cw, ccw, best_routing)
        if cur_max < best_max:
            best_max = cur_max
            best_routing = routing

    # Try local search from a few random starts if time permits
    rng = np.random.default_rng(42)
    for trial in range(20):
        if time.time() - start > 630 - _DEADLINE_MARGIN:
            break
        routing = (rng.random(D) >= 0.5).astype(float)
        loads = _max_load(cw, ccw, routing)
        cur_max = loads.max()
        if cur_max < best_max:
            best_max = cur_max
            best_routing = routing.copy()
        # local search from this start
        if time.time() - start > 630 - _DEADLINE_MARGIN:
            break
        routing2, cur_max2 = _local_search(cw, ccw, routing, max_iters=50000)
        if cur_max2 < best_max:
            best_max = cur_max2
            best_routing = routing2

    # Final local search on the best
    if best_routing is not None:
        if time.time() - start > 630 - _DEADLINE_MARGIN:
            return best_routing
        routing, cur_max = _local_search(cw, ccw, best_routing, max_iters=500000)
        if cur_max < best_max:
            best_max = cur_max
            best_routing = routing

    return best_routing


# EVOLVE-BLOCK-END
