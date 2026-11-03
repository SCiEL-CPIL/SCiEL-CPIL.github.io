"""Circle packing in unit square: pack 26 circles, disjoint interiors, maximize sum of radii."""
import time
import numpy as np

N = 26
_START = time.time()
_DEADLINE = _START + 130.0   # safety margin under the 150 s wall-clock limit


def _elapsed():
    return time.time() - _START


def _objective(z):
    """Return (value, grad) for maximizing sum(r) with smooth penalty on overlap/boundary."""
    z = np.asarray(z, dtype=float)
    n = N
    p = z[:2 * n].reshape(n, 2)
    r = z[2 * n:]

    # --- overlap penalty: soft-max of (ri+rj - dist)^2 over pairs ---
    iu = np.triu_indices(n, k=1)
    d = p[iu[0]] - p[iu[1]]
    dist = np.sqrt(np.sum(d * d, axis=1) + 1e-14)
    s = r[iu[0]] + r[iu[1]]
    gap = dist - s                      # >0 means separated
    # soft-max of (max(0,-gap))^2 via log-sum-exp on exp(lambda*gap)
    lam = 8.0
    g = -gap                            # want to push gap>0; large g => bad
    m = np.max(g)
    # logsumexp(g) = m + log(sum(exp(g-m)))
    lse = m + np.log(np.sum(np.exp(g - m)) + 1e-300)
    # penalty = (soft-max of (gap<0))^2 approx: use lse as a smooth "violated" measure
    # We want penalty large when any gap is very negative (large g). Use lse directly.
    overlap_pen = lse
    # gradient of lse w.r.t. g: softmax weights
    w = np.exp(g - lse)                 # sum to 1

    # --- boundary penalty: soft-max of (r - x)^2, (r - (1-x))^2, etc ---
    # For each circle: violations v = [r - x, r - (1-x), r - y, r - (1-y)]
    # We want all <= 0. Use soft-max of (max(0, v))^2.
    # Build matrix V of shape (n,4)
    V = np.empty((n, 4))
    V[:, 0] = r - p[:, 0]
    V[:, 1] = r - (1.0 - p[:, 0])
    V[:, 2] = r - p[:, 1]
    V[:, 3] = r - (1.0 - p[:, 1])
    # per-circle soft-max of V (large V => bad), squared
    lam2 = 8.0
    m2 = np.max(V, axis=1)
    lse2 = m2 + np.log(np.sum(np.exp(V - m2[:, None]), axis=1) + 1e-300)
    # penalty = sum of lse2 (each is a smooth max violation)
    boundary_pen = np.sum(lse2)
    w2 = np.exp(V - lse2[:, None])      # (n,4) weights

    # objective = sum(r) - mu*overlap_pen - mu*boundary_pen
    mu = 50.0
    value = np.sum(r) - mu * overlap_pen - mu * boundary_pen

    # --- gradients ---
    grad = np.empty_like(z)
    # d value/d r = 1 - mu * (d overlap_pen/dr + d boundary_pen/dr)
    # d overlap_pen/dr: d lse/d r_k = sum over pairs involving k of w * d g/d r_k
    # g_ij = - (dist_ij - r_i - r_j) = r_i + r_j - dist_ij ; d g_ij/d r_k = 1 if k in {i,j}
    # d lse/d g = w
    d_overlap_dr = np.zeros(n)
    # accumulate: for each pair (i,j), w contributes to d_overlap_dr[i] and [j]
    np.add.at(d_overlap_dr, iu[0], w)
    np.add.at(d_overlap_dr, iu[1], w)
    # d lse/d dist_ij = w * (-1)  (since g = r_i+r_j-dist, dg/ddist = -1)
    # d dist/d p
    d_overlap_dp = np.zeros((n, 2))
    # d dist_ij/d p_i = d_i/dist ; d dist_ij/d p_j = -d_i/dist
    inv = w / dist
    np.add.at(d_overlap_dp, iu[0], inv[:, None] * d)
    np.add.at(d_overlap_dp, iu[1], -inv[:, None] * d)

    # d boundary_pen/d r_k = sum over 4 sides of w2[k,s] * 1 = sum_s w2[k,s]
    d_boundary_dr = np.sum(w2, axis=1)
    # d boundary_pen/d x_k: side0 (r - x): w2[k,0]*(-1); side1 (r-(1-x)): w2[k,1]*(+1)
    d_boundary_dx = -w2[:, 0] + w2[:, 1]
    d_boundary_dy = -w2[:, 2] + w2[:, 3]

    grad[2 * n:] = 1.0 - mu * (d_overlap_dr + d_boundary_dr)
    grad[0::2] = -mu * (d_overlap_dp[:, 0] + d_boundary_dx)
    grad[1::2] = -mu * (d_overlap_dp[:, 1] + d_boundary_dy)

    return value, grad


def _is_valid(x, y, r, tol=1e-6):
    n = len(r)
    if np.any(r < -tol):
        return False
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if np.hypot(dx, dy) < r[i] + r[j] - tol:
                return False
    return True


def _make_initial(rng):
    """Jittered grid initial placement."""
    side = int(np.ceil(np.sqrt(N)))
    base_r = 0.5 / side
    x = np.empty(N)
    y = np.empty(N)
    r = np.empty(N)
    for idx in range(N):
        i = idx % side
        j = idx // side
        x[idx] = (i + 0.5) / side + (rng.random() - 0.5) * 0.4 * (1.0 / side)
        y[idx] = (j + 0.5) / side + (rng.random() - 0.5) * 0.4 * (1.0 / side)
        r[idx] = base_r * (0.9 + 0.1 * rng.random())
    # clip positions into a safe interior
    x = np.clip(x, 0.05, 0.95)
    y = np.clip(y, 0.05, 0.95)
    return x, y, r


def _repair(x, y, r):
    """Shrink circles until the configuration is valid. Returns (x,y,r, valid)."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    # keep positions inside with margin based on radius
    for _ in range(200):
        # ensure inside
        for k in range(N):
            r[k] = min(r[k], x[k], 1.0 - x[k], y[k], 1.0 - y[k], 0.5)
        if np.any(r < 0):
            r[r < 0] = 0.0
        # check overlap; if any, shrink the smaller of the two
        ok = True
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.hypot(dx, dy)
                need = r[i] + r[j]
                if dist < need - 1e-9:
                    ok = False
                    # shrink both to fit with a little slack
                    total = max(need - dist, 0.0)
                    # distribute shrink proportional to radii
                    s = r[i] + r[j]
                    if s <= 1e-12:
                        r[i] = 0.0
                        r[j] = 0.0
                    else:
                        shrink = total * 1.02
                        r[i] = max(0.0, r[i] - shrink * r[i] / s)
                        r[j] = max(0.0, r[j] - shrink * r[j] / s)
        if ok:
            break
    # final clip
    for k in range(N):
        r[k] = min(r[k], x[k], 1.0 - x[k], y[k], 1.0 - y[k], 0.5)
    return x, y, r, _is_valid(x, y, r)


def _optimize_from(x0, y0, r0):
    from scipy.optimize import minimize
    z0 = np.concatenate([np.concatenate([x0, y0]), r0])
    res = minimize(
        _objective, z0, method="L-BFGS-B", jac=True,
        bounds=[(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N,
        options={"maxiter": 300, "maxfun": 4000, "gtol": 1e-7, "ftol": 1e-9},
    )
    z = res.x
    x = z[:2 * N].reshape(N, 2)[:, 0]
    y = z[:2 * N].reshape(N, 2)[:, 1]
    r = z[2 * N:]
    return x, y, r


def construct():
    rng = np.random.default_rng(12345)

    best = None
    best_val = -np.inf

    def try_candidate(x, y, r):
        nonlocal best, best_val
        x, y, r, valid = _repair(x, y, r)
        if not valid:
            return
        val = float(np.sum(r))
        if val > best_val:
            best_val = val
            best = (x.copy(), y.copy(), r.copy())

    # --- baseline: jittered grid, repaired ---
    x, y, r = _make_initial(rng)
    try_candidate(x, y, r)

    # --- L-BFGS-B multi-restart ---
    restarts = 6
    for k in range(restarts):
        if _elapsed() > _DEADLINE - 25.0:
            break
        if k == 0:
            x0, y0, r0 = _make_initial(np.random.default_rng(12345))
        else:
            x0, y0, r0 = _make_initial(rng)
        # small random radius variation to escape symmetry
        r0 = r0 * (0.85 + 0.3 * rng.random(N))
        try:
            xo, yo, ro = _optimize_from(x0, y0, r0)
        except Exception:
            continue
        try_candidate(xo, yo, ro)

    # --- final polish: a couple of extra L-BFGS-B passes on the best result ---
    if best is not None:
        for _ in range(2):
            if _elapsed() > _DEADLINE - 10.0:
                break
            xb, yb, rb = best
            try:
                xo, yo, ro = _optimize_from(xb, yb, rb)
            except Exception:
                break
            x, y, r, valid = _repair(xo, yo, ro)
            if valid:
                val = float(np.sum(r))
                if val > best_val:
                    best_val = val
                    best = (x.copy(), y.copy(), r.copy())

    if best is None:
        # absolute fallback: tiny valid circles
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        arr = np.array(cells[:N])
        return arr

    x, y, r = best
    # tiny safety shrink to avoid boundary/overlap tolerance issues
    r = np.clip(r, 0.0, 0.5)
    for k in range(N):
        r[k] = min(r[k], x[k] - 1e-7, 1.0 - x[k] - 1e-7, y[k] - 1e-7, 1.0 - y[k] - 1e-7)
    x = np.clip(x, 1e-7, 1.0 - 1e-7)
    y = np.clip(y, 1e-7, 1.0 - 1e-7)

    out = np.column_stack([x, y, r])
    return out


if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print("sum r =", res[:, 2].sum())
    print(res)
