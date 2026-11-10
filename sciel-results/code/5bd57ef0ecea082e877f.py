"""Fresh design: Riemannian L-BFGS on the unit sphere for logarithmic energy minimization.

Entry point construct() -> (N,3) array of unit vectors (normalized).
Minimizes E = sum_{i<j} log(1 / |x_i - x_j|)  (natural log, chordal distances).
"""
import time
import math
import numpy as np

N = 150

# ---- wall-clock guard -------------------------------------------------------
_DEADLINE = 120.0
_START = time.time()
_SAFETY = 5.0          # seconds of margin before the hard kill
_TIME_LIMIT = _DEADLINE - _SAFETY


def _time_left():
    return _TIME_LIMIT - (time.time() - _START)


def _log_energy_precise(X):
    """Compute logarithmic energy using math.fsum for maximum precision."""
    total = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(X[i] - X[j])
            total.append(math.log(1.0 / d))
    return math.fsum(total)


# ---- energy and analytic gradient (vectorized O(N^2)) -----------------------
def _energy_and_grad(X, eps=1e-12):
    # X: (N,3)
    diff = X[:, None, :] - X[None, :, :]          # (N,N,3)
    d2 = np.einsum('ijk,ijk->ij', diff, diff)     # squared chordal distances
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, eps * eps)
    d = np.sqrt(d2)
    # E = - sum_{i<j} log(d)
    E = -0.5 * float(np.sum(np.log(d)))
    # grad_i = - sum_{j != i} (X_i - X_j) / |X_i - X_j|^2
    #         = sum_j X_i/d2 - sum_j X_j/d2   (diag of 1/d2 set to 0)
    inv = 1.0 / d2
    np.fill_diagonal(inv, 0.0)
    S1 = inv @ X                                   # sum_j X_j / d2_ij
    S2 = (inv.sum(axis=1, keepdims=True)) * X      # sum_j X_i / d2_ij
    G = S2 - S1                                    # (N,3)
    return E, G


def _structured_grid():
    """Generate a structured grid of points on the sphere using spherical coordinates."""
    # Use a lat/lon grid with golden-ratio offset to avoid regular artifacts
    n_lat = 12
    n_lon = 13
    points = []
    for i in range(n_lat):
        # Avoid exact poles
        theta = (i + 0.5) * np.pi / n_lat
        for j in range(n_lon):
            phi = (j + 0.5) * 2 * np.pi / n_lon + i * 0.3  # staggered
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
    # Trim or pad to N
    points = np.array(points[:N])
    if len(points) < N:
        # Add random points to fill
        rng = np.random.default_rng(42)
        extra = rng.normal(size=(N - len(points), 3))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        points = np.vstack([points, extra])
    return _retract(points)


def _project_tangent(G, X):
    # project G onto tangent space of the sphere at each X_i
    a = np.einsum('ij,ij->i', G, X)
    return G - a[:, None] * X


def _retract(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return X / n


# ---- L-BFGS two-loop recursion (operates on the flattened tangent vector) ---
def _bfgs_two_loop(v, s_list, y_list, rho_list, m):
    # v: (N*3,) vector; returns B v
    n = len(v)
    q = v.copy()
    alphas = np.zeros(m)
    for k in range(m - 1, -1, -1):
        s = s_list[k]
        y = y_list[k]
        rho = rho_list[k]
        alphas[k] = rho * float(np.dot(s, q))
        q = q - alphas[k] * y
    r = np.zeros(n)
    if y_list:
        yy = float(np.dot(y_list[-1], y_list[-1]))
        if yy > 1e-20:
            gamma = float(np.dot(s_list[-1], y_list[-1])) / yy
            r = gamma * q
    for k in range(m):
        s = s_list[k]
        y = y_list[k]
        rho = rho_list[k]
        beta = rho * float(np.dot(y, r))
        r = r + (alphas[k] - beta) * s
    return q + r


# ---- one L-BFGS run from a given start --------------------------------------
def _lbfsgs_run(X0, time_budget, m=15):
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    flat = g.ravel()

    s_list, y_list, rho_list = [], [], []
    it = 0
    best_E = E
    best_X = X.copy()

    # Track gradient norm for convergence check
    prev_gnorm = float(np.linalg.norm(flat))

    while _time_left() > 1.0:
        # search direction p = -B g  (B = BFGS approx of Hessian on tangent space)
        p = -_bfgs_two_loop(flat, s_list, y_list, rho_list, min(len(s_list), m))
        p = p.reshape(N, 3)
        p = _project_tangent(p, X)
        pnorm = float(np.linalg.norm(p))
        if pnorm < 1e-12:
            break

        # line search (Armijo + curvature) along p
        c1 = 1e-4
        c2 = 0.9
        gnorm = float(np.linalg.norm(flat))
        if gnorm < 1e-10:
            break
        alpha = 1.0
        found = False
        for _ in range(30):
            if _time_left() < 2.0:
                break
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            # sufficient decrease
            if En <= E + c1 * alpha * float(np.dot(flat, p.ravel())):
                # curvature condition
                if float(np.dot(gn.ravel(), p.ravel())) <= c2 * alpha * float(np.dot(flat, p.ravel())):
                    found = True
                    break
            alpha *= 0.5
        if not found:
            # accept the best tried step if it improves, else stop
            if En < E:
                found = True
            else:
                break

        # history update
        dX = (Xn - X).ravel()
        dy = (gn.ravel() - flat)
        sy = float(np.dot(dX, dy))
        if sy > 1e-12:
            s_list.append(dX)
            y_list.append(dy)
            rho_list.append(1.0 / sy)
            if len(s_list) > m:
                s_list.pop(0); y_list.pop(0); rho_list.pop(0)

        X = Xn
        E = En
        G = Gn
        g = gn
        flat = g.ravel()
        it += 1
        if E < best_E:
            best_E = E
            best_X = X.copy()
        
        # Check for convergence via gradient norm decrease
        curr_gnorm = float(np.linalg.norm(flat))
        if curr_gnorm < 1e-10 or (curr_gnorm > 0.999 * prev_gnorm and it > 50):
            break
        prev_gnorm = curr_gnorm
        
        if it % 5 == 0 and _time_left() <= 1.0:
            break

    return best_X, best_E


# ---- starting points --------------------------------------------------------
def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _random_start(seed):
    return _random_points(seed)


def _perturb_start(X, seed, sigma):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=X.shape) * sigma
    return _retract(X + noise)


def _validate(X):
    """Check that X is a valid point configuration."""
    if X.shape != (N, 3):
        return False
    if not np.all(np.isfinite(X)):
        return False
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms < 1e-12):
        return False
    # Check minimum distance
    diff = X[:, None, :] - X[None, :, :]
    dd = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dd, 2.0)
    min_d = float(dd.min())
    if min_d < 1e-9:
        return False
    return True


# ---- main construct ---------------------------------------------------------
def construct():
    best_X = _fibonacci_start()
    best_E = _energy_and_grad(best_X)[0]

    # Phase 1: optimize from multiple structured starts
    starts = [
        ("fibonacci", _fibonacci_start()),
        ("grid", _structured_grid()),
    ]
    for name, start in starts:
        if _time_left() > 10.0:
            X, E = _lbfsgs_run(start, 45.0)
            if E < best_E and _validate(X):
                best_E, best_X = E, X

    # Phase 2: several random starts, best-so-far
    seed = 12345
    while _time_left() > 15.0:
        start = _random_start(seed)
        X, E = _lbfsgs_run(start, 35.0)
        if E < best_E and _validate(X):
            best_E, best_X = E, X
        seed += 7919

    # Phase 3: local perturbations around the current best to escape local minima
    for sigma in (0.2, 0.1, 0.05):
        while _time_left() > 8.0:
            start = _perturb_start(best_X, seed, sigma)
            X, E = _lbfsgs_run(start, 25.0)
            if E < best_E and _validate(X):
                best_E, best_X = E, X
            seed += 104729

    # Final polish: run one more L-BFGS from the best point
    if _time_left() > 5.0:
        X, E = _lbfsgs_run(best_X.copy(), 20.0)
        if E < best_E and _validate(X):
            best_E, best_X = E, X

    # Final validation / normalization and sanity checks
    best_X = _retract(best_X)
    if not _validate(best_X):
        # fall back to the Fibonacci spiral (guaranteed valid)
        best_X = _fibonacci_start()
        best_X = _retract(best_X)

    return best_X
