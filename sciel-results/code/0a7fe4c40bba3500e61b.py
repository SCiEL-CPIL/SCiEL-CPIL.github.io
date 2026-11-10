"""
Minimize logarithmic energy of 150 points on the unit sphere.
E = sum_{i<j} log(1/|x_i - x_j|)  (natural log, chordal distances)

Strategy:
  1. Fast initial optimization from multiple diverse starts using L-BFGS on the sphere.
  2. Additional starts from structured and random configurations.
  3. High-precision polish phase with tighter tolerances.
  4. Local perturbations to escape shallow local minima.
"""
import time
import math
import numpy as np

N = 150
_DEADLINE = 150.0
_START = time.time()
_SAFETY = 8.0
_TIME_LIMIT = _DEADLINE - _SAFETY


def _time_left():
    return _TIME_LIMIT - (time.time() - _START)


def _retract(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return X / n


def _energy_and_grad(X, eps=1e-14):
    """Vectorized energy and gradient. Returns (E, G) where G is the Euclidean gradient."""
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, eps * eps)
    d = np.sqrt(d2)
    # E = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    log_d = np.log(d)
    np.fill_diagonal(log_d, 0.0)
    E = -0.5 * float(np.sum(log_d))
    # grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv = 1.0 / d2
    np.fill_diagonal(inv, 0.0)
    S1 = inv @ X
    S2 = (inv.sum(axis=1, keepdims=True)) * X
    G = S2 - S1
    return E, G


def _project_tangent(G, X):
    a = np.einsum('ij,ij->i', G, X)
    return G - a[:, None] * X


def _bfgs_two_loop(v, s_list, y_list, rho_list, m):
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


def _lbfgs_sphere(X0, time_budget, m=15, max_iter=5000, tol=1e-10):
    """L-BFGS on the sphere with retraction. Returns (best_X, best_E)."""
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    flat = g.ravel()

    s_list, y_list, rho_list = [], [], []
    it = 0
    best_E = E
    best_X = X.copy()

    while _time_left() > 2.0 and it < max_iter:
        # Search direction
        p_flat = -_bfgs_two_loop(flat, s_list, y_list, rho_list, min(len(s_list), m))
        p = p_flat.reshape(N, 3)
        p = _project_tangent(p, X)
        pnorm = float(np.linalg.norm(p))
        if pnorm < 1e-13:
            break

        gnorm = float(np.linalg.norm(flat))
        if gnorm < tol:
            break

        # Armijo line search
        c1 = 1e-4
        c2 = 0.9
        slope = float(np.dot(flat, p_flat))
        if slope >= 0:
            # Direction is not descent; try steepest descent
            p = -g
            p = _project_tangent(p, X)
            p_flat = p.ravel()
            slope = float(np.dot(flat, p_flat))
            if slope >= 0:
                break

        alpha = 1.0
        Xn = X
        En = E
        Gn = G
        gn = g
        for _ in range(50):
            if _time_left() < 3.0:
                break
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En <= E + c1 * alpha * slope:
                if float(np.dot(gn.ravel(), p_flat)) <= c2 * alpha * slope:
                    break
            alpha *= 0.5

        if En >= E:
            # Try a smaller step
            alpha = max(alpha * 0.1, 1e-8)
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En >= E:
                break

        # Update history
        dX = (Xn - X).ravel()
        dy = gn.ravel() - flat
        sy = float(np.dot(dX, dy))
        if sy > 1e-14:
            s_list.append(dX)
            y_list.append(dy)
            rho_list.append(1.0 / sy)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)

        X = Xn
        E = En
        G = Gn
        g = gn
        flat = g.ravel()
        it += 1

        if E < best_E:
            best_E = E
            best_X = X.copy()

    return best_X, best_E


def _cg_sphere(X0, time_budget, max_iter=3000, tol=1e-10):
    """Riemannian conjugate gradient on the sphere. Returns (best_X, best_E)."""
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    p = -g.copy()
    best_E = E
    best_X = X.copy()
    it = 0

    while _time_left() > 2.0 and it < max_iter:
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            break

        # Line search
        p_flat = p.ravel()
        slope = float(np.dot(g.ravel(), p_flat))
        if slope >= 0:
            p = -g
            p_flat = p.ravel()
            slope = float(np.dot(g.ravel(), p_flat))
            if slope >= 0:
                break

        c1 = 1e-4
        c2 = 0.9
        alpha = 1.0
        Xn = X
        En = E
        Gn = G
        gn = g
        for _ in range(40):
            if _time_left() < 3.0:
                break
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En <= E + c1 * alpha * slope:
                if float(np.dot(gn.ravel(), p_flat)) <= c2 * alpha * slope:
                    break
            alpha *= 0.5

        if En >= E:
            break

        # Polak-Ribiere update
        old_g = g
        g = gn
        gy = float(np.dot(g.ravel(), (g - old_g).ravel()))
        beta = max(0.0, gy / float(np.dot(old_g.ravel(), old_g.ravel())))
        p = -g + beta * p
        p = _project_tangent(p, X)

        X = Xn
        E = En
        G = Gn
        it += 1

        if E < best_E:
            best_E = E
            best_X = X.copy()

    return best_X, best_E


def _fibonacci_spiral(offset=0.0):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(N, 3))
    return _retract(v)


def _structured_grid():
    """Generate points on a structured grid on the sphere (latitude/longitude with jitter)."""
    n_lat = 12
    n_lon = 13
    points = []
    for i in range(n_lat):
        theta = (i + 0.5) * math.pi / n_lat
        for j in range(n_lon):
            phi = (j + 0.5) * 2 * math.pi / n_lon
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            points.append([x, y, z])
    # Add extra points to reach N
    while len(points) < N:
        rng = np.random.default_rng(len(points) * 7919)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        points.append(v)
    return _retract(np.array(points[:N]))


def _perturb(X, seed, sigma):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=X.shape) * sigma
    return _retract(X + noise)


def _swap_perturb(X, seed, n_swaps=5):
    """Swap positions of n_swaps pairs of points and add small noise."""
    rng = np.random.default_rng(seed)
    X_new = X.copy()
    for _ in range(n_swaps):
        i, j = rng.choice(N, size=2, replace=False)
        X_new[i], X_new[j] = X_new[j].copy(), X_new[i].copy()
        # Add small noise to swapped points
        noise = rng.normal(size=(2, 3)) * 0.05
        X_new[i] += noise[0]
        X_new[j] += noise[1]
    return _retract(X_new)


def _validate(X):
    """Check validity: finite, unit norm, no near-duplicates."""
    if not np.all(np.isfinite(X)):
        return False
    norms = np.linalg.norm(X, axis=1)
    if not np.all(np.abs(norms - 1.0) < 1e-6):
        return False
    # Check minimum pairwise distance
    diff = X[:, None, :] - X[None, :, :]
    dd = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dd, 2.0)
    if float(dd.min()) < 1e-9:
        return False
    return True


def _log_energy_precise(X):
    """Compute energy using math.fsum for precision."""
    terms = []
    for i in range(N):
        for j in range(i + 1, N):
            d = math.sqrt(float(np.dot(X[i] - X[j], X[i] - X[j])))
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fast_grad_descent(X0, time_budget, lr=0.001, max_iter=2000):
    """Fast gradient descent on the sphere as a quick pre-pass."""
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        g = _project_tangent(G, X)
        gnorm = float(np.linalg.norm(g))
        if gnorm < 1e-10:
            break
        
        # Adaptive step size
        alpha = lr / (1.0 + it * 0.01)
        Xn = _retract(X - alpha * g)
        En, Gn = _energy_and_grad(Xn)
        
        if En < E:
            X = Xn
            E = En
            G = Gn
            if E < best_E:
                best_E = E
                best_X = X.copy()
        else:
            # Reduce learning rate
            lr *= 0.5
    
    return best_X, best_E


def _random_spherical(seed, n=N):
    """Generate random points uniformly on the sphere."""
    rng = np.random.default_rng(seed)
    # Use Marsaglia's method for uniform distribution
    pts = []
    while len(pts) < n:
        v = rng.normal(size=(n, 3))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / norms
        pts.append(v)
    return np.vstack(pts[:n])


def _sobol_like(n=N, seed=42):
    """Generate quasi-random points using a simple low-discrepancy sequence."""
    rng = np.random.default_rng(seed)
    # Use stratified sampling in 3D then project to sphere
    k = int(np.ceil(n ** (1/3)))
    points = []
    for i in range(k):
        for j in range(k):
            for l in range(k):
                if len(points) >= n:
                    break
                u = (i + 0.5) / k
                v = (j + 0.5) / k
                w = (l + 0.5) / k
                # Map cube to sphere
                theta = np.arccos(2*u - 1)
                phi = 2 * np.pi * v
                r = np.sqrt(1 - (2*w - 1)**2)
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                z = 2*w - 1
                points.append([x, y, z])
    return _retract(np.array(points[:n]))


def construct():
    best_X = _fibonacci_spiral()
    best_E = _energy_and_grad(best_X)[0]

    # Phase 0: Quick gradient descent from initial config
    if _time_left() > 5.0:
        X, E = _fast_grad_descent(best_X, 5.0)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 1: Optimize from multiple diverse starts
    starts = []
    # Fibonacci with different offsets
    for offset in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        starts.append(_fibonacci_spiral(offset))
    # Structured grid
    starts.append(_structured_grid())
    # Random starts with various seeds
    for s in [42, 123, 456, 789, 1024, 2048, 3000, 4096, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]:
        starts.append(_random_points(s))
    # Sobol-like quasi-random
    for s in [42, 123, 456]:
        starts.append(_sobol_like(seed=s))
    
    # First pass: quick gradient descent from all starts
    for start in starts:
        if _time_left() > 8.0:
            X, E = _fast_grad_descent(start, 3.0)
            if E < best_E:
                best_E, best_X = E, X

    # Second pass: L-BFGS from the best quick results
    # Keep track of top candidates
    candidates = []
    for start in starts[:10]:  # Only try top 10 starts with L-BFGS
        if _time_left() > 12.0:
            X, E = _lbfgs_sphere(start, 10.0)
            candidates.append((E, X))
            if E < best_E:
                best_E, best_X = E, X

    # Sort candidates and try polishing the best few
    candidates.sort(key=lambda x: x[0])
    for E_cand, X_cand in candidates[:3]:
        if _time_left() > 8.0:
            X, E = _cg_sphere(X_cand, 6.0)
            if E < best_E:
                best_E, best_X = E, X

    # Phase 2: CG optimization from best
    if _time_left() > 8.0:
        X, E = _cg_sphere(best_X, 8.0)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 3: Local perturbations around best to escape local minima
    for sigma in [0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01]:
        if _time_left() > 6.0:
            for trial in range(4):
                if _time_left() < 3.0:
                    break
                start = _perturb(best_X, trial * 1000 + int(sigma * 10000), sigma)
                X, E = _lbfgs_sphere(start, 6.0)
                if E < best_E:
                    best_E, best_X = E, X

    # Phase 4: Swap perturbations
    if _time_left() > 8.0:
        for trial in range(8):
            if _time_left() < 3.0:
                break
            start = _swap_perturb(best_X, 50000 + trial * 111, n_swaps=2 + trial)
            X, E = _lbfgs_sphere(start, 5.0)
            if E < best_E:
                best_E, best_X = E, X

    # Phase 5: High-precision polish with tighter tolerance
    if _time_left() > 15.0:
        X, E = _lbfgs_sphere(best_X, 12.0, m=20, max_iter=5000, tol=1e-12)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 6: CG polish
    if _time_left() > 8.0:
        X, E = _cg_sphere(best_X, 8.0, tol=1e-12)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 7: Final ultra-precise polish
    if _time_left() > 8.0:
        X, E = _lbfgs_sphere(best_X, 6.0, m=25, max_iter=3000, tol=1e-14)
        if E < best_E:
            best_E, best_X = E, X

    # Phase 8: Additional local search with smaller perturbations
    if _time_left() > 10.0:
        for sigma in [0.005, 0.002, 0.001, 0.0005]:
            if _time_left() < 3.0:
                break
            for trial in range(5):
                if _time_left() < 2.0:
                    break
                start = _perturb(best_X, 90000 + trial * 100 + int(sigma * 100000), sigma)
                X, E = _lbfgs_sphere(start, 4.0, tol=1e-12)
                if E < best_E:
                    best_E, best_X = E, X

    # Final validation
    best_X = _retract(best_X)
    if not _validate(best_X):
        # Fallback to Fibonacci spiral
        best_X = _fibonacci_spiral()

    return best_X
