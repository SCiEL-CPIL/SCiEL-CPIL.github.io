"""
Minimize logarithmic energy of 150 points on the unit sphere.
E = sum_{i<j} log(1/|x_i - x_j|)  (natural log, chordal distances)

Strategy (combined):
  - Global: Riemannian L-BFGS directly on the sphere (no projection artifacts)
  - Pre-condition: Fast Cartesian repulsion to spread points
  - Polish: Riemannian Conjugate Gradient (CG) for high-precision local minimum
  - Time management: Strict deadline guards
"""
import time
import math
import numpy as np
from scipy.optimize import minimize  # Kept for potential fallback, though we implement custom Riemannian optimizers

N = 150
_DEADLINE = 150.0
_START = time.time()
_SAFETY = 10.0  # Safety margin in seconds
_TIME_LIMIT = _DEADLINE - _SAFETY


def _time_left():
    return _TIME_LIMIT - (time.time() - _START)


def _retract(X):
    """Project points back to the unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-15, 1.0, n)
    return X / n


def _energy_and_grad(X, eps=1e-14):
    """
    Vectorized energy and Euclidean gradient.
    E = sum_{i<j} log(1/|x_i - x_j|) = - sum_{i<j} log(|x_i - x_j|)
    Gradient of E w.r.t X:
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    diff = X[:, None, :] - X[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, eps * eps)
    d = np.sqrt(d2)
    log_d = np.log(d)
    np.fill_diagonal(log_d, 0.0)
    E = -0.5 * float(np.sum(log_d))
    
    # Gradient calculation
    inv = 1.0 / d2
    np.fill_diagonal(inv, 0.0)
    # S1[i] = sum_j (inv[i,j] * X[j])
    S1 = inv @ X
    # S2[i] = (sum_j inv[i,j]) * X[i]
    S2 = (inv.sum(axis=1, keepdims=True)) * X
    # G = S2 - S1
    G = S2 - S1
    return E, G


def _project_tangent(G, X):
    """Project Euclidean gradient to the tangent space of the sphere at X."""
    a = np.einsum('ij,ij->i', G, X)
    return G - a[:, None] * X


def _bfgs_two_loop(v, s_list, y_list, rho_list, m):
    """Two-loop recursion for L-BFGS approximation of inverse Hessian."""
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
    """
    Riemannian L-BFGS on the sphere with retraction and strong Wolfe line search.
    """
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    flat = g.ravel()

    s_list, y_list, rho_list = [], [], []
    it = 0
    best_E = E
    best_X = X.copy()

    while _time_left() > 2.0 and it < max_iter:
        # Compute search direction
        p_flat = -_bfgs_two_loop(flat, s_list, y_list, rho_list, min(len(s_list), m))
        p = p_flat.reshape(N, 3)
        p = _project_tangent(p, X)
        pnorm = float(np.linalg.norm(p))
        if pnorm < 1e-13:
            break

        gnorm = float(np.linalg.norm(flat))
        if gnorm < tol:
            break

        # Ensure descent direction
        c1 = 1e-4
        c2 = 0.9
        slope = float(np.dot(flat, p_flat))
        if slope >= 0:
            # Fallback to steepest descent
            p = -g
            p = _project_tangent(p, X)
            p_flat = p.ravel()
            slope = float(np.dot(flat, p_flat))
            if slope >= 0:
                break

        # Line search (Strong Wolfe conditions)
        alpha = 1.0
        Xn, En, Gn, gn = X, E, G, g
        for _ in range(60):
            if _time_left() < 3.0:
                break
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En <= E + c1 * alpha * slope:
                if float(np.dot(gn.ravel(), p_flat)) <= c2 * alpha * slope:
                    break
            alpha *= 0.5

        # If line search failed, try a very small step
        if En >= E:
            alpha = max(alpha * 0.1, 1e-8)
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En >= E:
                break

        # Update L-BFGS memory
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


def _cg_sphere(X0, time_budget, max_iter=2000, tol=1e-10):
    """
    Riemannian Conjugate Gradient on the sphere with backtracking line search.
    Used for final polishing.
    """
    X = _retract(X0.copy())
    E, G = _energy_and_grad(X)
    g = _project_tangent(G, X)
    p = -g
    best_E = E
    best_X = X.copy()
    beta = 0.0
    it = 0

    while _time_left() > 2.0 and it < max_iter:
        gnorm = float(np.linalg.norm(g))
        if gnorm < tol:
            break

        p = _project_tangent(p, X)
        pnorm = float(np.linalg.norm(p))
        if pnorm < 1e-14:
            break

        slope = float(np.dot(g.ravel(), p.ravel()))
        if slope >= 0:
            # Reset direction to steepest descent
            p = -g
            p = _project_tangent(p, X)
            slope = float(np.dot(g.ravel(), p.ravel()))
            if slope >= 0:
                break
            beta = 0.0

        # Backtracking line search
        alpha = 1.0
        Xn, En, Gn, gn = X, E, G, g
        for _ in range(80):
            if _time_left() < 3.0:
                break
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En <= E + 1e-4 * alpha * slope:
                break
            alpha *= 0.5

        if En >= E:
            alpha = max(alpha * 0.1, 1e-8)
            Xn = _retract(X + alpha * p)
            En, Gn = _energy_and_grad(Xn)
            gn = _project_tangent(Gn, Xn)
            if En >= E:
                break

        # Update direction (Polak-Ribiere formula)
        gnew2 = float(np.dot(gn.ravel(), gn.ravel()))
        if gnorm > 0:
            beta = gnew2 / (gnorm * gnorm)
            if beta > 1.0:
                beta = 1.0

        p = -gn + beta * p
        p = _project_tangent(p, X)

        X = Xn
        E = En
        G = Gn
        g = gn
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
    n_lat, n_lon = 12, 13
    points = []
    for i in range(n_lat):
        theta = (i + 0.5) * math.pi / n_lat
        for j in range(n_lon):
            phi = (j + 0.5) * 2 * math.pi / n_lon
            points.append([math.sin(theta) * math.cos(phi),
                           math.sin(theta) * math.sin(phi),
                           math.cos(theta)])
    while len(points) < N:
        rng = np.random.default_rng(len(points) * 7919)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        points.append(v)
    return _retract(np.array(points[:N]))


def _repulsion_refine_cartesian(pts, n_iters=50, lr=0.01, deadline=None):
    """
    Fast Cartesian repulsion pre-conditioner.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1.0)
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        pts += lr * forces
        pts = _retract(pts)
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_fibonacci_spiral(0.0))
    inits.append(_fibonacci_spiral(0.5))
    inits.append(_fibonacci_spiral(1.0))
    inits.append(_random_points(42))
    inits.append(_random_points(123))
    inits.append(_structured_grid())
    
    best_pts = None
    best_energy = 1e30
    
    # Phase 1: Pre-condition and Optimize with Riemannian L-BFGS
    for i, init in enumerate(inits):
        if _time_left() < 5.0:
            break
            
        # 1. Pre-condition with Cartesian repulsion
        refined_cart = _repulsion_refine_cartesian(init, n_iters=50, lr=0.01, deadline=deadline)
        
        # 2. Optimize with Riemannian L-BFGS
        # Allocate remaining time for this optimizer, but keep some for polishing
        # We use a fraction of remaining time for L-BFGS
        time_budget = _time_left() - 5.0
        if time_budget < 1.0:
            break
            
        X_opt, E_opt = _lbfgs_sphere(refined_cart, time_budget)
        
        # Check validity
        diff = X_opt[:, np.newaxis, :] - X_opt[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 1e-9:
            continue
            
        if E_opt < best_energy:
            best_energy = E_opt
            best_pts = X_opt.copy()
            
    # Phase 2: Polish with Riemannian CG
    if best_pts is not None and _time_left() > 2.0:
        X_polished, E_polished = _cg_sphere(best_pts, _time_left() - 1.0)
        
        # Check validity
        diff = X_polished[:, np.newaxis, :] - X_polished[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist >= 1e-9:
            if E_polished < best_energy:
                best_energy = E_polished
                best_pts = X_polished.copy()
    
    if best_pts is None:
        # Fallback to Fibonacci spiral if optimization failed completely
        best_pts = _fibonacci_spiral(0.0)
    
    # Final normalization
    best_pts = _retract(best_pts)
    
    return best_pts

# EVOLVE-BLOCK-END
