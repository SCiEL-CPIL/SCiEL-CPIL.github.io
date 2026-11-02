import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 150.0
SAFETY_MARGIN = 8.0


def _log_energy(X):
    """Sum over i<j of log(1/d) with chordal distances, via math.fsum."""
    n = X.shape[0]
    terms = []
    for i in range(n):
        d = X[i] - X[i + 1:]
        dist = np.sqrt(np.sum(d * d, axis=1))
        terms.extend(math.log(1.0 / di) for di in dist)
    return math.fsum(terms)


def _energy_and_grad(X):
    """Energy and Riemannian gradient (tangent) for log(1/d) sum over pairs."""
    n = X.shape[0]
    diff = X[:, None, :] - X[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    dist = np.sqrt(dist2)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    E = -np.sum(np.log(dist[iu]))
    
    # Euclidean gradient: grad_eucl[i] = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    inv_dist2 = np.zeros_like(dist2)
    mask = dist2 > 0
    inv_dist2[mask] = 1.0 / dist2[mask]
    np.fill_diagonal(inv_dist2, 0.0)
    eucl_grad = -np.sum(diff * inv_dist2[:, :, None], axis=1)
    
    # Project to tangent space
    dot = np.sum(eucl_grad * X, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * X
    
    return E, riem_grad


def _project_to_sphere(X):
    """Normalize each row to unit norm."""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < 1e-15] = 1.0
    return X / nrm


def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_init(n, seed=1):
    """Halton sequence mapped to sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    pts = []
    for i in range(seed, seed + n):
        pts.append([halton(i, 2), halton(i, 3), halton(i, 5)])
    pts = np.array(pts)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts = pts / norms
    return pts


def _farthest_point_init(n, rng_seed=42, n_candidates=50):
    """Farthest-point sampling for diverse initialization."""
    rng = np.random.default_rng(rng_seed)
    points = np.zeros((n, 3))
    
    # Start with random point
    p = rng.standard_normal(3)
    points[0] = p / np.linalg.norm(p)
    
    for i in range(1, n):
        # Generate candidates
        candidates = rng.standard_normal((n_candidates, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, compute min distance to existing points
        best_min_dist = -1
        best_candidate = candidates[0]
        
        for c in candidates:
            dists = np.linalg.norm(points[:i] - c, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = c
        
        points[i] = best_candidate
    
    return points


def _lbfgs_refine(X, time_budget):
    """L-BFGS-B on ambient coords with periodic re-projection."""
    best = X.copy()
    best_E = _log_energy(best)
    t0 = time.time()

    def fun(g):
        Xc = _project_to_sphere(g.reshape(N, 3))
        E, _ = _energy_and_grad(Xc)
        return E

    def jac(g):
        Xc = _project_to_sphere(g.reshape(N, 3))
        _, G = _energy_and_grad(Xc)
        # G is already Riemannian (tangent) gradient
        return G.reshape(-1)

    try:
        res = minimize(fun, X.reshape(-1), jac=jac, method='L-BFGS-B',
                       options={'maxiter': 500, 'maxfun': 1000, 'maxls': 40,
                                'ftol': 1e-14, 'gtol': 1e-10})
        Xr = _project_to_sphere(res.x.reshape(N, 3))
        Er = _log_energy(Xr)
        if Er < best_E:
            best = Xr
            best_E = Er
    except Exception:
        pass
    
    return best, best_E


def _riemannian_cg(X, time_budget, max_iter=5000):
    """Riemannian conjugate gradient with momentum on the sphere (fixed tangent projection)."""
    best = X.copy()
    best_E = _log_energy(best)
    t0 = time.time()
    Xc = X.copy()
    E, G = _energy_and_grad(Xc)
    d = -G.copy()
    # Project initial direction to tangent space
    d = d - np.sum(d * Xc, axis=1, keepdims=True) * Xc
    beta = 0.0
    velocity = np.zeros_like(Xc)
    momentum = 0.0
    stagnation = 0
    
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        
        # Gradually increase momentum
        if it >= 100:
            momentum = 0.2
        elif it >= 50:
            momentum = 0.1
        
        # Effective direction with momentum, projected to tangent space
        if momentum > 0:
            effective_d = d + momentum * velocity
        else:
            effective_d = d
        # Project to tangent space to prevent drift
        effective_d = effective_d - np.sum(effective_d * Xc, axis=1, keepdims=True) * Xc
        
        # Backtracking line search with Armijo condition
        alpha = 0.1
        prev_E = E
        accepted = False
        for ls in range(30):
            Xn = Xc + alpha * effective_d
            Xn = _project_to_sphere(Xn)
            En, Gn = _energy_and_grad(Xn)
            if En < prev_E - 1e-14:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Try with plain gradient direction
            alpha = 0.05
            for ls in range(20):
                Xn = Xc - alpha * G
                Xn = _project_to_sphere(Xn)
                En, Gn = _energy_and_grad(Xn)
                if En < prev_E - 1e-14:
                    accepted = True
                    break
                alpha *= 0.5
        
        if not accepted:
            # Reset direction and try smaller step
            alpha = 0.01
            Xn = Xc - alpha * G
            Xn = _project_to_sphere(Xn)
            En, Gn = _energy_and_grad(Xn)
            if En >= prev_E:
                break
            accepted = True
        
        # Update velocity (projected to tangent space)
        if momentum > 0:
            velocity = momentum * velocity - alpha * G
            velocity = velocity - np.sum(velocity * Xc, axis=1, keepdims=True) * Xc
        
        Xc, E = Xn, En
        if E < best_E:
            best = Xc.copy()
            best_E = E
            stagnation = 0
        else:
            stagnation += 1
        
        # Check convergence
        grad_norm = np.linalg.norm(Gn)
        if grad_norm < 1e-13:
            break
        
        # Update conjugate direction (Polak-Ribiere) with tangent projection
        G_old = G
        G = Gn
        beta_num = np.sum(G * (G - G_old))
        beta_den = np.sum(G_old * G_old)
        beta = max(0.0, beta_num / max(beta_den, 1e-30))
        d = -G + beta * d
        # CRITICAL: Project to tangent space to prevent drift
        d = d - np.sum(d * Xc, axis=1, keepdims=True) * Xc
    
    return best, best_E


def _final_polish(X, time_budget):
    """High-precision final polish with tight tolerances and multiple strategies."""
    t0 = time.time()
    best = X.copy()
    best_E = _log_energy(best)
    
    # Strategy 1: L-BFGS with very tight tolerances
    for attempt in range(4):
        if time.time() - t0 > time_budget:
            break
        
        def fun(g):
            Xc = _project_to_sphere(g.reshape(N, 3))
            E, _ = _energy_and_grad(Xc)
            return E

        def jac(g):
            Xc = _project_to_sphere(g.reshape(N, 3))
            _, G = _energy_and_grad(Xc)
            return G.reshape(-1)

        try:
            res = minimize(fun, best.reshape(-1), jac=jac, method='L-BFGS-B',
                           options={'maxiter': 300, 'maxfun': 600, 'maxls': 30,
                                    'ftol': 1e-18, 'gtol': 1e-14, 'maxcor': 20})
            Xr = _project_to_sphere(res.x.reshape(N, 3))
            Er = _log_energy(Xr)
            if Er < best_E:
                best = Xr
                best_E = Er
        except Exception:
            pass
    
    # Strategy 2: Short Riemannian CG polish from the L-BFGS result
    if time.time() < t0 + time_budget - 1.0:
        rem = time_budget - (time.time() - t0)
        if rem > 2.0:
            Xcg, Ecg = _riemannian_cg(best, min(rem - 1.0, 5.0), max_iter=2000)
            if Ecg < best_E:
                best = Xcg
                best_E = Ecg
    
    # Strategy 3: One more L-BFGS pass
    if time.time() < t0 + time_budget - 0.5:
        def fun(g):
            Xc = _project_to_sphere(g.reshape(N, 3))
            E, _ = _energy_and_grad(Xc)
            return E

        def jac(g):
            Xc = _project_to_sphere(g.reshape(N, 3))
            _, G = _energy_and_grad(Xc)
            return G.reshape(-1)

        try:
            res = minimize(fun, best.reshape(-1), jac=jac, method='L-BFGS-B',
                           options={'maxiter': 100, 'maxfun': 200, 'maxls': 20,
                                    'ftol': 1e-18, 'gtol': 1e-15, 'maxcor': 10})
            Xr = _project_to_sphere(res.x.reshape(N, 3))
            Er = _log_energy(Xr)
            if Er < best_E:
                best = Xr
                best_E = Er
        except Exception:
            pass
    
    return best, best_E


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN

    # Multiple diverse initializations
    inits = []
    inits.append(_fibonacci_init())
    inits.append(_halton_init(N, seed=1))
    inits.append(_farthest_point_init(N, rng_seed=42, n_candidates=50))
    
    rng = np.random.default_rng(12345)
    inits.append(_project_to_sphere(_fibonacci_init() + 0.03 * rng.standard_normal((N, 3))))
    inits.append(_project_to_sphere(_fibonacci_init() + 0.15 * rng.standard_normal((N, 3))))
    inits.append(_project_to_sphere(rng.standard_normal((N, 3))))
    inits.append(_project_to_sphere(rng.standard_normal((N, 3))))
    inits.append(_project_to_sphere(rng.standard_normal((N, 3))))
    
    best = inits[0].copy()
    best_E = _log_energy(best)

    # Phase 1: Riemannian CG from each init
    for k, X0 in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        rem = max(1.0, deadline - 15.0 - time.time())
        budget_per = rem / (len(inits) - k)
        Xb, Eb = _riemannian_cg(X0, min(budget_per, 12.0))
        if Eb < best_E:
            best, best_E = Xb, Eb

    # Phase 2: L-BFGS polish on best
    if time.time() < deadline - 8.0:
        Xp, Ep = _lbfgs_refine(best, min(15.0, deadline - 8.0 - time.time()))
        if Ep < best_E:
            best, best_E = Xp, Ep

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 2.0:
        Xf, Ef = _final_polish(best, min(6.0, deadline - 2.0 - time.time()))
        if Ef < best_E:
            best, best_E = Xf, Ef

    # Ensure valid: finite, normalized, no near-duplicates
    best = _project_to_sphere(best)
    if not np.all(np.isfinite(best)):
        best = _fibonacci_init()
    
    # Check for near-duplicates
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best[i] - best[j])
            if d < 1e-9:
                # Nudge one point slightly
                best[j] = best[j] + 1e-8 * rng.standard_normal(3)
                best[j] /= np.linalg.norm(best[j])
    
    return best
