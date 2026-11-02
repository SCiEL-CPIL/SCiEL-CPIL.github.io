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


def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along geodesics."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    # For small norms, use Taylor expansion; for larger, use exact formula
    small = norms < 1e-8
    cos_n = np.ones_like(norms)
    sin_n_over_n = np.ones_like(norms)
    
    large = ~small
    if np.any(large):
        nl = norms[large]
        cos_n[large] = np.cos(nl)
        sin_n_over_n[large] = np.sin(nl) / nl
    
    # For small norms: cos(norm) ≈ 1 - norm^2/2, sin(norm)/norm ≈ 1 - norm^2/6
    if np.any(small):
        ns = norms[small]
        cos_n[small] = 1.0 - 0.5 * ns * ns
        sin_n_over_n[small] = 1.0 - ns * ns / 6.0
    
    return cos_n * pts + sin_n_over_n * tangent_vecs


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
    
    p = rng.standard_normal(3)
    points[0] = p / np.linalg.norm(p)
    
    for i in range(1, n):
        candidates = rng.standard_normal((n_candidates, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
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


def _riemannian_cg_exp(X, time_budget, max_iter=5000):
    """Riemannian conjugate gradient using exponential map for line search."""
    best = X.copy()
    best_E = _log_energy(best)
    t0 = time.time()
    Xc = X.copy()
    E, G = _energy_and_grad(Xc)
    d = -G.copy()
    beta = 0.0
    
    for it in range(max_iter):
        if time.time() - t0 > time_budget:
            break
        
        # Backtracking line search using exponential map
        alpha = 0.5
        accepted = False
        prev_E = E
        for ls in range(30):
            Xn = _exponential_map(Xc, alpha * d)
            En, Gn = _energy_and_grad(Xn)
            if En <= prev_E * (1 + 1e-12) or En < prev_E:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Reset direction to negative gradient
            d = -G.copy()
            alpha = 0.1
            for ls in range(20):
                Xn = _exponential_map(Xc, alpha * d)
                En, Gn = _energy_and_grad(Xn)
                if En < prev_E:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
        
        Xc, E = Xn, En
        if E < best_E:
            best = Xc.copy()
            best_E = E
        
        # Check convergence
        if np.linalg.norm(Gn) < 1e-13:
            break
        
        # Update conjugate direction (Polak-Ribiere)
        G_old = G
        G = Gn
        beta_num = np.sum(G * (G - G_old))
        beta_den = np.sum(G_old * G_old)
        beta = max(0.0, beta_num / max(beta_den, 1e-30))
        d = -G + beta * d
        
        # Ensure d is in tangent space
        dot_d = np.sum(d * Xc, axis=1, keepdims=True)
        d -= dot_d * Xc
    
    return best, best_E


def _coordinate_descent(X, time_budget, max_sweeps=200):
    """Coordinate descent: optimize one point at a time on the sphere.
    This is a structurally different method that can find different local minima."""
    best = X.copy()
    best_E = _log_energy(best)
    t0 = time.time()
    Xc = X.copy()
    
    for sweep in range(max_sweeps):
        if time.time() - t0 > time_budget:
            break
        
        improved = False
        for i in range(N):
            if time.time() - t0 > time_budget:
                break
            
            # Fix all other points, optimize point i on the sphere
            # The energy contribution of point i: sum_{j!=i} log(1/|x_i - x_j|)
            # We want to minimize this over x_i on the unit sphere.
            
            # Use a local optimization: try moving in gradient direction
            E_full, G_full = _energy_and_grad(Xc)
            Gi = G_full[i]
            
            if np.linalg.norm(Gi) < 1e-14:
                continue
            
            # Try several step sizes in the gradient direction
            best_trial = Xc[i].copy()
            best_trial_E = E_full
            
            # Compute energy contribution of point i for comparison
            # Actually, let's just do a mini line search
            for alpha in [0.1, 0.05, 0.02, 0.01, 0.005]:
                X_trial = Xc.copy()
                X_trial[i] = _exponential_map(Xc[i:i+1], alpha * Gi.reshape(1,3))[0]
                E_trial, _ = _energy_and_grad(X_trial)
                if E_trial < best_trial_E:
                    best_trial_E = E_trial
                    best_trial = X_trial[i]
            
            if best_trial_E < E_full - 1e-15:
                Xc[i] = best_trial
                E_full = best_trial_E
                improved = True
        
        if not improved:
            break
        
        if E_full < best_E:
            best = Xc.copy()
            best_E = E_full
    
    return best, best_E


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


def _final_polish(X, time_budget):
    """High-precision final polish with tight tolerances using exponential map CG."""
    t0 = time.time()
    best = X.copy()
    best_E = _log_energy(best)
    
    # Multiple short Riemannian CG runs with exponential map
    for attempt in range(4):
        if time.time() - t0 > time_budget:
            break
        
        Xb, Eb = _riemannian_cg_exp(best, min(2.0, max(0.5, time_budget - (time.time() - t0))))
        if Eb < best_E:
            best, best_E = Xb, Eb
    
    return best, best_E


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN

    # Multiple diverse initializations
    inits = []
    inits.append(_fibonacci_init())
    inits.append(_halton_init(N, seed=1))
    inits.append(_farthest_point_init(N, rng_seed=42, n_candidates=40))
    
    rng = np.random.default_rng(12345)
    inits.append(_project_to_sphere(_fibonacci_init() + 0.05 * rng.standard_normal((N, 3))))
    inits.append(_project_to_sphere(_fibonacci_init() + 0.2 * rng.standard_normal((N, 3))))
    inits.append(_project_to_sphere(rng.standard_normal((N, 3))))
    inits.append(_project_to_sphere(rng.standard_normal((N, 3))))
    
    best = inits[0].copy()
    best_E = _log_energy(best)

    # Phase 1: Riemannian CG with exponential map from each init
    for k, X0 in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        rem = max(1.0, deadline - 15.0 - time.time())
        budget_per = rem / (len(inits) - k)
        Xb, Eb = _riemannian_cg_exp(X0, min(budget_per, 12.0))
        if Eb < best_E:
            best, best_E = Xb, Eb

    # Phase 2: Coordinate descent to escape potential local minima
    if time.time() < deadline - 10.0:
        Xcd, Ecd = _coordinate_descent(best, min(8.0, deadline - 10.0 - time.time()))
        if Ecd < best_E:
            best, best_E = Xcd, Ecd

    # Phase 3: L-BFGS polish
    if time.time() < deadline - 5.0:
        Xp, Ep = _lbfgs_refine(best, min(15.0, deadline - 5.0 - time.time()))
        if Ep < best_E:
            best, best_E = Xp, Ep

    # Phase 4: Final high-precision polish with exponential map CG
    if time.time() < deadline - 2.0:
        Xf, Ef = _final_polish(best, min(5.0, deadline - 2.0 - time.time()))
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
                best[j] = best[j] + 1e-8 * rng.standard_normal(3)
                best[j] /= np.linalg.norm(best[j])
    
    return best
