"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

# Deterministic global RNG so construct() is reproducible within a run.
_RNG = np.random.default_rng(12345)


def _pairwise_chordal_sq(pts):
    """Return (N, N) matrix of squared chordal distances."""
    # ||a - b||^2 = 1 + 1 - 2 a.b = 2 - 2 a.b
    G = pts @ pts.T
    return np.maximum(2.0 - 2.0 * G, 0.0)


def _log_energy(pts):
    """Compute log energy sum_{i<j} log(1/d_ij). Vectorized, numerically safe."""
    D2 = _pairwise_chordal_sq(pts)
    # Use upper triangle excluding diagonal
    iu = np.triu_indices(N, k=1)
    d2 = D2[iu]
    # clip to avoid log(0) if two points coincide (shouldn't happen after normalize)
    d2 = np.maximum(d2, 1e-30)
    # log(1/d) = -0.5 * log(d^2)
    return float(np.sum(-0.5 * np.log(d2)))


def _log_energy_grad(pts):
    """Compute energy and gradient w.r.t. pts (N,3)."""
    D2 = _pairwise_chordal_sq(pts)
    iu = np.triu_indices(N, k=1)
    d2 = D2[iu]
    d2 = np.maximum(d2, 1e-30)
    # energy = sum -0.5 log(d2)
    E = float(np.sum(-0.5 * np.log(d2)))
    # grad: for pair (i,j), term = -0.5 log(||x_i - x_j||^2)
    # d/dx_i term = -0.5 * (1/d2) * d(d2)/dx_i = -0.5 * (1/d2) * (-2 (x_i - x_j)) = (x_i - x_j)/d2
    # d/dx_j term = (x_j - x_i)/d2
    # So total grad for x_i = sum_{j != i} (x_i - x_j)/d2_ij
    # We can build via matrix ops:
    # Let S = (x_i - x_j)/d2_ij for i<j, then grad_i += S, grad_j -= S
    # Vectorized: compute full matrix M where M[i,j] = (x_i - x_j)/d2_ij for i != j
    # M[i,j] = (pts[i] - pts[j]) / D2[i,j]
    # But D2[i,j] = 0 on diagonal; we'll set diagonal to 1 to avoid div0.
    D2_safe = np.where(np.eye(N) > 0, 1.0, D2)
    # M[i,j] = (pts[i] - pts[j]) / D2_safe[i,j]
    # (pts[i] - pts[j]) = pts[i] * ones - pts[j] * ones
    # M = (pts[:, None, :] - pts[None, :, :]) / D2_safe[:, :, None]
    M = (pts[:, None, :] - pts[None, :, :]) / D2_safe[:, :, None]
    # Set diagonal to 0
    M[np.diag_indices(N)] = 0.0
    # grad_i = sum_j M[i,j]
    grad = np.sum(M, axis=1)
    return E, grad


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _project_to_sphere(pts):
    """Project points onto unit sphere and return (pts, energy)."""
    pts = _normalize(pts)
    E, _ = _log_energy_grad(pts)
    return pts, E


def _lbfgs_step(pts, max_iter=200, time_limit=30.0):
    """Run L-BFGS on the sphere using a retraction-based approach.
    For simplicity, we optimize in R^3 and normalize at each evaluation."""
    start_time = time.time()
    
    def fun_and_grad(x):
        pts = _normalize(x)
        E, g = _log_energy_grad(pts)
        # Project gradient back to ambient space (approximate)
        # Since we normalize inside, the gradient w.r.t. x is not exactly g,
        # but for small deviations it's close. A better approach uses the
        # Riemannian gradient, but we'll use the Euclidean gradient for simplicity
        # and rely on the normalization to keep points on the sphere.
        return E, g
    
    try:
        res = minimize(fun_and_grad, pts, method='L-BFGS-B', 
                       jac=True, options={'maxiter': max_iter, 'maxfun': 10000})
        if res.success or res.fun < _lbfgs_step._best_E:
            _lbfgs_step._best_E = res.fun
            _lbfgs_step._best_pts = _normalize(res.x)
    except Exception:
        pass
    return _lbfgs_step._best_pts

# Initialize class-level best
_lbfgs_step._best_E = float('inf')
_lbfgs_step._best_pts = None


def _simulated_annealing(pts, max_iter=5000, T_start=0.1, T_end=0.001, time_limit=30.0):
    """Simulated annealing with Gaussian perturbations on the sphere."""
    start_time = time.time()
    best_pts = pts.copy()
    best_E = _log_energy(pts)
    current_E = best_E
    T = T_start
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        # Generate perturbation
        delta = _RNG.normal(0, 1, size=pts.shape)
        # Scale perturbation by temperature
        new_pts = pts + T * delta
        new_pts = _normalize(new_pts)
        
        new_E = _log_energy(new_pts)
        
        # Accept if better, or with probability exp(-(new_E - current_E)/T)
        if new_E < current_E or _RNG.random() < math.exp(-(new_E - current_E) / max(T, 1e-12)):
            pts = new_pts
            current_E = new_E
            if current_E < best_E:
                best_E = current_E
                best_pts = pts.copy()
        
        # Cool down
        T = max(T * 0.999, T_end)
    
    return best_pts


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    global _RNG
    _RNG = np.random.default_rng(12345)
    
    deadline = time.time() + 140.0  # Safety margin: 10s before 150s limit
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts0 = _normalize(pts0)
    
    best_pts = pts0.copy()
    best_E = _log_energy(pts0)
    
    # Phase 1: L-BFGS from Fibonacci start
    _lbfgs_step._best_E = float('inf')
    _lbfgs_step._best_pts = None
    if time.time() < deadline - 30.0:
        pts_lbfgs = _lbfgs_step(pts0, max_iter=300, time_limit=30.0)
        if pts_lbfgs is not None:
            E_lbfgs = _log_energy(pts_lbfgs)
            if E_lbfgs < best_E:
                best_E = E_lbfgs
                best_pts = pts_lbfgs
    
    # Phase 2: Multi-restart with random initializations + SA
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Random initial points
        pts_rand = _RNG.normal(0, 1, size=(N, 3))
        pts_rand = _normalize(pts_rand)
        
        # Short L-BFGS to get to a local minimum
        _lbfgs_step._best_E = float('inf')
        _lbfgs_step._best_pts = None
        if time.time() < deadline - 15.0:
            pts_opt = _lbfgs_step(pts_rand, max_iter=100, time_limit=10.0)
            if pts_opt is not None:
                E_opt = _log_energy(pts_opt)
                if E_opt < best_E:
                    best_E = E_opt
                    best_pts = pts_opt.copy()
        
        # Simulated annealing from the best point found so far
        if time.time() < deadline - 10.0:
            pts_sa = _simulated_annealing(best_pts, max_iter=2000, 
                                         T_start=0.05, T_end=0.001, 
                                         time_limit=10.0)
            E_sa = _log_energy(pts_sa)
            if E_sa < best_E:
                best_E = E_sa
                best_pts = pts_sa
    
    # Final polish with L-BFGS
    if time.time() < deadline - 5.0:
        _lbfgs_step._best_E = float('inf')
        _lbfgs_step._best_pts = None
        pts_final = _lbfgs_step(best_pts, max_iter=200, time_limit=10.0)
        if pts_final is not None:
            E_final = _log_energy(pts_final)
            if E_final < best_E:
                best_E = E_final
                best_pts = pts_final
    
    # Ensure output is valid: normalize, check finiteness
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci spiral
        best_pts = pts0
    
    return best_pts
# EVOLVE-BLOCK-END
