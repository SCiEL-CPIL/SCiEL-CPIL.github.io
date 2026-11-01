"""Optimized logarithmic energy minimization for 150 points on the unit sphere."""
import numpy as np
import math
import time

N = 150


def _log_energy(X):
    """Compute sum_{i<j} log(1/|xi - xj|) using math.fsum."""
    n = len(X)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-15:
                return float('inf')
            terms.append(-math.log(d))
    return math.fsum(terms)


def _log_energy_fast(X):
    """Vectorized log energy for optimization (not the official scoring metric)."""
    n = len(X)
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    # Avoid log(0)
    dist = np.maximum(dist, 1e-15)
    # Upper triangle only
    idx = np.triu_indices(n, k=1)
    return np.sum(-np.log(dist[idx]))


def _log_energy_grad(X):
    """Gradient of log energy: grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2 * (-1)
    Actually, E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    dE/dxi = sum_{j!=i} (xi - xj)/|xi-xj|^2
    """
    n = len(X)
    grad = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = X[i] - X[j]
            dist_sq = np.dot(d, d)
            if dist_sq < 1e-30:
                continue
            grad[i] += d / dist_sq
    return grad


def _project_to_sphere(X):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_initialization(n=N):
    """Golden-angle Fibonacci spiral initialization."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_initialization(n=N, seed=42):
    """Random initialization on sphere."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    best_X = None
    best_energy = float('inf')
    
    # Try multiple initializations
    inits = []
    inits.append(_fibonacci_initialization())
    for seed in [1, 7, 42, 123, 456]:
        inits.append(_random_initialization(seed=seed))
    
    for init_idx, X0 in enumerate(inits):
        if time.time() > deadline:
            break
        
        current_X = X0.copy()
        current_energy = _log_energy_fast(current_X)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_X = current_X.copy()
        
        # Local optimization with L-BFGS-B on flattened, then project
        def objective_and_grad(flat):
            X = flat.reshape(-1, 3)
            X = _project_to_sphere(X)
            e = _log_energy_fast(X)
            g = _log_energy_grad(X)
            # Project gradient onto tangent space
            for i in range(len(X)):
                g[i] -= np.dot(g[i], X[i]) * X[i]
            return e, g.ravel()
        
        # Wrap for scipy
        def obj(flat):
            return objective_and_grad(flat)[0]
        
        def grad(flat):
            return objective_and_grad(flat)[1]
        
        try:
            from scipy.optimize import minimize
            res = minimize(obj, current_X.ravel(), jac=grad, method='L-BFGS-B',
                           options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-14})
            if time.time() > deadline:
                break
            X_opt = _project_to_sphere(res.x.reshape(-1, 3))
            e_opt = _log_energy_fast(X_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_X = X_opt.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances on best result
    if best_X is not None and time.time() < deadline:
        try:
            from scipy.optimize import minimize
            def obj(flat):
                X = _project_to_sphere(flat.reshape(-1, 3))
                return _log_energy_fast(X)
            
            def grad(flat):
                X = _project_to_sphere(flat.reshape(-1, 3))
                g = _log_energy_grad(X)
                for i in range(len(X)):
                    g[i] -= np.dot(g[i], X[i]) * X[i]
                return g.ravel()
            
            res = minimize(obj, best_X.ravel(), jac=grad, method='L-BFGS-B',
                           options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-16})
            X_final = _project_to_sphere(res.x.reshape(-1, 3))
            e_final = _log_energy_fast(X_final)
            if e_final < best_energy:
                best_X = X_final
        except Exception:
            pass
    
    # Final normalization
    best_X = _project_to_sphere(best_X)
    
    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                # Nudge apart
                best_X[j] += 1e-8 * np.random.randn(3)
                best_X = _project_to_sphere(best_X)
                break
    
    return best_X
