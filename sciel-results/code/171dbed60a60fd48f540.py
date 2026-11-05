"""QAOA angle design for fixed-depth p=2 MaxCut on small graphs.

Vectorized state-vector approach:
- Full state vector simulation with vectorized mixer application
- Gradient-based optimization with many restarts and polish pass
- For n<=10, dim=2^n <= 1024, which is manageable
- Key: vectorize the mixer to reduce per-evaluation cost significantly
"""
import numpy as np
import time
import scipy.optimize

_start_time = time.time()
_DEADLINE = 142.0  # safety margin before 150s kill


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    dim = 1 << n
    best = 0
    edge_list = list(edges)
    for mask in range(dim):
        cut = 0
        for u, v in edge_list:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best


def _precompute_cut_values(n, edges):
    """Precompute cut values for all 2^n assignments."""
    dim = 1 << n
    edge_list = list(edges)
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cv = 0
        for u, v in edge_list:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cv += 1
        cut_vals[mask] = cv
    return cut_vals


def _apply_mixer_fast(psi, n, beta):
    """Apply exp(-i*beta*sum X_i) using vectorized operations."""
    cb = np.cos(beta)
    sb = np.sin(beta)
    dim = 1 << n
    for i in range(n):
        bit = 1 << i
        # Create index arrays for pairs differing in bit i
        idx0 = np.where((np.arange(dim) & bit) == 0)[0]
        idx1 = idx0 | bit
        a = psi[idx0].copy()
        b = psi[idx1].copy()
        psi[idx0] = cb * a - 1j * sb * b
        psi[idx1] = -1j * sb * a + cb * b
    return psi


def _qaoa_state(n, cut_vals, g0, b0, g1, b1):
    """Compute the QAOA state vector for p=2."""
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    psi *= np.exp(-1j * g0 * cut_vals)
    psi = _apply_mixer_fast(psi, n, b0)
    psi *= np.exp(-1j * g1 * cut_vals)
    psi = _apply_mixer_fast(psi, n, b1)
    return psi


def _qaoa_expectation(n, cut_vals, g0, b0, g1, b1):
    """Compute expected cut value for given QAOA parameters."""
    psi = _qaoa_state(0, cut_vals, g0, b0, g1, b1)
    probs = np.abs(psi) ** 2
    return np.dot(probs, cut_vals)


def _qaoa_gradient(n, cut_vals, g0, b0, g1, b1, eps=1e-5):
    """Finite difference gradient of expected cut."""
    def f(params):
        return _qaoa_expectation(n, cut_vals, params[0], params[1], params[2], params[3])
    
    params = np.array([g0, b0, g1, b1])
    grad = np.zeros(4)
    for i in range(4):
        params_p = params.copy()
        params_m = params.copy()
        params_p[i] += eps
        params_m[i] -= eps
        grad[i] = (f(params_p) - f(params_m)) / (2 * eps)
    return grad


def angles(n, edges, p):
    global _start_time, _DEADLINE
    start_time = time.time()
    
    if n > 12:
        # Fallback for large n
        return np.array([[0.4, 0.35], [0.4, 0.35]])
    
    # Precompute cut values once
    dim = 1 << n
    edge_list = list(edges)
    
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cv = 0
        for u, v in edge_list:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cv += 1
        cut_vals[mask] = cv
    
    # Precompute index arrays for mixer (done once, reused in every eval)
    mixer_indices = []
    for i in range(n):
        bit = 1 << i
        idx0 = np.where((np.arange(dim) & bit) == 0)[0]
        idx1 = idx0 | bit
        mixer_indices.append((idx0, idx1))
    
    def eval_fn(params):
        """Evaluate expected cut for given parameters [g0, b0, g1, b1]."""
        g0, b0, g1, b1 = params
        psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        psi *= np.exp(-1j * g0 * cut_vals)
        
        # Mixer 1
        cb = np.cos(b0)
        sb = np.sin(b0)
        for idx0, idx1 in mixer_indices:
            a = psi[idx0].copy()
            b = psi[idx1].copy()
            psi[idx0] = cb * a - 1j * sb * b
            psi[idx1] = -1j * sb * a + cb * b
        
        psi *= np.exp(-1j * g1 * cut_vals)
        
        # Mixer 2
        cb = np.cos(b1)
        sb = np.sin(b1)
        for idx0, idx1 in mixer_indices:
            a = psi[idx0].copy()
            b = psi[idx1].copy()
            psi[idx0] = cb * a - 1j * sb * b
            psi[idx1] = -1j * sb * a + cb * b
        
        probs = np.abs(psi) ** 2
        return np.dot(probs, cut_vals)
    
    def objective(params):
        return -eval_fn(params)
    
    def gradient(params):
        eps = 1e-5
        g0, b0, g1, b1 = params
        grad = np.zeros(4)
        grad[0] = (eval_fn([g0+eps, b0, g1, b1]) - eval_fn([g0-eps, b0, g1, b1])) / (2*eps)
        grad[1] = (eval_fn([g0, b0+eps, g1, b1]) - eval_fn([g0, b0-eps, g1, b1])) / (2*eps)
        grad[2] = (eval_fn([g0, b0, g1+eps, b1]) - eval_fn([g0, b0, g1-eps, b1])) / (2*eps)
        grad[3] = (eval_fn([g0, b0, g1, b1+eps]) - eval_fn([g0, b0, g1, b1-eps])) / (2*eps)
        return -grad  # negate since we're minimizing negative
    
    # Try multiple initial points - expanded set
    inits = [
        [0.5, 0.5, 0.3, 0.3],
        [0.7, 0.7, 0.4, 0.4],
        [0.3, 0.4, 0.2, 0.3],
        [0.5, 0.3, 0.5, 0.3],
        [0.4, 0.4, 0.4, 0.4],
        [0.6, 0.2, 0.3, 0.5],
        [0.2, 0.2, 0.2, 0.2],
        [0.8, 0.6, 0.5, 0.4],
        [0.3, 0.6, 0.3, 0.6],
        [0.6, 0.4, 0.6, 0.4],
    ]
    
    best_score = -np.inf
    best_params = np.array([0.4, 0.35, 0.4, 0.35])
    
    np.random.seed(123)
    
    # Add more random inits
    for _ in range(25):
        inits.append(np.random.uniform([0.05, 0.05, 0.05, 0.05], 
                                       [1.5, 1.5, 1.5, 1.5]).tolist())
    
    bounds = [(0.0, np.pi)] * 4
    
    # Phase 1: Broad search with L-BFGS-B
    for init in inits:
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        try:
            res = scipy.optimize.minimize(
                objective, np.array(init, dtype=np.float64),
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Polish pass with tighter tolerances from best solution
    if time.time() - start_time < _DEADLINE - 5:
        try:
            res = scipy.optimize.minimize(
                objective, best_params.copy(),
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array([[best_params[0], best_params[1]],
                     [best_params[2], best_params[3]]])
