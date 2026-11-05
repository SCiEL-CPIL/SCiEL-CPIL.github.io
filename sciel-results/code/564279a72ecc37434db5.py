"""QAOA parameter design: graph-dependent two-layer angles for MaxCut."""
import numpy as np
from scipy.optimize import minimize
import time

# Cache for exact MaxCut and graph features
_exact_cache = {}
_features_cache = {}


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    key = (n, tuple(sorted(edges)))
    if key in _exact_cache:
        return _exact_cache[key]
    
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        if cut > best:
            best = cut
    _exact_cache[key] = best
    return best


def _precompute_graph_features(n, edges):
    """Precompute features needed for fast QAOA evaluation."""
    key = (n, tuple(sorted(edges)))
    if key in _features_cache:
        return _features_cache[key]
    
    dim = 1 << n
    m = len(edges)
    
    # Precompute cut values for each mask
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        cut = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                cut += 1
        cut_vals[mask] = cut
    
    # Precompute mixer transition structure
    # H_M = sum_j X_j, so H_M[mask, mask'] = 1 if mask' differs from mask in exactly 1 bit
    # We need to apply exp(-i*beta*H_M) to a state vector
    # H_M has eigenvalues in {-n, -n+2, ..., n}
    # For small n, we can precompute the eigen-decomposition
    
    # Build H_M sparsely: for each mask, it connects to n neighbors (flip each bit)
    # Actually, let's just build the full matrix for n <= 10
    H_M = np.zeros((dim, dim), dtype=np.float64)
    for mask in range(dim):
        for j in range(n):
            H_M[mask, mask ^ (1 << j)] = 1.0
    
    # Eigen-decompose H_M (symmetric real matrix)
    eigvals, eigvecs = np.linalg.eigh(H_M)
    
    _features_cache[key] = {
        'dim': dim,
        'cut_vals': cut_vals,
        'eigvals': eigvals,
        'eigvecs': eigvecs,
        'n': n,
        'm': m
    }
    return _features_cache[key]


def _qaoa_expected_cut(n, edges, gammas, betas, features=None):
    """
    Compute the expected cut value of QAOA with given parameters using
    state-vector simulation with precomputed features.
    """
    if features is None:
        features = _precompute_graph_features(n, edges)
    
    dim = features['dim']
    cut_vals = features['cut_vals']
    eigvals = features['eigvals']
    eigvecs = features['eigvecs']
    
    # Initialize state: |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost unitary: exp(-i*gamma*C) where C is diagonal
        psi = psi * np.exp(-1j * gamma * cut_vals)
        
        # Apply mixer unitary: exp(-i*beta*H_M)
        # Use eigen-decomposition: U = V @ diag(exp(-i*beta*eigvals)) @ V^T
        phases = np.exp(-1j * beta * eigvals)
        # psi' = V @ (phases * (V^T @ psi))
        tmp = eigvecs.T @ psi  # V^T @ psi
        tmp *= phases
        psi = eigvecs @ tmp  # V @ tmp
    
    # Expected cut
    probs = np.abs(psi) ** 2
    expected_cut = np.sum(probs * cut_vals)
    
    return expected_cut


def _qaoa_gradient(n, edges, gammas, betas, features=None):
    """
    Compute gradient of expected cut w.r.t. (gamma_0, beta_0, gamma_1, beta_1)
    using parameter-shift or finite differences. For simplicity, use central
    finite differences with a small step.
    """
    if features is None:
        features = _precompute_graph_features(n, edges)
    
    eps = 1e-6
    grad = np.zeros(4)
    
    # Current value
    val = _qaoa_expected_cut(n, edges, gammas, betas, features)
    
    for i in range(2):
        # gamma_i
        g_plus = gammas.copy()
        g_plus[i] += eps
        val_plus = _qaoa_expected_cut(n, edges, g_plus, betas, features)
        
        g_minus = gammas.copy()
        g_minus[i] -= eps
        val_minus = _qaoa_expected_cut(n, edges, g_minus, betas, features)
        
        grad[i] = (val_plus - val_minus) / (2 * eps)
        
        # beta_i
        b_plus = betas.copy()
        b_plus[i] += eps
        val_plus = _qaoa_expected_cut(n, edges, gammas, b_plus, features)
        
        b_minus = betas.copy()
        b_minus[i] -= eps
        val_minus = _qaoa_expected_cut(n, edges, gammas, b_minus, features)
        
        grad[2 + i] = (val_plus - val_minus) / (2 * eps)
    
    return grad


def _objective(params, n, edges, exact_opt, features):
    """Objective function to minimize (negative of ratio)."""
    gammas = params[:2]
    betas = params[2:]
    
    try:
        exp_cut = _qaoa_expected_cut(n, edges, gammas, betas, features)
    except Exception:
        return 1e6
    
    ratio = exp_cut / exact_opt if exact_opt > 0 else 0
    return -ratio


def _optimize_for_graph(n, edges, p=2, deadline=None):
    """Optimize QAOA parameters for a specific graph."""
    exact_opt = _exact_maxcut(n, edges)
    features = _precompute_graph_features(n, edges)
    
    # Initial guesses based on known good QAOA values
    # For p=2, common good starting points
    candidates = [
        np.array([0.35, 0.35, 0.25, 0.25]),
        np.array([0.5, 0.4, 0.3, 0.2]),
        np.array([0.25, 0.25, 0.35, 0.35]),
        np.array([0.4, 0.3, 0.2, 0.3]),
        np.array([0.3, 0.3, 0.3, 0.3]),
    ]
    
    best_params = candidates[0].copy()
    best_score = _objective(candidates[0], n, edges, exact_opt, features)
    
    # Multi-start L-BFGS-B optimization
    n_starts = 8
    for i in range(n_starts):
        if deadline is not None and time.time() > deadline:
            break
        
        if i < len(candidates):
            x_start = candidates[i].copy()
        else:
            # Random initialization
            x_start = np.random.uniform(-np.pi/2, np.pi/2, size=4)
        
        try:
            result = minimize(
                _objective,
                x_start,
                args=(n, edges, exact_opt, features),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * 4,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    gammas = best_params[:2]
    betas = best_params[2:]
    
    return np.array([[gammas[0], betas[0]], [gammas[1], betas[1]]])


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of qubits/nodes
        edges: list of edges (u, v)
        p: number of QAOA layers (should be 2)
    
    Returns:
        Array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        result = _optimize_for_graph(n, edges, p=2, deadline=deadline)
        return result
    except Exception:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
