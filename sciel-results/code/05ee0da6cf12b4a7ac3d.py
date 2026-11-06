"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

# Cache for optimal angles per graph
_angle_cache = {}
_T0 = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 15.0

def _get_remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _T0)

def _build_c_diagonal(n, edges):
    """
    Compute the diagonal values of the MaxCut Hamiltonian C.
    C(b) = number of edges crossing the cut defined by b.
    """
    diag = np.zeros(2**n)
    for mask in range(2**n):
        cut_val = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut_val += 1
        diag[mask] = cut_val
    return diag

def _exact_optimum(n, edges):
    """Compute the exact maximum cut value."""
    best = 0
    for mask in range(2**n):
        cut_val = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut_val += 1
        if cut_val > best:
            best = cut_val
    return best

def _apply_mixer(state, beta, n):
    """
    Apply exp(-i beta sum X) to state.
    sum X = sum_j X_j, which is a product of single-qubit gates.
    Gate on each qubit: [[cos(beta), -i*sin(beta)], [-i*sin(beta), cos(beta)]]
    """
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    
    # Reshape to tensor
    state_tensor = state.reshape([2] * n)
    
    def apply_gate_to_axis(state, axis, gate):
        state = np.moveaxis(state, axis, 0)
        orig_shape = state.shape
        state = state.reshape(-1, 2)
        state = gate @ state
        state = state.reshape(orig_shape)
        state = np.moveaxis(state, 0, axis)
        return state
    
    gate = np.array([[c, s], [s, c]], dtype=complex)
    for j in range(n):
        state_tensor = apply_gate_to_axis(state_tensor, j, gate)
        
    return state_tensor.reshape(-1)

def _apply_cost(state, gamma, c_diag):
    """Apply exp(-i gamma C). C is diagonal."""
    phase = np.exp(-1j * gamma * c_diag)
    return state * phase

def _expected_cut_value(n, edges, gammas, betas, c_diag=None):
    """
    Compute expected cut for given parameters.
    Order: Start |+>^n, for each k: apply exp(-i gamma[k] C), then exp(-i beta[k] sum X)
    """
    p = len(gammas)
    if c_diag is None:
        c_diag = _build_c_diagonal(n, edges)
    
    # Initial state |+>^n
    state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    
    for k in range(p):
        # Apply Cost
        state = _apply_cost(state, gammas[k], c_diag)
        # Apply Mixer
        state = _apply_mixer(state, betas[k], n)
        
    # Compute expectation <psi| C |psi>
    prob = np.abs(state)**2
    return np.dot(prob, c_diag)

def _optimize_angles(n, edges, p):
    """
    Optimize angles for given graph.
    """
    c_diag = _build_c_diagonal(n, edges)
    
    # Initial guesses - diverse starting points
    starts = [
        [0.35, 0.25] * p,
        [0.5, 0.2] * p,
        [1.0, 0.5] * p,
        [np.pi/4, np.pi/8] * p,
        [0.2, 0.4] * p,
        [0.8, 0.3] * p,
        [0.6, 0.15] * p,
        [np.pi/6, np.pi/12] * p,
        [0.4, 0.3] * p,
        [0.7, 0.25] * p,
    ]
    
    # Add some random starts
    rng = np.random.RandomState(42)
    for _ in range(4):
        starts.append(rng.uniform(0, np.pi, 2*p).tolist())
    
    best_val = -np.inf
    best_params = None
    
    # Objective function to minimize (negative expected cut)
    def obj(params):
        gammas = params[:p]
        betas = params[p:]
        val = _expected_cut_value(n, edges, gammas, betas, c_diag)
        return -val
    
    # Bounds
    bounds = [(0, np.pi)] * (2 * p)
    
    for start in starts:
        # Check time
        if _get_remaining_time() < 8.0:
            break
            
        try:
            res = minimize(obj, start, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to seed
        return np.tile([0.35, 0.25], (p, 1))
        
    # Extract gammas and betas
    gammas = best_params[:p]
    betas = best_params[p:]
    
    # Format output: (p, 2) array, rows (gamma, beta)
    angles = np.column_stack((gammas, betas))
    return angles

def angles(n, edges, p):
    # Create a cache key
    edges_tuple = tuple(sorted((u, v) if u < v else (v, u) for u, v in edges))
    key = (n, edges_tuple, p)
    
    if key in _angle_cache:
        return _angle_cache[key]
        
    # Check time
    if _get_remaining_time() < 2.0:
        # Not enough time to optimize, return default
        res = np.tile([0.35, 0.25], (p, 1))
        return res
        
    res = _optimize_angles(n, edges, p)
    
    # Store in cache
    _angle_cache[key] = res
    
    return res
# EVOLVE-BLOCK-END
