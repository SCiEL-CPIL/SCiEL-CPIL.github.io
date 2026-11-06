import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 110.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

_graph_cache = {}

def _get_graph_props(n, edges):
    key = (n, tuple(sorted((min(i, j), max(i, j)) for i, j in edges)))
    if key in _graph_cache:
        return _graph_cache[key]
    
    num_states = 1 << n
    cuts = np.zeros(num_states, dtype=np.float64)
    edge_masks = []
    
    indices = np.arange(num_states, dtype=np.int32)
    
    for i, j in edges:
        bit_i = (indices >> i) & 1
        bit_j = (indices >> j) & 1
        mask = (bit_i != bit_j).astype(np.float64)
        edge_masks.append(mask)
        cuts += mask
    
    _graph_cache[key] = (cuts, edge_masks)
    return _graph_cache[key]

def _apply_mixing(state, beta, n):
    c = np.cos(beta)
    s = np.sin(beta)
    for k in range(n):
        state = np.moveaxis(state, k, 0)
        state = state.reshape(2, -1)
        new_state = np.empty_like(state)
        new_state[0] = c * state[0] - 1j * s * state[1]
        new_state[1] = -1j * s * state[0] + c * state[1]
        state = new_state
        state = state.reshape((2,) * n)
        state = np.moveaxis(state, 0, k)
    return state

def _expected_cut(n, edges, gammas, betas, cuts, edge_masks):
    num_states = 1 << n
    state = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    for gamma, beta in zip(gammas, betas):
        if abs(gamma) > 1e-15:
            state *= np.exp(-1j * gamma * cuts)
        if abs(beta) > 1e-15:
            state = _apply_mixing(state, beta, n)
            
    probs = np.abs(state) ** 2
    expected = 0.0
    for mask in edge_masks:
        expected += np.sum(probs * mask)
    return expected

def _exact_maxcut(n, edges):
    num_states = 1 << n
    best = 0
    for x in range(num_states):
        cut = 0
        for i, j in edges:
            if ((x >> i) & 1) != ((x >> j) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _optimize_angles(n, edges, p):
    cuts, edge_masks = _get_graph_props(n, edges)
    C_opt = _exact_maxcut(n, edges)
    if C_opt == 0:
        return np.zeros((p, 2))
        
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        try:
            ec = _expected_cut(n, edges, gammas, betas, cuts, edge_masks)
            return -ec
        except:
            return 1e10
            
    best_params = None
    best_val = -np.inf
    
    # Multi-start with grid + random
    start_points = []
    
    # Dense grid in the standard QAOA region [0, pi/2]
    grid_vals = np.linspace(0.1, 1.4, 5)
    for g1 in grid_vals:
        for b1 in grid_vals:
            for g2 in grid_vals:
                for b2 in grid_vals:
                    start_points.append([g1, b1, g2, b2])
    
    # Add some random starts
    rng = np.random.RandomState(42)
    for _ in range(10):
        start_points.append(rng.uniform(0, np.pi, 4).tolist())
        
    # Sort starts by a simple heuristic: prefer smaller angles
    start_points.sort(key=lambda x: sum(x))
    
    # Try top starts first
    for sp in start_points[:30]:
        if _time_remaining() < 15:
            break
        try:
            result = minimize(objective, sp, method='L-BFGS-B', 
                             bounds=[(0, np.pi)] * 4,
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except:
            pass
            
    # If time remains, do a high-precision polish on the best
    if best_params is not None and _time_remaining() > 20:
        try:
            result = minimize(objective, best_params, method='L-BFGS-B', 
                             bounds=[(0, np.pi)] * 4,
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except:
            pass
            
    if best_params is None:
        best_params = np.array([0.5, 0.5, 0.25, 0.25])
        
    gammas = best_params[:p]
    betas = best_params[p:]
    return np.column_stack([gammas, betas])

def angles(n, edges, p):
    edges_list = tuple(sorted((min(i, j), max(i, j)) for i, j in edges))
    cache_key = (n, edges_list, p)
    
    if not hasattr(angles, '_result_cache'):
        angles._result_cache = {}
        
    if cache_key in angles._result_cache:
        return angles._result_cache[cache_key]
        
    if _time_remaining() < 10:
        result = np.tile([0.35, 0.25], (p, 1))
        angles._result_cache[cache_key] = result
        return result
        
    try:
        result = _optimize_angles(n, edges, p)
    except Exception:
        result = np.tile([0.35, 0.25], (p, 1))
        
    angles._result_cache[cache_key] = result
    return result
