import numpy as np
import time
from scipy.optimize import minimize

# Global cache for precomputed graph data
_graph_cache = {}

def _simulate_qaoa_fast(n, edges, params, cut_sizes, flip_indices):
    """
    Simulate QAOA for given graph and parameters.
    params: array of shape (p, 2) with (gamma, beta).
    Returns expected cut size.
    """
    p = len(params)
    N = 1 << n
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    
    for gamma, beta in params:
        # Apply Cost
        state *= np.exp(-1j * gamma * cut_sizes)
        
        # Apply Mixer
        c = np.cos(beta)
        s = -1j * np.sin(beta)
        
        for i in range(n):
            # Use precomputed flip indices
            state = c * state + s * state[flip_indices[i]]
            
    probs = np.abs(state)**2
    return np.sum(probs * cut_sizes)

def _expected_cut_from_params(params_flat, n, cut_sizes, flip_indices):
    """
    Compute expected cut from flat parameter array [gamma1, beta1, gamma2, beta2].
    """
    p = len(params_flat) // 2
    params = params_flat.reshape(p, 2)
    return _simulate_qaoa_fast(n, None, params, cut_sizes, flip_indices)

def _batch_evaluate(params_array, n, cut_sizes, flip_indices):
    """
    Evaluate multiple parameter sets. params_array: (num_params, 4) for p=2.
    Returns array of expected cut values.
    """
    num = params_array.shape[0]
    results = np.zeros(num, dtype=float)
    N = 1 << n
    
    for idx in range(num):
        params = params_array[idx].reshape(2, 2)
        state = np.ones(N, dtype=complex) / np.sqrt(N)
        
        for gamma, beta in params:
            state *= np.exp(-1j * gamma * cut_sizes)
            c = np.cos(beta)
            s = -1j * np.sin(beta)
            for i in range(n):
                state = c * state + s * state[flip_indices[i]]
        
        probs = np.abs(state)**2
        results[idx] = np.sum(probs * cut_sizes)
    
    return results

def angles(n, edges, p):
    n = int(n)
    # Normalize edges to list of tuples to ensure hashability and consistent sorting
    if isinstance(edges, np.ndarray):
        edges = [tuple(e) for e in edges]
    else:
        edges = [tuple(e) for e in edges]

    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    # Get or compute graph data
    edge_key = (n, tuple(sorted(edges)))
    if edge_key in _graph_cache:
        data = _graph_cache[edge_key]
    else:
        # Compute cut sizes
        N = 1 << n
        cut_sizes = np.zeros(N, dtype=float)
        states = np.arange(N, dtype=np.int64)
        for u, v in edges:
            bits_u = (states >> u) & 1
            bits_v = (states >> v) & 1
            cut_sizes += (bits_u != bits_v).astype(float)
        
        max_cut = float(np.max(cut_sizes))
        flip_indices = [np.arange(N) ^ (1 << i) for i in range(n)]
        
        data = {
            'cut_sizes': cut_sizes,
            'flip_indices': flip_indices,
            'max_cut': max_cut,
            'n': n
        }
        _graph_cache[edge_key] = data
    
    cut_sizes = data['cut_sizes']
    flip_indices = data['flip_indices']
    max_cut = float(data['max_cut'])
    
    if max_cut == 0.0:
        return np.zeros((p, 2))
    
    # Define objective function
    def local_obj(params):
        exp_cut = _expected_cut_from_params(params, n, cut_sizes, flip_indices)
        return float(-exp_cut / max_cut)
    
    best_params = None
    best_score = -np.inf
    
    def _update_best(params, score):
        nonlocal best_params, best_score
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # Phase 1: Coarse global grid search
    # 10 points per dimension = 10000 evaluations
    # For n=10, each eval ~0.01ms, so 10000 * 0.01ms = 100ms
    if time.time() - start_time < deadline - 60.0:
        grid_points = 10
        g_vals = np.linspace(0.05, np.pi - 0.05, grid_points)
        b_vals = np.linspace(0.05, np.pi/2 - 0.05, grid_points)
        
        # Build all combinations: (g1, b1, g2, b2)
        g1_grid, b1_grid, g2_grid, b2_grid = np.meshgrid(
            g_vals, b_vals, g_vals, b_vals, indexing='ij'
        )
        all_params = np.column_stack([
            g1_grid.ravel(), b1_grid.ravel(),
            g2_grid.ravel(), b2_grid.ravel()
        ])
        
        # Evaluate in batches
        batch_size = 200
        for i in range(0, len(all_params), batch_size):
            if time.time() - start_time > deadline - 55.0:
                break
            batch = all_params[i:i+batch_size]
            results = _batch_evaluate(batch, n, cut_sizes, flip_indices)
            scores = results / max_cut
            best_idx = np.argmax(scores)
            _update_best(batch[best_idx], float(scores[best_idx]))
            
            # Also track top 5 for next phase
            top_indices = np.argsort(scores)[-5:]
            for idx in top_indices:
                _update_best(batch[idx], float(scores[idx]))
    
    # Phase 2: Medium-resolution grid around best region
    if best_params is not None and time.time() - start_time < deadline - 30.0:
        center = best_params
        # Search in a box of size 0.4 around center
        delta = 0.4
        g1_range = np.linspace(max(0.01, center[0] - delta), min(np.pi - 0.01, center[0] + delta), 12)
        b1_range = np.linspace(max(0.01, center[1] - delta), min(np.pi/2 - 0.01, center[1] + delta), 12)
        g2_range = np.linspace(max(0.01, center[2] - delta), min(np.pi - 0.01, center[2] + delta), 12)
        b2_range = np.linspace(max(0.01, center[3] - delta), min(np.pi/2 - 0.01, center[3] + delta), 12)
        
        g1_grid, b1_grid, g2_grid, b2_grid = np.meshgrid(
            g1_range, b1_range, g2_range, b2_range, indexing='ij'
        )
        all_params = np.column_stack([
            g1_grid.ravel(), b1_grid.ravel(),
            g2_grid.ravel(), b2_grid.ravel()
        ])
        
        batch_size = 200
        for i in range(0, len(all_params), batch_size):
            if time.time() - start_time > deadline - 25.0:
                break
            batch = all_params[i:i+batch_size]
            results = _batch_evaluate(batch, n, cut_sizes, flip_indices)
            scores = results / max_cut
            best_idx = np.argmax(scores)
            _update_best(batch[best_idx], float(scores[best_idx]))
    
    # Phase 3: Fine grid around best
    if best_params is not None and time.time() - start_time < deadline - 15.0:
        center = best_params
        delta = 0.15
        g1_range = np.linspace(max(0.01, center[0] - delta), min(np.pi - 0.01, center[0] + delta), 10)
        b1_range = np.linspace(max(0.01, center[1] - delta), min(np.pi/2 - 0.01, center[1] + delta), 10)
        g2_range = np.linspace(max(0.01, center[2] - delta), min(np.pi - 0.01, center[2] + delta), 10)
        b2_range = np.linspace(max(0.01, center[3] - delta), min(np.pi/2 - 0.01, center[3] + delta), 10)
        
        g1_grid, b1_grid, g2_grid, b2_grid = np.meshgrid(
            g1_range, b1_range, g2_range, b2_range, indexing='ij'
        )
        all_params = np.column_stack([
            g1_grid.ravel(), b1_grid.ravel(),
            g2_grid.ravel(), b2_grid.ravel()
        ])
        
        batch_size = 100
        for i in range(0, len(all_params), batch_size):
            if time.time() - start_time > deadline - 10.0:
                break
            batch = all_params[i:i+batch_size]
            results = _batch_evaluate(batch, n, cut_sizes, flip_indices)
            scores = results / max_cut
            best_idx = np.argmax(scores)
            _update_best(batch[best_idx], float(scores[best_idx]))
    
    # Phase 4: Nelder-Mead polish from best grid point
    if best_params is not None and time.time() - start_time < deadline - 5.0:
        try:
            res = minimize(
                local_obj, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            score = float(-res.fun)
            _update_best(res.x, score)
        except Exception:
            pass
    
    # Phase 5: Additional Nelder-Mead from slightly perturbed best
    if best_params is not None and time.time() - start_time < deadline - 2.0:
        for perturb in [0.05, -0.05, 0.1, -0.1]:
            if time.time() - start_time > deadline - 2.0:
                break
            try:
                x0 = best_params.copy()
                x0[np.random.RandomState(42).randint(0, 4)] += perturb
                x0 = np.clip(x0, 0.01, np.pi - 0.01)
                res = minimize(
                    local_obj, x0, method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                score = float(-res.fun)
                _update_best(res.x, score)
            except Exception:
                continue
    
    if best_params is None:
        return np.array([[0.5, 0.5], [0.5, 0.5]])
    
    return np.array(best_params).reshape(p, 2)
