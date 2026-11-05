"""QAOA parameter design for fixed-depth (p=2) MaxCut."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 140.0  # seconds, leave 10s safety margin for 150s limit

def _remaining_time():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value via brute force for small n."""
    m = len(edges)
    if m == 0:
        return 0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edges:
            if (mask >> u) & 1 != (mask >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best

def _expected_cut_state_vector(n, edges, gammas, betas):
    """Exact state vector simulation for small n (n <= 12)."""
    dim = 1 << n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute Z operators: Z_i |j> = (+1 if bit i is 0, -1 if bit i is 1) |j>
    Z = np.zeros((n, dim), dtype=complex)
    for i in range(n):
        for j in range(dim):
            Z[i, j] = 1.0 if (j >> i) & 1 == 0 else -1.0
    
    # Precompute ZZ products for each edge
    m = len(edges)
    ZZ = np.zeros((m, dim), dtype=complex)
    for k, (u, v) in enumerate(edges):
        ZZ[k] = Z[u] * Z[v]
    
    for step in range(2):
        gamma = gammas[step]
        beta = betas[step]
        
        # Apply exp(-i * gamma * C) where C = 0.5 * sum_k (1 - ZZ_k)
        # exp(-i*gamma*C) = exp(-i*gamma*m/2) * prod_k exp(i*gamma*ZZ_k/2)
        phase_id = np.exp(-1j * gamma * 0.5 * m)
        
        total_phase = np.ones(dim, dtype=complex)
        for k in range(m):
            zz_vals = ZZ[k]
            total_phase *= np.exp(1j * gamma * 0.5 * zz_vals)
        
        state = phase_id * total_phase * state
        
        # Apply exp(-i * beta * sum_i X_i)
        # sum_i X_i flips all bits: X_sum |j> = |j XOR (2^n - 1)>
        # exp(-i*beta*X_sum) = cos(beta)*I - i*sin(beta)*X_sum
        state = np.cos(beta) * state - 1j * np.sin(beta) * state[::-1]
    
    # Compute expected cut
    cut_val = 0.0
    all_ones = dim - 1
    for (u, v) in edges:
        flip_mask = (1 << u) | (1 << v)
        corr = np.sum(np.conj(state) * state[np.arange(dim) ^ flip_mask]).real
        cut_val += 0.5 - 0.5 * corr
    
    return cut_val

def _objective(params, n, edges, optimum):
    """Negative ratio: minimize -expected_cut/optimum."""
    if _remaining_time() < 1.0:
        return 1e10
    
    gammas = params[0::2]
    betas = params[1::2]
    
    try:
        ec = _expected_cut_state_vector(n, edges, gammas, betas)
        if optimum <= 0:
            return 0.0
        return -ec / optimum
    except:
        return 1.0

def _optimize_angles(n, edges, p=2):
    """Find optimal QAOA angles for a given graph using multi-start Nelder-Mead."""
    if _remaining_time() < 5.0:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    optimum = _exact_maxcut(n, edges)
    if optimum <= 0:
        return np.zeros((p, 2))
    
    n_params = 2 * p  # 4 parameters for p=2
    best_params = None
    best_val = 1e10
    
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0
    
    # Generate diverse initializations
    inits = []
    
    # Known good starting points from literature
    base_inits = [
        [0.5, 0.3, 0.3, 0.2],
        [0.3, 0.5, 0.5, 0.3],
        [np.pi/4, np.pi/6, np.pi/6, np.pi/8],
        [0.2, 0.4, 0.4, 0.2],
        [0.4, 0.2, 0.2, 0.4],
        [0.1, 0.6, 0.6, 0.1],
        [0.6, 0.1, 0.1, 0.6],
        [0.25, 0.35, 0.35, 0.25],
        [0.4, 0.4, 0.4, 0.4],
        [0.3, 0.3, 0.3, 0.3],
        [0.5, 0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2, 0.2],
        [0.7, 0.1, 0.1, 0.7],
        [0.1, 0.7, 0.7, 0.1],
    ]
    for b in base_inits:
        inits.append(np.array(b, dtype=float))
    
    # Density-dependent initializations
    if density < 0.4:
        inits.append(np.array([0.3, 0.4, 0.4, 0.3]))
        inits.append(np.array([0.2, 0.5, 0.5, 0.2]))
        inits.append(np.array([0.15, 0.45, 0.45, 0.15]))
    else:
        inits.append(np.array([0.4, 0.3, 0.3, 0.4]))
        inits.append(np.array([0.5, 0.2, 0.2, 0.5]))
        inits.append(np.array([0.45, 0.25, 0.25, 0.45]))
    
    # Random initializations with fixed seed for reproducibility
    rng = np.random.RandomState(12345)
    for _ in range(10):
        inits.append(rng.uniform(0.05, 0.8, size=n_params))
    
    # Evaluate all initializations first to find the best starting point
    init_vals = []
    for init in inits:
        if _remaining_time() < 5.0:
            break
        try:
            val = _objective(init, n, edges, optimum)
            init_vals.append((val, init.copy()))
        except:
            continue
    
    if not init_vals:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    # Sort by objective value (ascending, since we're minimizing)
    init_vals.sort(key=lambda x: x[0])
    
    # Use top initializations for optimization
    top_inits = [iv[1] for iv in init_vals[:min(8, len(init_vals))]]
    
    # Also include the absolute best initialization
    if init_vals[0][1] not in top_inits:
        top_inits.append(init_vals[0][1])
    
    for init in top_inits:
        if _remaining_time() < 5.0:
            break
        
        try:
            result = minimize(_objective, init, args=(n, edges, optimum),
                             method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-8,
                                      'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except:
            continue
    
    if best_params is None:
        # Use the best initialization as fallback
        best_params = init_vals[0][1]
    
    # Post-process: refine from the best point
    if _remaining_time() > 10.0:
        for _ in range(3):
            if _remaining_time() < 5.0:
                break
            try:
                result = minimize(_objective, best_params, args=(n, edges, optimum),
                                 method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-9,
                                          'adaptive': True})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except:
                break
    
    gammas = best_params[0::2]
    betas = best_params[1::2]
    return np.column_stack([gammas, betas])

def angles(n, edges, p):
    """Main entry point: return QAOA angles for the given graph."""
    if _remaining_time() < 2.0:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    try:
        result = _optimize_angles(n, edges, p)
        # Ensure output shape is (p, 2)
        if result.shape != (p, 2):
            result = np.tile([0.35, 0.25], (p, 1))
        return result
    except:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
# EVOLVE-BLOCK-END
