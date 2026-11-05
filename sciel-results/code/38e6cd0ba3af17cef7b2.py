"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache

_cache = {}

def _vectorized_evaluate(n, edges, gammas, betas):
    """
    Fast vectorized QAOA expected cut evaluation.
    Note: This version is kept for compatibility but _optimize_fast uses a faster internal version.
    """
    dim = 1 << n
    m = len(edges)
    
    if m == 0:
        return 0.0
    
    # Precompute bit patterns for all basis states (vectorized)
    bits = np.array([[(idx >> k) & 1 for k in range(n)] for idx in range(dim)], dtype=np.int8)
    
    # Precompute for each edge: (-1)^(bit_u + bit_v) for all basis states
    edge_parity = np.zeros((dim, m), dtype=np.float64)
    for j, (u, v) in enumerate(edges):
        edge_parity[:, j] = np.where(bits[:, u] == bits[:, v], 1.0, -1.0)
    
    c_eigenvalues = edge_parity.sum(axis=1)  # shape (dim,)
    
    # Start state: |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute cos/sin for each layer
    for l in range(len(gammas)):
        gamma = gammas[l]
        beta = betas[l]
        
        # Apply exp(-i * gamma * C): diagonal phase
        psi *= np.exp(-1j * gamma * c_eigenvalues)
        
        # Apply exp(-i * beta * sum X_i)
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        # Apply to each qubit using reshape trick
        psi_tensor = psi.reshape([2] * n)
        
        for q in range(n):
            slc0 = [slice(None)] * n
            slc0[q] = 0
            slc1 = [slice(None)] * n
            slc1[q] = 1
            slc0 = tuple(slc0)
            slc1 = tuple(slc1)
            
            tmp0 = psi_tensor[slc0].copy()
            tmp1 = psi_tensor[slc1].copy()
            
            psi_tensor[slc0] = cos_b * tmp0 - 1j * sin_b * tmp1
            psi_tensor[slc1] = cos_b * tmp1 - 1j * sin_b * tmp0
        
        psi = psi_tensor.reshape(-1)
    
    # Compute expected cut
    probs = np.abs(psi) ** 2  # shape (dim,)
    
    # For each edge j: <Z_u Z_v> = sum_i probs[i] * edge_parity[i, j]
    zz_expects = edge_parity.T @ probs  # shape (m,)
    
    expected_cut = np.sum((1.0 - zz_expects) / 2.0)
    return float(expected_cut)


def _exhaustive_maxcut_fast(n, edges):
    """Fast exhaustive MaxCut using vectorized operations."""
    if not edges:
        return 0
    
    dim = 1 << n
    # For each assignment mask, compute cut size
    # Vectorized: for each edge (u,v), check if bits differ
    edges_arr = np.array(edges)  # shape (m, 2)
    
    # Create all masks
    masks = np.arange(dim)
    
    # Extract bits for all edges
    # bits_for_edge_u = (masks >> edges[:, 0]) & 1
    # This needs broadcasting: masks[:, None] >> edges[None, :, 0] & 1
    u_bits = (masks[:, None] >> edges_arr[None, :, 0]) & 1  # shape (dim, m)
    v_bits = (masks[:, None] >> edges_arr[None, :, 1]) & 1  # shape (dim, m)
    
    cut_sizes = np.sum(u_bits != v_bits, axis=1)
    return int(cut_sizes.max())


def _optimize_fast(n, edges, p=2):
    """
    Multi-start L-BFGS-B optimization with precomputed constants for speed.
    """
    from scipy.optimize import minimize
    
    best_maxcut = _exhaustive_maxcut_fast(n, edges)
    if best_maxcut == 0:
        return np.zeros((p, 2))
    
    # Precompute constant terms for the graph
    dim = 1 << n
    m = len(edges)
    edges_arr = np.array(edges)
    u_idx = edges_arr[:, 0]
    v_idx = edges_arr[:, 1]
    
    # Compute edge parities and c_eigenvalues
    indices = np.arange(dim)
    bit_u = (indices[:, None] >> u_idx[None, :]) & 1
    bit_v = (indices[:, None] >> v_idx[None, :]) & 1
    edge_parity = np.where(bit_u == bit_v, 1.0, -1.0) # shape (dim, m)
    c_eigenvalues = edge_parity.sum(axis=1) # shape (dim,)
    
    # Precompute X-swap indices for each qubit
    x_swap_indices = []
    for q in range(n):
        idx = np.arange(dim)
        flipped = idx ^ (1 << q)
        x_swap_indices.append(flipped)
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        
        psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        
        for l in range(p):
            # Apply C phase
            psi *= np.exp(-1j * gammas[l] * c_eigenvalues)
            
            # Apply X rotation
            cos_b = np.cos(betas[l])
            sin_b = np.sin(betas[l])
            
            for q in range(n):
                swapped_psi = psi[x_swap_indices[q]]
                psi = cos_b * psi - 1j * sin_b * swapped_psi
        
        probs = np.abs(psi) ** 2
        zz_expects = edge_parity.T @ probs
        expected_cut = np.sum((1.0 - zz_expects) / 2.0)
        return -expected_cut / best_maxcut
    
    bounds = [(0.0, np.pi)] * (2 * p)
    
    # Multiple starting points
    best_result = None
    best_val = np.inf
    
    # Generate diverse starting points
    rng = np.random.RandomState(42)
    start_points = []
    
    # Known good initializations for p=2
    base_g = 0.5 * np.sqrt(np.pi * np.log(2)) / p
    base_b = 0.5 * np.pi / (2 * p)
    
    start_points.append(np.array([base_g] * p + [base_b] * p))
    start_points.append(np.array([0.3] * p + [0.2] * p))
    start_points.append(np.array([0.5] * p + [0.3] * p))
    start_points.append(np.array([0.2] * p + [0.1] * p))
    start_points.append(np.array([0.4] * p + [0.25] * p))
    start_points.append(np.array([0.6] * p + [0.15] * p))
    
    # Add random starts (increased count)
    for _ in range(10):
        gs = rng.uniform(0.1, 0.8, size=p)
        bs = rng.uniform(0.05, 0.4, size=p)
        start_points.append(np.concatenate([gs, bs]))
    
    for x0 in start_points:
        x0 = np.clip(x0, 0.01, np.pi - 0.01)
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-7})
            val = result.fun
            if val < best_val:
                best_val = val
                best_result = result.x
        except Exception:
            continue
    
    if best_result is None:
        return np.tile([0.35, 0.25], (p, 1))
    
    params = np.clip(best_result, 0, np.pi)
    gammas = params[:p]
    betas = params[p:]
    return np.column_stack([gammas, betas])


def angles(n, edges, p):
    """
    Compute QAOA angles for the given graph using fast vectorized optimization.
    Results are cached for repeated calls with the same graph.
    """
    edges_list = tuple(sorted((int(u), int(v)) for u, v in edges))
    cache_key = (n, edges_list, p)
    
    if cache_key in _cache:
        return _cache[cache_key].copy()
    
    try:
        opt_angles = _optimize_fast(n, list(edges_list), p=p)
        opt_angles = opt_angles.reshape(p, 2)
        opt_angles = np.clip(opt_angles, 0, np.pi)
        _cache[cache_key] = opt_angles
        return opt_angles
    except Exception:
        fallback = np.tile([0.35, 0.25], (p, 1))
        return fallback
# EVOLVE-BLOCK-END
