"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0
_START = time.time()

def _check_deadline():
    return (time.time() - _START) > _DEADLINE

def _maxcut_exact(n, edges):
    if n == 0:
        return 0
    best = 0
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
        
    for mask in range(1, 1 << n):
        cut = 0
        for u, v in edge_set:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _qaoa_expectation_ultra_fast(n, edges, p, gammas, betas, precomputed):
    """
    Ultra-fast QAOA expectation calculation.
    Uses pre-computed cut counts and index swaps for rotations.
    """
    dim = 1 << n
    indices = precomputed['indices']
    cut_counts = precomputed['cut_counts']
    swap_indices = precomputed['swap_indices'] # List of arrays for each qubit
    
    # Initialize state
    state = precomputed['init_state'].copy()
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply U_C: phase kickback
        if abs(gamma) > 1e-12:
            # phase = exp(-i * gamma * cut_count)
            # cut_counts is integer array
            state *= np.exp(-1j * gamma * cut_counts)
            
        # Apply U_B: X rotations
        if abs(beta) > 1e-12:
            c = np.cos(beta)
            s = np.sin(beta)
            for q in range(n):
                idx_0 = swap_indices[q]
                # idx_0 contains indices where bit q is 0
                # idx_1 = idx_0 | (1 << q)
                # We pre-computed pairs? 
                # Let's just use the mask logic or pre-computed pairs.
                # Pre-computed pairs is faster.
                # swap_indices[q] should be a list of tuples (i0, i1) or two arrays.
                # Let's use two arrays: i0, i1
                i0, i1 = precomputed['rot_pairs'][q]
                
                old_0 = state[i0]
                old_1 = state[i1]
                
                state[i0] = c * old_0 - 1j * s * old_1
                state[i1] = -1j * s * old_0 + c * old_1
        
        if _check_deadline():
            break

    # Calculate Expectation
    probs = np.abs(state)**2
    exp_val = 0.0
    # Pre-computed edge masks for Z_u Z_v?
    # Z_u Z_v = 1 if bits same, -1 if different.
    # We can pre-compute the vector of Z_u Z_v for each edge.
    for edge_idx, (u, v) in enumerate(precomputed['edge_list']):
        zz = precomputed['zz_vectors'][edge_idx]
        exp_zz = np.sum(probs * zz)
        exp_val += 0.5 * (1 - exp_zz)
        
    return exp_val

def _precompute_graph(n, edges):
    dim = 1 << n
    indices = np.arange(dim)
    
    # Pre-compute cut counts
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    edge_list = list(edge_set)
    
    cut_counts = np.zeros(dim, dtype=np.int32)
    for u, v in edge_list:
        mask_u = 1 << u
        mask_v = 1 << v
        # Cut if bits differ
        # (mask_u XOR mask_v) != 0 ? No, that's not right.
        # Bit u != Bit v
        # We can compute this vectorized:
        bits_u = (indices >> u) & 1
        bits_v = (indices >> v) & 1
        cut_counts += (bits_u != bits_v).astype(np.int32)
        
    # Pre-compute rotation pairs
    rot_pairs = []
    for q in range(n):
        mask = 1 << q
        i0 = indices[(indices & mask) == 0]
        i1 = i0 | mask
        rot_pairs.append((i0, i1))
        
    # Pre-compute Z_u Z_v vectors
    zz_vectors = []
    for u, v in edge_list:
        bits_u = (indices >> u) & 1
        bits_v = (indices >> v) & 1
        # Z = 1 - 2*bit
        z_u = 1 - 2 * bits_u
        z_v = 1 - 2 * bits_v
        zz = z_u * z_v
        zz_vectors.append(zz)
        
    # Initial state
    init_state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    return {
        'indices': indices,
        'cut_counts': cut_counts,
        'rot_pairs': rot_pairs,
        'edge_list': edge_list,
        'zz_vectors': zz_vectors,
        'init_state': init_state
    }

def angles(n, edges, p):
    if _check_deadline():
        return np.tile([0.35, 0.25], (p, 1))
    
    # Precompute graph properties
    precomp = _precompute_graph(n, edges)
    
    best_val = -np.inf
    best_params = np.zeros(p * 2)
    
    # Generate starts
    starts = []
    
    # 1. Spectral/Heuristic starts
    # Estimate avg degree
    m = len(edges)
    d = 2.0 * m / n if n > 0 else 1.0
    # Heuristic for p=2
    # gamma ~ pi / (4 * sqrt(d))
    g_est = np.pi / (4 * np.sqrt(d))
    b_est = np.pi / 4
    
    starts.append(np.concatenate([np.array([g_est, g_est]), np.array([b_est, b_est])]))
    starts.append(np.concatenate([np.array([g_est*0.5, g_est*0.5]), np.array([b_est, b_est])]))
    starts.append(np.concatenate([np.array([g_est*2.0, g_est*2.0]), np.array([b_est, b_est])]))
    
    # 2. Random starts
    np.random.seed(42)
    # Increase number of starts due to faster simulation
    n_starts = 40
    for _ in range(n_starts):
        g = np.random.uniform(0, np.pi/2, p)
        b = np.random.uniform(0, np.pi/2, p)
        starts.append(np.concatenate([g, b]))
        
    for x0 in starts:
        if _check_deadline():
            break
            
        x0 = np.clip(x0, 0, np.pi)
        
        def objective(x):
            if _check_deadline():
                return 0.0
            gammas = x[:p]
            betas = x[p:]
            try:
                val = _qaoa_expectation_ultra_fast(n, edges, p, gammas, betas, precomp)
                return -val
            except Exception:
                return 0.0
                
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=[(0, np.pi)] * (2 * p),
                options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
                
        except Exception:
            continue
            
    # Final polish with Nelder-Mead if time permits
    if not _check_deadline():
        x0 = best_params
        if x0 is None or np.all(x0 == 0):
            x0 = np.tile([0.3, 0.3], p)
            
        def objective_nm(x):
            if _check_deadline():
                return 0.0
            gammas = x[:p]
            betas = x[p:]
            try:
                val = _qaoa_expectation_ultra_fast(n, edges, p, gammas, betas, precomp)
                return -val
            except Exception:
                return 0.0
                
        try:
            res = minimize(
                objective_nm, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            val = -res.fun
            if val > best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
            
    gammas = best_params[:p]
    betas = best_params[p:]
    
    return np.column_stack((gammas, betas))
# EVOLVE-BLOCK-END
