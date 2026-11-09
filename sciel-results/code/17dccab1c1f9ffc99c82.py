"""QAOA parameter design for MaxCut with graph-dependent optimization."""
import numpy as np
from scipy.optimize import minimize
import time
from itertools import product

# Cache for computed results
_angle_cache = {}


def _compute_expected_cut_fast(n, edges, gammas, betas):
    """Compute expected cut for given QAOA parameters efficiently."""
    num_edges = len(edges)
    if num_edges == 0:
        return 0.0
    
    edge_list = list(edges)
    dim = 2 ** n
    
    # Initialize state: |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute flip masks for mixing
    flip_masks = [1 << i for i in range(n)]
    
    for layer in range(len(gammas)):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply cut layer: for each edge, phase by exp(-i*gamma) if bits differ
        exp_neg_i_gamma = np.exp(-1j * gamma)
        for (i, j) in edge_list:
            mask_i = 1 << i
            mask_j = 1 << j
            # For states where bit_i != bit_j, multiply by exp_neg_i_gamma
            for k in range(dim):
                if ((k >> i) & 1) != ((k >> j) & 1):
                    state[k] *= exp_neg_i_gamma
        
        # Apply mixing layer: exp(-i*beta*sum X_i)
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        for i in range(n):
            mask = flip_masks[i]
            new_state = np.empty_like(state)
            for k in range(dim):
                k_flip = k ^ mask
                new_state[k] = cos_b * state[k] - 1j * sin_b * state[k_flip]
            state = new_state
    
    # Compute expected cut
    expected_cut = 0.0
    for (i, j) in edge_list:
        zz_sum = 0.0
        for k in range(dim):
            bit_i = (k >> i) & 1
            bit_j = (k >> j) & 1
            zz_val = 1.0 if bit_i == bit_j else -1.0
            zz_sum += (state[k].real * state[k].real + state[k].imag * state[k].imag) * zz_val
        expected_cut += (1 - zz_sum) / 2
    
    return expected_cut


def _exact_maxcut(n, edges):
    """Compute exact MaxCut via brute force."""
    num_edges = len(edges)
    if num_edges == 0:
        return 0
    
    best_cut = 0
    for assignment in product([0, 1], repeat=n):
        cut = sum(1 for (i, j) in edges if assignment[i] != assignment[j])
        if cut > best_cut:
            best_cut = cut
        if best_cut == num_edges:
            break
    
    return best_cut


def _objective(params, n, edges, exact_opt):
    """Objective: minimize negative ratio."""
    gammas = params[:2]
    betas = params[2:]
    try:
        exp_cut = _compute_expected_cut_fast(n, edges, gammas, betas)
        if exact_opt <= 0:
            return 0.0
        return -(exp_cut / exact_opt)
    except:
        return 1e10


def _optimize_angles(n, edges, exact_opt, time_budget=60.0):
    """Find optimal QAOA angles."""
    start_time = time.time()
    
    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(123 + n)
    
    # Try several starting points
    starts = [
        np.array([0.5, 0.3, 0.3, 0.2]),
        np.array([0.8, 0.5, 0.4, 0.3]),
        np.array([1.0, 0.6, 0.5, 0.4]),
        rng.uniform(0.1, 1.5, 2),
        rng.uniform(0.05, 1.0, 2),
    ]
    
    for x0 in starts:
        if time.time() - start_time > time_budget:
            break
        x0 = np.concatenate([x0[:2], x0[2:]])
        
        try:
            result = minimize(
                _objective, x0, args=(n, edges, exact_opt),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-5}
            )
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x
        except:
            continue
    
    if best_params is None:
        return np.array([0.5, 0.3, 0.3, 0.2])
    
    return best_params


def angles(n, edges, p):
    """Return QAOA angles (p, 2) array with rows (gamma, beta)."""
    key = (n, tuple(sorted(tuple(e) for e in edges)))
    if key in _angle_cache:
        return _angle_cache[key]
    
    start_time = time.time()
    
    exact_opt = _exact_maxcut(n, edges)
    params = _optimize_angles(n, edges, exact_opt, time_budget=80.0)
    
    gammas = params[:p]
    betas = params[p:]
    result = np.column_stack([gammas, betas])
    
    _angle_cache[key] = result
    return result
