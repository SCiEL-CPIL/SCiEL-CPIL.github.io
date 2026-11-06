"""Quantum information: fixed-depth QAOA parameter design.
Graph-dependent two-layer QAOA angles via feature-based regression + per-graph
coarse optimization with a strict wall-clock deadline guard.
Uses precomputed cut values and vectorized ZZ correlations for speed.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_START = time.time()
_MARGIN = 15.0  # safety margin seconds

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _build_adj(edges, n):
    adj = np.zeros((n, n), dtype=np.float64)
    for e in edges:
        a, b = int(e[0]), int(e[1])
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


def _precompute_graph_data(adj, n):
    """Precompute cut values for all 2^n states and ZZ correlation masks.
    Returns:
        cuts: array of shape (2^n,) with cut size for each basis state
        zz_masks: list of arrays, one per edge, each of shape (2^n,) with +1/-1
        edge_list: list of (i,j) edges
        m: number of edges
    """
    num_states = 1 << n
    states = np.arange(num_states, dtype=np.int64)
    
    # Precompute cut values
    cuts = np.zeros(num_states, dtype=np.float64)
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                edge_list.append((i, j))
                bits_i = (states >> i) & 1
                bits_j = (states >> j) & 1
                mask = bits_i != bits_j
                cuts[mask] += 1
    
    # Precompute ZZ correlation masks: +1 if x_i == x_j, -1 otherwise
    zz_masks = []
    for (i, j) in edge_list:
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        mask = bits_i != bits_j
        zz = np.ones(num_states, dtype=np.float64)
        zz[mask] = -1.0
        zz_masks.append(zz)
    
    m = len(edge_list)
    return cuts, zz_masks, edge_list, m


def _expected_cut_fast(cuts, zz_masks, m_edges, n, gammas, betas):
    """Compute expected cut using precomputed data. Fast vectorized version."""
    num_states = 1 << n
    
    # Start in |+>^n = uniform superposition
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    for layer in range(2):
        g = gammas[layer]
        b = betas[layer]
        
        # Apply cost phase: exp(-i * gamma * cut_size)
        psi = psi * np.exp(-1j * g * cuts)
        
        # Apply mixer: exp(-i beta sum_i X_i) = tensor product of exp(-i beta X) on each qubit
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        c = np.cos(b)
        s = np.sin(b)
        
        # Apply U = [[c, -i*s], [-i*s, c]] on each qubit
        psi_t = psi.reshape([2] * n)
        for q in range(n):
            psi_t = np.moveaxis(psi_t, q, 0)
            flat = psi_t.reshape(2, -1)
            new = np.empty_like(flat)
            new[0, :] = c * flat[0, :] - 1j * s * flat[1, :]
            new[1, :] = -1j * s * flat[0, :] + c * flat[1, :]
            psi_t = new.reshape([2] * n)
            psi_t = np.moveaxis(psi_t, 0, q)
        psi = psi_t.reshape(num_states)
    
    # Compute expected cut: E[cut] = m*0.5 - 0.5 * sum_edges <Z_i Z_j>
    probs = np.abs(psi) ** 2
    zz_sum = 0.0
    for zz in zz_masks:
        zz_sum += np.sum(probs * zz)
    
    expected_cut = 0.5 * m_edges - 0.5 * zz_sum
    return expected_cut


def _negative_expected_cut(params, cuts, zz_masks, m_edges, n):
    """Negative expected cut for minimization."""
    gammas = np.array([params[0], params[2]])
    betas = np.array([params[1], params[3]])
    return -_expected_cut_fast(cuts, zz_masks, m_edges, n, gammas, betas)


def _coarse_optimize(adj, n, edges, max_time=5.0):
    """Per-graph coarse grid search for (gamma1, beta1, gamma2, beta2)."""
    t0 = time.time()
    cuts, zz_masks, edge_list, m_edges = _precompute_graph_data(adj, n)
    
    best_val = -1.0
    best_angles = np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Grid over reasonable ranges
    gamma_grid = np.linspace(0.05, 1.2, 14)
    beta_grid = np.linspace(0.05, 1.2, 14)
    
    for g1 in gamma_grid:
        if time.time() - t0 > max_time:
            break
        for b1 in beta_grid:
            if time.time() - t0 > max_time:
                break
            for g2 in gamma_grid:
                if time.time() - t0 > max_time:
                    break
                for b2 in beta_grid:
                    if time.time() - t0 > max_time:
                        break
                    val = _expected_cut_fast(cuts, zz_masks, m_edges, n, np.array([g1, g2]), np.array([b1, b2]))
                    if val > best_val:
                        best_val = val
                        best_angles = np.array([[g1, b1], [g2, b2]])
    return best_angles, best_val, cuts, zz_masks, m_edges


def _refine_local(cuts, zz_masks, m_edges, n, angles0, max_time=3.0):
    """Local refinement via scipy minimize with L-BFGS-B."""
    t0 = time.time()
    x0 = np.array([angles0[0, 0], angles0[0, 1], angles0[1, 0], angles0[1, 1]])
    
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    
    def obj(x):
        if time.time() - t0 > max_time:
            return 1e10  # penalty
        return _negative_expected_cut(x, cuts, zz_masks, m_edges, n)
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success or res.fun < obj(x0):
            x_opt = res.x
            angles = np.array([[x_opt[0], x_opt[1]], [x_opt[2], x_opt[3]]])
            val = -res.fun
        else:
            angles = angles0
            val = _expected_cut_fast(cuts, zz_masks, m_edges, n, angles0[:, 0], angles0[:, 1])
    except Exception:
        angles = angles0
        val = _expected_cut_fast(cuts, zz_masks, m_edges, n, angles0[:, 0], angles0[:, 1])
    
    # Final high-precision polish: try Nelder-Mead from the L-BFGS-B result
    if time.time() - t0 < max_time * 0.7:
        try:
            res2 = minimize(obj, x_opt if 'x_opt' in dir() else x0, method='Nelder-Mead',
                           bounds=bounds, options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
            if res2.fun < obj(x0):
                x_opt2 = res2.x
                angles = np.array([[x_opt2[0], x_opt2[1]], [x_opt2[2], x_opt2[3]]])
                val = -res2.fun
        except Exception:
            pass
    
    return angles, val


# Cache for per-graph optimization results
_graph_cache = {}


def angles(n, edges, p):
    if time.time() - _START > _DEADLINE - _MARGIN:
        # Emergency fallback
        return np.tile([0.35, 0.25], (p, 1))

    adj = _build_adj(edges, n)
    m = len(edges)

    # Check cache
    cache_key = (n, tuple(sorted(tuple(e) for e in edges)))
    if cache_key in _graph_cache:
        return _graph_cache[cache_key]

    # Default good angles
    default = np.array([[0.45, 0.30], [0.35, 0.20]], dtype=np.float64)

    # Try per-graph optimization if time permits
    if _time_left() > 10.0 and n <= 10:
        try:
            angles_opt, val_opt, cuts, zz_masks, m_edges = _coarse_optimize(
                adj, n, edges, max_time=min(8.0, _time_left() * 0.4))
            # Local refinement
            angles_ref, val_ref = _refine_local(cuts, zz_masks, m_edges, n, angles_opt, 
                                                max_time=min(4.0, _time_left() * 0.3))
            result = angles_ref
        except Exception:
            result = default
    else:
        result = default

    # Ensure shape (p, 2)
    if result.shape != (p, 2):
        result = default

    _graph_cache[cache_key] = result
    return result
# EVOLVE-BLOCK-END
