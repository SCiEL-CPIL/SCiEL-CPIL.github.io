"""QAOA angle design for fixed-depth p=2 MaxCut on small graphs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 130.0  # safety margin before 150s kill
_CACHE = {}


def _precompute_graph(n, edges):
    """Precompute cut counts for all 2^n states."""
    num_states = 2 ** n
    cut_counts = np.zeros(num_states, dtype=np.float64)
    m = len(edges)
    if m == 0:
        return cut_counts, 0
    # For each edge (u,v), the cut condition is bit u != bit v
    # Build a mask for each edge: states where u and v differ
    for u, v in edges:
        # For states where bit u != bit v, increment cut count
        # bit u is at position u, bit v is at position v
        # Create mask: states where bit u XOR bit v == 1
        # We can do this by: for each state, check ((idx >> u) ^ (idx >> v)) & 1
        idxs = np.arange(num_states, dtype=np.int64)
        mask = ((idxs >> u) ^ (idxs >> v)) & 1
        cut_counts += mask
    return cut_counts, m


def _expected_cut_fast(n, edges, cut_counts, gamma1, beta1, gamma2, beta2):
    """Compute expected cut value via vectorized state-vector simulation."""
    num_states = 2 ** n
    psi = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    cos_b1 = np.cos(beta1)
    sin_b1 = np.sin(beta1)
    cos_b2 = np.cos(beta2)
    sin_b2 = np.sin(beta2)
    
    phase1 = np.exp(-1j * gamma1 * cut_counts)
    phase2 = np.exp(-1j * gamma2 * cut_counts)
    
    for layer in range(2):
        # Apply exp(-i gamma C): diagonal phase
        if layer == 0:
            psi = psi * phase1
            cb, sb = cos_b1, sin_b1
        else:
            psi = psi * phase2
            cb, sb = cos_b2, sin_b2
        
        # Apply exp(-i beta sum_j X_j) = tensor product of single-qubit rotations
        # For each qubit j: new_psi[idx] = cb * psi[idx] - 1j * sb * psi[idx ^ (1<<j)]
        for j in range(n):
            flip_mask = 1 << j
            # Vectorized: for each idx, flipped = idx ^ flip_mask
            idxs = np.arange(num_states, dtype=np.int64)
            flipped = idxs ^ flip_mask
            psi = cb * psi - 1j * sb * psi[flipped]
    
    probs = np.abs(psi) ** 2
    return np.sum(probs * cut_counts)


def _exact_optimum(n, edges, cut_counts):
    """Compute exact MaxCut optimum."""
    return int(np.max(cut_counts))


def _optimize_angles(n, edges):
    """Optimize QAOA angles for a given graph."""
    key = (n, tuple(sorted(edges)))
    if key in _CACHE:
        return _CACHE[key]
    
    if time.time() > _DEADLINE:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    cut_counts, m = _precompute_graph(n, edges)
    opt = _exact_optimum(n, edges, cut_counts)
    if opt == 0:
        result = np.array([[0.0, 0.0], [0.0, 0.0]])
        _CACHE[key] = result
        return result
    
    def neg_ratio(params):
        g1, b1, g2, b2 = params
        ec = _expected_cut_fast(n, edges, cut_counts, g1, b1, g2, b2)
        return -ec / opt
    
    # Stage 1: Coarse grid search
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Use a moderate grid
    grid_points = 10
    grid = np.linspace(0.0, np.pi, grid_points)
    
    for g1 in grid:
        for b1 in grid:
            for g2 in grid:
                for b2 in grid:
                    if time.time() > _DEADLINE - 10:
                        break
                    val = neg_ratio([g1, b1, g2, b2])
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
    
    # Stage 2: Local refinement using Nelder-Mead
    if time.time() < _DEADLINE - 15:
        try:
            from scipy.optimize import minimize
            result = minimize(
                neg_ratio,
                best_params,
                method='Nelder-Mead',
                bounds=[(0, np.pi)] * 4,
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Stage 3: Multi-start local refinement from a few additional points
    if time.time() < _DEADLINE - 10:
        try:
            from scipy.optimize import minimize
            # Try a few different starting points
            starts = [
                np.array([0.5, 0.3, 0.5, 0.3]),
                np.array([1.0, 0.5, 1.0, 0.5]),
                np.array([0.2, 0.8, 0.2, 0.8]),
            ]
            for start in starts:
                if time.time() > _DEADLINE - 5:
                    break
                result = minimize(
                    neg_ratio,
                    start,
                    method='Nelder-Mead',
                    bounds=[(0, np.pi)] * 4,
                    options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x
        except Exception:
            pass
    
    angles_arr = np.array([[best_params[0], best_params[1]],
                           [best_params[2], best_params[3]]])
    _CACHE[key] = angles_arr
    return angles_arr


def angles(n, edges, p):
    """Return optimized QAOA angles for the given graph."""
    if time.time() > _DEADLINE:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        result = _optimize_angles(n, edges)
        return result
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
