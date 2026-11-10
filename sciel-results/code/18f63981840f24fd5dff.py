"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 140.0
_START = time.time()


def _check_deadline():
    return (time.time() - _START) > _DEADLINE


def _build_adj(n, edges):
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _maxcut_exact(n, edges):
    if n == 0 or not edges:
        return 0
    # Vectorized brute force for small n
    dim = 1 << n
    masks = np.arange(dim, dtype=np.int64)
    best = 0
    for u, v in edges:
        bu = (masks >> u) & 1
        bv = (masks >> v) & 1
        diff = (bu != bv).astype(np.int64)
        # For each mask, count how many edges are cut
        # We need to accumulate per-mask cut counts
        # Better approach: compute all cuts at once
    # Recompute properly
    cut_counts = np.zeros(dim, dtype=np.int64)
    for u, v in edges:
        bu = (masks >> u) & 1
        bv = (masks >> v) & 1
        cut_counts += (bu != bv).astype(np.int64)
    best = int(cut_counts.max())
    return best


def _precompute_bit_arrays(n):
    """Precompute bit arrays for all basis states."""
    dim = 1 << n
    masks = np.arange(dim, dtype=np.int64)
    bit_arrays = []
    for q in range(n):
        bit_arrays.append((masks >> q) & 1)
    return bit_arrays


def _precompute_edge_masks(n, edges):
    """Precompute per-edge cut indicator arrays (1 where bits differ)."""
    dim = 1 << n
    masks = np.arange(dim, dtype=np.int64)
    edge_masks = []
    for u, v in edges:
        bu = (masks >> u) & 1
        bv = (masks >> v) & 1
        edge_masks.append((bu != bv).astype(np.float64))
    return edge_masks


def _precompute_flip_indices(n):
    """Precompute index pairs for X-gate application per qubit."""
    dim = 1 << n
    masks = np.arange(dim, dtype=np.int64)
    flip_indices = []
    for q in range(n):
        bit_q = (masks >> q) & 1
        idx0 = np.where(bit_q == 0)[0]
        flip_mask = 1 << q
        idx1 = idx0 | flip_mask
        flip_indices.append((idx0, idx1))
    return flip_indices


def _qaoa_expected_cut(n, edges, gammas, betas, edge_masks=None, flip_indices=None):
    """
    Compute expected cut value efficiently using precomputed masks and indices.
    """
    if edge_masks is None:
        edge_masks = _precompute_edge_masks(n, edges)
    if flip_indices is None:
        flip_indices = _precompute_flip_indices(n)
    
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    m = len(edges)
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost phase: exp(-i * gamma * C) where C = sum (1-Z_i Z_j)/2
        # C|b> = (|E| - cut(b))/2 |b>
        # Phase: exp(-i * gamma * (|E| - cut(b)) / 2)
        
        # Compute cut count using precomputed edge masks
        cut_counts = np.zeros(dim, dtype=np.float64)
        for em in edge_masks:
            cut_counts += em
        
        phase = np.exp(-1j * gamma * (m - cut_counts) / 2.0)
        state *= phase
        
        # Apply mixer: exp(-i * beta * sum X)
        c = np.cos(beta / 2)
        s = np.sin(beta / 2)
        
        for q in range(n):
            idx0, idx1 = flip_indices[q]
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            state[idx0] = c * old0 - 1j * s * old1
            state[idx1] = c * old1 - 1j * s * old0
    
    # Compute expected cut using precomputed edge masks
    probs = np.abs(state) ** 2
    
    expected_cut = 0.0
    for em in edge_masks:
        expected_cut += np.sum(probs * em)
    
    return expected_cut


def _optimize_angles(n, edges, p):
    """Optimize QAOA angles using three-phase: NM multi-start → L-BFGS-B → NM polish."""
    if _check_deadline():
        return np.tile([0.35, 0.25], (p, 1))
    
    opt = _maxcut_exact(n, edges)
    if opt == 0:
        return np.zeros((p, 2))
    
    # Precompute once
    edge_masks = _precompute_edge_masks(n, edges)
    flip_indices = _precompute_flip_indices(n)
    
    def objective(params):
        if _check_deadline():
            return 1e10
        gammas = params[:p]
        betas = params[p:]
        try:
            ec = _qaoa_expected_cut(n, edges, gammas, betas, edge_masks, flip_indices)
        except Exception:
            return 1e10
        return -ec / opt
    
    def objective_grad(params):
        """Central-difference gradient for L-BFGS-B."""
        g = np.zeros_like(params)
        h = 1e-6
        f0 = objective(params)
        for i in range(len(params)):
            params[i] += h
            fp = objective(params)
            params[i] -= 2 * h
            fm = objective(params)
            params[i] += h
            g[i] = (fp - fm) / (2 * h)
        return g, g
    
    # Generate diverse starting points
    starts = []
    starts.append(np.array([0.4, 0.4, 0.25, 0.25]))
    starts.append(np.array([0.3, 0.3, 0.2, 0.2]))
    starts.append(np.array([0.5, 0.5, 0.3, 0.3]))
    starts.append(np.array([0.2, 0.2, 0.15, 0.15]))
    starts.append(np.array([0.6, 0.4, 0.2, 0.4]))
    starts.append(np.array([0.35, 0.25, 0.25, 0.15]))
    starts.append(np.array([0.45, 0.35, 0.15, 0.25]))
    
    rng = np.random.default_rng(42)
    for _ in range(4):
        g = rng.uniform(0.1, 0.8, p)
        b = rng.uniform(0.05, 0.5, p)
        starts.append(np.concatenate([g, b]))
    
    # Phase 1: Multi-start Nelder-Mead
    best_val = 1e10
    best_params = None
    
    for start in starts:
        if _check_deadline():
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
    
    if best_params is None:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Phase 2: L-BFGS-B refinement (with finite-diff gradient)
    if not _check_deadline():
        try:
            def obj_and_grad(params):
                val = objective(params)
                g = np.zeros_like(params)
                h = 1e-5
                for i in range(len(params)):
                    params[i] += h
                    fp = objective(params)
                    params[i] -= 2 * h
                    fm = objective(params)
                    params[i] += h
                    g[i] = (fp - fm) / (2 * h)
                return val, g
            
            res2 = minimize(obj_and_grad, best_params, method='L-BFGS-B',
                           jac=True, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final Nelder-Mead polish with tighter tolerance
    if not _check_deadline():
        try:
            res3 = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12,
                                    'adaptive': True})
            if res3.fun < best_val:
                best_val = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
    
    gammas = np.mod(best_params[:p], 2 * np.pi)
    betas = np.mod(best_params[p:], 2 * np.pi)
    return np.column_stack([gammas, betas])


def angles(n, edges, p):
    """Main entry point: compute optimal QAOA angles for the given graph."""
    if _check_deadline():
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        result = _optimize_angles(n, edges, p)
        if result is None or result.shape != (p, 2):
            return np.tile([0.35, 0.25], (p, 1))
        return result
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
