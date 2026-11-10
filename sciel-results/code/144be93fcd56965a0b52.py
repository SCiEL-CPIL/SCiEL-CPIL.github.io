"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 135.0
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


def _qaoa_expected_cut(n, edges, gammas, betas, bit_arrays=None):
    """
    Compute expected cut value efficiently using vectorized operations.
    Uses the fact that for p=2, we can track the state vector.
    """
    if bit_arrays is None:
        bit_arrays = _precompute_bit_arrays(n)
    
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    edge_list = list(edges)
    
    for gamma, beta in zip(gammas, betas):
        # Apply cost phase: exp(-i * gamma * C) where C = sum (1-Z_i Z_j)/2
        # Phase for each basis state: exp(-i * gamma * num_same_bits/2 * |E| ... )
        # Actually: C|b> = (|E| - cut(b))/2 |b> where cut(b) is number of cut edges
        # So exp(-i*gamma*C) applies phase exp(-i*gamma*(|E|-cut(b))/2)
        
        # Compute cut count for each basis state
        cut_counts = np.zeros(dim, dtype=np.float64)
        for u, v in edge_list:
            bu = bit_arrays[u]
            bv = bit_arrays[v]
            cut_counts += (bu != bv).astype(np.float64)
        
        # Phase: exp(-i * gamma * (|E| - cut_counts) / 2)
        m = len(edge_list)
        phase = np.exp(-1j * gamma * (m - cut_counts) / 2.0)
        state *= phase
        
        # Apply mixer: exp(-i * beta * sum X) = product over q of (cos(beta/2) I - i sin(beta/2) X_q)
        c = np.cos(beta / 2)
        s = np.sin(beta / 2)
        
        for q in range(n):
            # X_q flips bit q: partner of basis b is b XOR (1<<q)
            flip_mask = 1 << q
            # For each basis state, mix with its partner
            # We need to be careful: we can't do this in-place naively
            # Use reshape trick: pair up states that differ only in bit q
            
            # Create index pairs
            # States where bit q is 0: indices where (b >> q) & 1 == 0
            # Their partners have bit q = 1
            
            # Vectorized approach:
            # For all b where bit q is 0:
            #   new[b] = c*old[b] - i*s*old[b|flip]
            #   new[b|flip] = c*old[b|flip] - i*s*old[b]
            
            # Get indices where bit q is 0
            bit_q = bit_arrays[q]
            idx0 = np.where(bit_q == 0)[0]
            idx1 = idx0 | flip_mask
            
            old0 = state[idx0].copy()
            old1 = state[idx1].copy()
            
            state[idx0] = c * old0 - 1j * s * old1
            state[idx1] = c * old1 - 1j * s * old0
    
    # Compute expected cut
    probs = np.abs(state) ** 2
    
    expected_cut = 0.0
    for u, v in edge_list:
        bu = bit_arrays[u]
        bv = bit_arrays[v]
        p_diff = np.sum(probs[bu != bv])
        expected_cut += p_diff
    
    return expected_cut


def _optimize_angles(n, edges, p):
    """Optimize QAOA angles for a single graph using multi-start Nelder-Mead."""
    if _check_deadline():
        return np.tile([0.35, 0.25], (p, 1))
    
    opt = _maxcut_exact(n, edges)
    if opt == 0:
        return np.zeros((p, 2))
    
    # Precompute bit arrays once
    bit_arrays = _precompute_bit_arrays(n)
    
    def objective(params):
        if _check_deadline():
            return 1e10
        gammas = params[:p]
        betas = params[p:]
        try:
            ec = _qaoa_expected_cut(n, edges, gammas, betas, bit_arrays)
        except Exception:
            return 1e10
        return -ec / opt
    
    # Generate diverse starting points
    starts = []
    # Known good initial values for p=2
    starts.append(np.array([0.4, 0.4, 0.25, 0.25]))
    starts.append(np.array([0.3, 0.3, 0.2, 0.2]))
    starts.append(np.array([0.5, 0.5, 0.3, 0.3]))
    starts.append(np.array([0.2, 0.2, 0.15, 0.15]))
    starts.append(np.array([0.6, 0.4, 0.2, 0.4]))
    starts.append(np.array([0.35, 0.25, 0.25, 0.15]))
    starts.append(np.array([0.45, 0.35, 0.15, 0.25]))
    
    # Add some random starts
    rng = np.random.default_rng(42)
    for _ in range(3):
        g = rng.uniform(0.1, 0.8, p)
        b = rng.uniform(0.05, 0.5, p)
        starts.append(np.concatenate([g, b]))
    
    best_val = 1e10
    best_params = None
    
    for start in starts:
        if _check_deadline():
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
    
    if best_params is None:
        return np.tile([0.35, 0.25], (p, 1))
    
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
