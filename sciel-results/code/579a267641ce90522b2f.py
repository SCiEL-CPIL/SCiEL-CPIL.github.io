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
    # For small n (<= 10), brute force all 2^(n-1) assignments
    best = 0
    for mask in range(1, 1 << n):
        bits = [(mask >> i) & 1 for i in range(n)]
        cut = 0
        for u, v in edges:
            if bits[u] != bits[v]:
                cut += 1
        if cut > best:
            best = cut
    return best


def _qaoa_state(n, edges, gammas, betas):
    """
    Compute the QAOA state vector for given angles.
    State is a complex vector of size 2^n.
    """
    # Initialize |+>^n state: all entries 1/sqrt(2^n)
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute edge lists for efficiency
    edge_list = list(edges)
    
    for gamma, beta in zip(gammas, betas):
        # Apply CNOT layer: exp(-i * gamma * C)
        # C = sum_{(i,j) in E} Z_i Z_j / 4 * (-1) ... actually for MaxCut:
        # C = sum_{(i,j) in E} (1 - Z_i Z_j)/2
        # So exp(-i*gamma*C) = exp(-i*gamma/2 * sum (1-Z_i Z_j))
        # = exp(-i*gamma*|E|/2) * prod_{(i,j)} exp(i*gamma/2 * Z_i Z_j)
        # exp(i*gamma/2 * Z_i Z_j) applies phase: if Z_i Z_j = 1 (same bit), phase = exp(i*gamma/2)
        # if Z_i Z_j = -1 (diff bit), phase = exp(-i*gamma/2)
        
        # We apply the phase for each edge
        for u, v in edge_list:
            # Apply phase based on bits u and v
            # For basis state |b>, bit u is (b >> u) & 1, bit v is (b >> v) & 1
            # If same: phase = exp(i*gamma/2), else exp(-i*gamma/2)
            phase_same = np.exp(1j * gamma / 2)
            phase_diff = np.exp(-1j * gamma / 2)
            
            for basis in range(dim):
                bu = (basis >> u) & 1
                bv = (basis >> v) & 1
                if bu == bv:
                    state[basis] *= phase_same
                else:
                    state[basis] *= phase_diff
        
        # Apply H layer: exp(-i * beta * sum X) = prod (cos(beta/2) I - i sin(beta/2) X)
        # This is a product of single-qubit rotations
        c = np.cos(beta / 2)
        s = np.sin(beta / 2)
        
        for q in range(n):
            # For each qubit q, apply rotation
            # The rotation acts on basis states where qubit q is 0 -> mix with 1 and vice versa
            for basis in range(dim):
                if (basis >> q) & 1 == 0:
                    partner = basis | (1 << q)
                    old_b = state[basis]
                    old_p = state[partner]
                    state[basis] = c * old_b - 1j * s * old_p
                    state[partner] = c * old_p - 1j * s * old_b
    
    return state


def _expected_cut(n, edges, gammas, betas):
    """Compute expected cut value from QAOA state."""
    state = _qaoa_state(n, edges, gammas, betas)
    probs = np.abs(state) ** 2
    
    expected_cut = 0.0
    for u, v in edges:
        # P(bit_u != bit_v) = sum over basis where bits differ
        p_diff = 0.0
        for basis in range(1 << n):
            bu = (basis >> u) & 1
            bv = (basis >> v) & 1
            if bu != bv:
                p_diff += probs[basis]
        expected_cut += p_diff
    
    return expected_cut


def _optimize_angles(n, edges, p):
    """Optimize QAOA angles for a single graph using multi-start Nelder-Mead."""
    if _check_deadline():
        return np.tile([0.35, 0.25], (p, 1))
    
    opt = _maxcut_exact(n, edges)
    if opt == 0:
        return np.zeros((p, 2))
    
    def objective(params):
        if _check_deadline():
            return 1e10
        gammas = params[:p]
        betas = params[p:]
        # Wrap angles to [0, 2*pi) for consistency
        gammas = np.mod(gammas, 2 * np.pi)
        betas = np.mod(betas, 2 * np.pi)
        try:
            ec = _expected_cut(n, edges, gammas, betas)
        except Exception:
            return 1e10
        return -ec / opt
    
    # Multiple starting points
    starts = [
        np.array([0.5, 0.5] * p),
        np.array([0.3, 0.3, 0.2, 0.2][:2*p]),
        np.array([0.8, 0.8, 0.1, 0.1][:2*p]),
        np.array([0.1, 0.1, 0.7, 0.7][:2*p]),
        np.array([0.5, 0.2, 0.3, 0.6][:2*p]),
    ]
    
    best_val = -1e10
    best_params = None
    
    for start in starts:
        if _check_deadline():
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
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
