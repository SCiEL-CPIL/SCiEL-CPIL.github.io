"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 145.0
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
    if n == 0:
        return 0
    adj = _build_adj(n, edges)
    best = 0
    for mask in range(1, 1 << n):
        bits = np.array([(mask >> i) & 1 for i in range(n)], dtype=np.float64)
        diff = 1 - bits[:, None] * bits[None, :]
        cut = np.sum(adj * diff) / 2.0
        if cut > best:
            best = cut
    return best

def _qaoa_expectation(n, edges, gammas, betas):
    """
    Compute the expected cut size for given QAOA angles using full state vector.
    """
    if n == 0:
        return 0.0
    
    m = len(edges)
    if m == 0:
        return 0.0
    
    edge_list = [(u, v) for u, v in edges]
    dim = 1 << n
    
    # Precompute cut values for all basis states
    indices = np.arange(dim)
    bits = ((indices[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    
    cut_val = np.zeros(dim)
    for i, j in edge_list:
        cut_val += (1 - bits[:, i] * bits[:, j]) / 2.0
    
    # Initialize state as |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Apply layers: for each k, apply exp(-i gamma_k H_C) then exp(-i beta_k H_B)
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i gamma H_C): diagonal in Z basis
        # H_C |z> = cut_val[z] |z>
        phase = np.exp(-1j * gamma * cut_val)
        state = state * phase
        
        if _check_deadline():
            break
        
        # Apply exp(-i beta H_B) where H_B = sum_i X_i
        # This is a product of single-qubit X rotations
        c = np.cos(beta)
        s = np.sin(beta)
        
        for q in range(n):
            mask = 1 << q
            # For each pair (z, z^mask) where z has bit q = 0
            # new[z] = c*state[z] - i*s*state[z^mask]
            # new[z^mask] = -i*s*state[z] + c*state[z^mask]
            # Vectorized approach using reshape
            # Reshape state to have q as the last axis of size 2
            shape = [1] * n
            shape[q] = 2
            # Actually, let's use a simpler indexing approach
            # Create arrays of indices where bit q is 0
            # z where (z & mask) == 0
            z0 = np.array([z for z in range(dim) if (z & mask) == 0])
            z1 = z0 | mask
            
            # Get current values
            v0 = state[z0].copy()
            v1 = state[z1].copy()
            
            # Apply rotation
            state[z0] = c * v0 - 1j * s * v1
            state[z1] = -1j * s * v0 + c * v1
    
    # Compute expectation value <H_C>
    probs = np.abs(state) ** 2
    return np.sum(probs * cut_val)

def _optimize_angles(n, edges, p):
    """
    Optimize QAOA angles for the given graph using multi-start Nelder-Mead.
    """
    m = len(edges)
    if m == 0:
        return np.zeros((p, 2))
    
    from scipy.optimize import minimize
    
    def objective(params):
        if _check_deadline():
            return 0.0
        g = params[:p]
        b = params[p:]
        return -_qaoa_expectation(n, edges, g, b)
    
    # Multiple initial guesses
    initial_guesses = [
        np.concatenate([np.full(p, 0.5), np.full(p, 0.5)]),
        np.concatenate([np.full(p, 0.1), np.full(p, 0.1)]),
        np.concatenate([np.full(p, 1.0), np.full(p, 0.5)]),
        np.concatenate([np.full(p, 0.3), np.full(p, 0.7)]),
        np.concatenate([np.full(p, 0.8), np.full(p, 0.3)]),
    ]
    
    best_x = initial_guesses[0]
    best_val = objective(best_x)
    
    options = {'maxiter': 200, 'fatol': 1e-6, 'xatol': 1e-6}
    
    for x0 in initial_guesses:
        if _check_deadline():
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options=options)
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    g_opt = best_x[:p]
    b_opt = best_x[p:]
    
    return np.column_stack([g_opt, b_opt])

def angles(n, edges, p):
    if _check_deadline():
        return np.tile([0.5, 0.5], (p, 1))
    
    if n <= 10:
        try:
            return _optimize_angles(n, edges, p)
        except Exception:
            pass
    
    return np.tile([0.5, 0.5], (p, 1))
# EVOLVE-BLOCK-END
