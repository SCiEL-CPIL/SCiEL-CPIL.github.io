"""QAOA p=2 graph-dependent angle optimizer with vectorized state evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin below 120s wall-clock limit


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _vectorized_expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut value for p=2 QAOA using vectorized operations.
    
    E[cut] = sum_{(u,v) in E} (1 - <Z_u Z_v>)/2
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute Z values for each qubit: z_i[state] = 1 if bit i is 0, -1 if bit i is 1
    z_vals = np.ones((n, dim), dtype=np.float64)
    for i in range(n):
        for state in range(dim):
            z_vals[i, state] = 1.0 if ((state >> i) & 1) == 0 else -1.0
    
    for layer in range(2):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply exp(-i gamma * C) = prod_{(u,v) in E} exp(-i gamma/2 * (1 - Z_u Z_v))
        # = exp(-i gamma * m/2) * prod_{(u,v) in E} exp(i gamma/2 * Z_u Z_v)
        # Global phase doesn't matter, so apply phase per edge
        phase = np.ones(dim, dtype=complex)
        for u, v in edges:
            zu = z_vals[u]
            zv = z_vals[v]
            zz = zu * zv  # +1 if same, -1 if different
            # exp(i * gamma/2 * zz) where zz is +1 or -1
            phase *= np.where(zz > 0, np.exp(1j * gamma / 2), np.exp(-1j * gamma / 2))
        psi *= phase
        
        # Apply exp(-i beta * sum_i X_i) = prod_i exp(-i beta * X_i)
        # For each qubit i: |0><0| cos(b) - i sin(b)|0><1| - i sin(b)|1><0| + cos(b)|1><1|
        # Vectorized: for each qubit, pair states differing in bit i
        for i in range(n):
            mask = 1 << i
            # Create index arrays for pairing
            # States where bit i is 0 and their partners where bit i is 1
            idx0 = []
            idx1 = []
            for state in range(dim):
                if ((state >> i) & 1) == 0:
                    idx0.append(state)
                    idx1.append(state | mask)
            idx0 = np.array(idx0, dtype=np.int64)
            idx1 = np.array(idx1, dtype=np.int64)
            
            c = np.cos(beta)
            s = np.sin(beta)
            
            psi0 = psi[idx0]
            psi1 = psi[idx1]
            
            new_psi0 = c * psi0 - 1j * s * psi1
            new_psi1 = -1j * s * psi0 + c * psi1
            
            psi[idx0] = new_psi0
            psi[idx1] = new_psi1
    
    # Compute expected cut: sum_{(u,v) in E} (1 - <Z_u Z_v>)/2
    total = 0.0
    for u, v in edges:
        zu = z_vals[u]
        zv = z_vals[v]
        zz = zu * zv
        prob = np.abs(psi) ** 2
        corr = np.sum(prob * zz)
        total += (1.0 - corr) / 2.0
    
    return total


def _exhaustive_maxcut(n, edges):
    """Compute exact MaxCut by exhaustive search."""
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best


def _objective(params, n, edges):
    """Objective to minimize: negative expected cut."""
    gammas = params[:2]
    betas = params[2:4]
    return -_vectorized_expected_cut(n, edges, gammas, betas)


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    Returns (p, 2) array with rows (gamma, beta) in radians.
    """
    adj = np.zeros((n, n), dtype=float)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    
    m = len(edges)
    
    # Multi-start optimization
    rng = np.random.RandomState(113)
    
    # Generate diverse initial points
    candidates = []
    
    # Known good region for p=2 QAOA
    candidates.append(np.array([0.3, 0.15, 0.2, 0.1]))
    candidates.append(np.array([0.5, 0.2, 0.3, 0.15]))
    candidates.append(np.array([0.4, 0.25, 0.25, 0.12]))
    
    # Add random starts
    for _ in range(15):
        g1 = rng.uniform(0.1, 0.8)
        b1 = rng.uniform(0.05, 0.4)
        g2 = rng.uniform(0.05, 0.6)
        b2 = rng.uniform(0.03, 0.3)
        candidates.append(np.array([g1, b1, g2, b2]))
    
    best_params = None
    best_val = -np.inf
    
    for params in candidates:
        if _remaining() < 5.0:
            break
        val = _vectorized_expected_cut(n, edges, params[:2], params[2:4])
        if val > best_val:
            best_val = val
            best_params = params.copy()
    
    if best_params is None:
        best_params = np.array([0.3, 0.15, 0.2, 0.1])
        best_val = _vectorized_expected_cut(n, edges, best_params[:2], best_params[2:4])
    
    # Local refinement with BFGS
    if _remaining() > 10.0:
        try:
            res = minimize(
                _objective,
                best_params,
                args=(n, edges),
                method='BFGS',
                options={'maxiter': 80, 'gtol': 1e-7}
            )
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Further refinement with Nelder-Mead if time permits
    if _remaining() > 15.0:
        try:
            res = minimize(
                _objective,
                best_params,
                args=(n, edges),
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Format output
    out = np.zeros((p, 2))
    for i in range(min(p, 2)):
        out[i, 0] = best_params[i]
        out[i, 1] = best_params[2 + i]
    
    return out
# EVOLVE-BLOCK-END
