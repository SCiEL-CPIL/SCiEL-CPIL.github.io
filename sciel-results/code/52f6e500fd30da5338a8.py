"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _build_cut_diag(n, edges):
    """Precompute the cut diagonal for all 2^n states using vectorized bit ops."""
    num_states = 1 << n
    states = np.arange(num_states, dtype=np.int64)
    C_diag = np.zeros(num_states, dtype=float)
    for (u, v) in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        C_diag += (bits_u != bits_v).astype(float)
    return C_diag


def _expected_cut_value(n, C_diag, gammas, betas):
    """
    Compute the expected cut value for a given QAOA circuit.
    Uses exact state simulation for small n.
    """
    num_states = 1 << n
    psi = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    # Apply layers
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i*gamma*C): diagonal
        phase = np.exp(-1j * gamma * C_diag)
        psi = psi * phase
        
        # Apply exp(-i*beta*M) = product_i (cos(beta)*I - i*sin(beta)*X_i)
        c = np.cos(beta)
        s = np.sin(beta)
        
        for i in range(n):
            mask = 1 << i
            # Vectorized application of the 2x2 rotation to pairs of states
            idx0 = np.arange(0, num_states, 2 * mask)[::1] if mask == 1 else None
            # More efficient: use reshape and vector operations
            # States where bit i is 0: (state & mask) == 0
            # We can use numpy to extract pairs
            # For each block of size 2*mask, the first mask states have bit i = 0
            # and the next mask states have bit i = 1
            
            # Use a more efficient approach with indexing
            if mask == 1:
                # Alternate pairs
                psi0 = psi[0::2]
                psi1 = psi[1::2]
                new_psi0 = c * psi0 - 1j * s * psi1
                new_psi1 = -1j * s * psi0 + c * psi1
                psi[0::2] = new_psi0
                psi[1::2] = new_psi1
            else:
                # For larger masks, process in blocks
                block_size = 2 * mask
                for block_start in range(0, num_states, block_size):
                    psi0 = psi[block_start:block_start + mask]
                    psi1 = psi[block_start + mask:block_start + 2 * mask]
                    new_psi0 = c * psi0 - 1j * s * psi1
                    new_psi1 = -1j * s * psi0 + c * psi1
                    psi[block_start:block_start + mask] = new_psi0
                    psi[block_start + mask:block_start + 2 * mask] = new_psi1
    
    # Compute expected cut
    probs = np.abs(psi) ** 2
    expected_cut = np.sum(probs * C_diag)
    
    return expected_cut


def _exhaustive_maxcut(n, C_diag):
    """Compute the exact maximum cut."""
    return np.max(C_diag)


def _spectral_init(n, edges, p):
    """Generate initial angles using spectral/degree-based heuristics."""
    # Build adjacency matrix
    A = np.zeros((n, n))
    for (u, v) in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    
    # Compute degrees
    degrees = A.sum(axis=1)
    
    # Simple heuristic: gamma based on average degree, beta fixed
    avg_deg = np.mean(degrees) if len(degrees) > 0 else 1.0
    gamma0 = 0.5 / (1.0 + avg_deg)
    beta0 = 0.3
    
    gammas = np.full(p, gamma0)
    betas = np.full(p, beta0)
    
    return np.concatenate([gammas, betas])


def _optimize_angles_for_graph(n, edges, p, time_budget=100.0):
    """Optimize QAOA angles for a specific graph using multi-start BFGS."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Precompute
    C_diag = _build_cut_diag(n, edges)
    opt_cut = _exhaustive_maxcut(n, C_diag)
    
    if opt_cut == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        try:
            val = _expected_cut_value(n, C_diag, gammas, betas)
            ratio = val / opt_cut
            return -ratio
        except Exception:
            return 1e10
    
    # Multi-start optimization
    best_params = None
    best_value = -np.inf
    
    # Generate multiple starting points
    starts = []
    
    # Spectral initialization
    starts.append(_spectral_init(n, edges, p))
    
    # Fixed reasonable defaults
    starts.append(np.array([0.35, 0.25] * p))
    
    # Random starts
    rng = np.random.RandomState(42)
    for _ in range(3):
        gammas = rng.uniform(0.1, 1.0, p)
        betas = rng.uniform(0.1, 0.8, p)
        starts.append(np.concatenate([gammas, betas]))
    
    # Try each starting point
    for x0 in starts:
        if time.time() > deadline - 5.0:
            break
        
        try:
            result = minimize(objective, x0, method='BFGS', 
                             options={'maxiter': 50, 'gtol': 1e-5})
            
            if result.success or result.fun < best_value:
                if result.fun < best_value:
                    best_value = result.fun
                    best_params = result.x.copy()
        except Exception:
            continue
    
    # If no optimization succeeded, use best initial guess
    if best_params is None:
        best_params = starts[0]
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 3.0:
        try:
            result = minimize(objective, best_params, method='BFGS',
                             options={'maxiter': 30, 'gtol': 1e-7})
            if result.fun < best_value:
                best_params = result.x.copy()
                best_value = result.fun
        except Exception:
            pass
    
    # Clip angles to reasonable range
    best_params = np.clip(best_params, -np.pi, np.pi)
    
    # Format output
    result_arr = np.zeros((p, 2))
    result_arr[:, 0] = best_params[:p]  # gammas
    result_arr[:, 1] = best_params[p:]  # betas
    
    return result_arr


def angles(n, edges, p):
    # Time guard: must finish within 120s total
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin
    
    try:
        result = _optimize_angles_for_graph(n, edges, p, time_budget=time_budget)
    except Exception:
        # Fallback to reasonable default
        result = np.tile([0.35, 0.25], (p, 1))
    
    return result
# EVOLVE-BLOCK-END
