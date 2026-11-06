import numpy as np
import time
import scipy.optimize

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    dim = 1 << n
    psi0 = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute cost values (number of cut edges) for each basis state
    costs = np.zeros(dim, dtype=float)
    for state in range(dim):
        cut_count = 0
        for u, v in edges:
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                cut_count += 1
        costs[state] = cut_count
    
    # Identify the optimal subspace (states with maximum cut)
    c_max = np.max(costs)
    opt_indices = np.where(costs == c_max)[0]
    n_opt = len(opt_indices)
    
    # Precompute indices for mixer to speed up vectorized application
    mixer_indices = []
    for i in range(n):
        bit = 1 << i
        idx0 = np.where((np.arange(dim) & bit) == 0)[0]
        idx1 = idx0 + bit
        mixer_indices.append((idx0, idx1))

    def _apply_mixer(psi, beta):
        """Apply exp(-i beta * sum X_i) efficiently using vectorized ops."""
        c = np.cos(beta)
        s = np.sin(beta)
        for idx0, idx1 in mixer_indices:
            a0 = psi[idx0]
            a1 = psi[idx1]
            psi[idx0] = c * a0 - 1j * s * a1
            psi[idx1] = -1j * s * a0 + c * a1
        return psi
    
    def _qaoa_state(gammas, betas):
        """Compute QAOA state for given angles."""
        psi = psi0.copy()
        for k in range(p):
            # Apply cost unitary: exp(-i gamma_k * H_C)
            psi = psi * np.exp(-1j * gammas[k] * costs)
            # Apply mixer unitary
            psi = _apply_mixer(psi, betas[k])
        return psi
    
    def _fidelity(psi):
        """Compute fidelity with the optimal subspace state."""
        return np.sum(np.abs(psi[opt_indices])**2) / n_opt
    
    def objective(params):
        gammas = params[0:p]
        betas = params[p:2*p]
        psi = _qaoa_state(gammas, betas)
        return -_fidelity(psi)
    
    # Initial guesses
    initial_guesses = [
        [0.5, 0.5, 0.5, 0.5],
        [0.3, 0.3, 0.3, 0.3],
        [0.7, 0.4, 0.7, 0.4],
        [0.2, 0.6, 0.2, 0.6],
        [1.0, 0.5, 1.0, 0.5],
        [0.4, 0.2, 0.4, 0.2],
        [0.1, 0.1, 0.1, 0.1],
        [0.8, 0.8, 0.8, 0.8],
        [0.25, 0.25, 0.25, 0.25],
        [0.15, 0.15, 0.15, 0.15],
    ]
    
    best_params = None
    best_val = np.inf
    
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline:
            break
        
        try:
            # Use scipy's Nelder-Mead
            result = scipy.optimize.minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
                
        except Exception:
            continue
    
    if best_params is None:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Final polish if time remains
    if time.time() < deadline - 5:
        try:
            result = scipy.optimize.minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass
    
    gammas = best_params[0:p]
    betas = best_params[p:2*p]
    return np.array(list(zip(gammas, betas)))
