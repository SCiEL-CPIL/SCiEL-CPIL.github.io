import numpy as np
import time
from scipy.optimize import minimize, dual_annealing

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
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
    
    # Precompute index pairs for mixer: for each qubit i, pairs of states (state, state|bit)
    # where bit i is 0
    mixer_pairs = []
    for i in range(n):
        bit = 1 << i
        # States where bit i is 0
        idx0 = np.array([s for s in range(dim) if (s & bit) == 0], dtype=np.int64)
        idx1 = idx0 | bit
        mixer_pairs.append((idx0, idx1))
    
    def _apply_mixer(psi, beta, n):
        """Apply exp(-i beta * sum X_i) using vectorized operations."""
        c = np.cos(beta)
        s = np.sin(beta)
        for idx0, idx1 in mixer_pairs:
            a0 = psi[idx0]
            a1 = psi[idx1]
            psi[idx0] = c * a0 - 1j * s * a1
            psi[idx1] = -1j * s * a0 + c * a1
        return psi
    
    def _qaoa_state(gammas, betas):
        """Compute QAOA state for given angles."""
        psi = psi0.copy()
        for k in range(p):
            psi = psi * np.exp(-1j * gammas[k] * costs)
            psi = _apply_mixer(psi, betas[k], n)
        return psi
    
    def _expected_cut(psi):
        """Compute expected cut from state vector."""
        probs = np.abs(psi)**2
        return np.dot(probs, costs)
    
    def objective(params):
        gammas = params[0:p]
        betas = params[p:2*p]
        psi = _qaoa_state(gammas, betas)
        return -_expected_cut(psi)
    
    # Physics-informed initial guesses based on QAOA literature
    # For MaxCut, typical optimal angles are gamma in [0.3, 0.9], beta in [0.1, 0.5]
    initial_guesses = [
        [0.55, 0.35, 0.45, 0.25],
        [0.70, 0.40, 0.50, 0.30],
        [0.40, 0.30, 0.35, 0.20],
        [0.80, 0.45, 0.55, 0.35],
        [0.30, 0.25, 0.25, 0.15],
        [0.60, 0.50, 0.40, 0.30],
        [0.45, 0.20, 0.35, 0.15],
        [0.90, 0.35, 0.60, 0.25],
        [0.50, 0.40, 0.30, 0.20],
        [0.65, 0.30, 0.50, 0.20],
        [0.35, 0.45, 0.25, 0.30],
        [0.75, 0.25, 0.55, 0.15],
        # Additional diverse starts
        [0.50, 0.50, 0.40, 0.40],
        [0.85, 0.20, 0.65, 0.10],
        [0.42, 0.42, 0.32, 0.32],
        [0.72, 0.28, 0.52, 0.18],
    ]
    
    best_params = None
    best_val = np.inf
    
    # Phase 1: Global search with dual_annealing if time permits
    if time.time() < deadline - 60:
        try:
            res = dual_annealing(objective, [(-1, 1)] * (2*p), maxiter=20, time_limit=25)
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple starts using Powell with more iterations
    starts = initial_guesses.copy()
    if best_params is not None:
        starts.append(best_params)
        
    for x0 in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 300, 'maxfev': 2000, 'xtol': 1e-8, 'ftol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue
            
    # Phase 3: BFGS polish with numerical gradients for faster convergence
    if best_params is not None and time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='BFGS', options={'maxiter': 200, 'gtol': 1e-10, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Phase 4: Final ultra-precise Nelder-Mead polish
    if best_params is not None and time.time() < deadline - 3:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        return np.array([[0.55, 0.35], [0.45, 0.25]])
    
    gammas = best_params[0:p]
    betas = best_params[p:2*p]
    return np.array(list(zip(gammas, betas)))
