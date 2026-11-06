"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _build_cut_diag(n, edges):
    """Precompute the cut diagonal for all 2^n states using vectorized bit ops."""
    num_states = 1 << n
    # Create bit indices for all states
    states = np.arange(num_states, dtype=np.int64)
    C_diag = np.zeros(num_states, dtype=float)
    for (u, v) in edges:
        # Extract bits u and v from all states
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        # Count edges that are cut (bits differ)
        C_diag += (bits_u != bits_v).astype(float)
    return C_diag


def _expected_cut_value(n, gammas, betas, C_diag):
    """
    Compute the expected cut value for a given QAOA circuit.
    Uses exact state simulation with vectorized operations.
    """
    num_states = 1 << n
    # Start in |+>^n = uniform superposition
    psi = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    # Precompute all states once
    states = np.arange(num_states, dtype=np.int64)
    
    # Apply layers
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i*gamma*C): diagonal
        phase = np.exp(-1j * gamma * C_diag)
        psi = psi * phase
        
        # Apply exp(-i*beta*M) where M = sum_i X_i
        # This is a product of single-qubit rotations
        c = np.cos(beta)
        s = np.sin(beta)
        
        for i in range(n):
            mask = 1 << i
            partner = states ^ mask
            psi_partner = psi[partner]
            psi_new = c * psi - 1j * s * psi_partner
            psi = psi_new
    
    # Compute expected cut
    probs = np.abs(psi) ** 2
    expected_cut = np.sum(probs * C_diag)
    
    return expected_cut


def _exhaustive_maxcut(n, edges, C_diag):
    """Compute the exact maximum cut."""
    return int(np.max(C_diag))


def _nelder_mead(func, x0, max_iter=200, tol=1e-8, time_limit=None):
    """Nelder-Mead optimization using scipy."""
    result = minimize(func, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
    return result.x, result.fun


def _optimize_angles_for_graph(n, edges, p, time_budget=50.0):
    """Optimize QAOA angles using global grid search followed by Nelder-Mead refinement."""
    start_time = time.time()
    time_limit = start_time + time_budget
    
    # Precompute cut diagonal
    C_diag = _build_cut_diag(n, edges)
    opt_cut = _exhaustive_maxcut(n, edges, C_diag)
    if opt_cut == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    # Define objective function
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        try:
            val = _expected_cut_value(n, gammas, betas, C_diag)
            return -val / opt_cut  # Minimize negative for maximization
        except Exception:
            return 1e10
    
    # Phase 1: Global grid search over coarsely spaced parameter space
    # Scale grid based on graph size - larger graphs may need different ranges
    if n <= 6:
        gamma_range = np.linspace(0.1, 1.2, 8)
        beta_range = np.linspace(0.05, 0.8, 8)
    elif n <= 8:
        gamma_range = np.linspace(0.1, 1.0, 7)
        beta_range = np.linspace(0.05, 0.7, 7)
    else:
        gamma_range = np.linspace(0.1, 0.8, 6)
        beta_range = np.linspace(0.05, 0.6, 6)
    
    best_grid_value = -1
    best_grid_params = None
    
    # Grid search over all combinations
    for g0 in gamma_range:
        if time.time() > time_limit - 5.0:
            break
        for b0 in beta_range:
            if time.time() > time_limit - 5.0:
                break
            for g1 in gamma_range:
                if time.time() > time_limit - 5.0:
                    break
                for b1 in beta_range:
                    params = np.array([g0, b0, g1, b1])
                    val = objective(params)
                    if val < best_grid_value:
                        best_grid_value = val
                        best_grid_params = params.copy()
    
    # If grid search found nothing, use a reasonable default
    if best_grid_params is None:
        best_grid_params = np.array([0.35, 0.25, 0.35, 0.25])
        best_grid_value = objective(best_grid_params)
    
    # Phase 2: Single high-precision Nelder-Mead refinement from best grid point
    if time.time() < time_limit - 10.0:
        try:
            x_opt, f_opt = _nelder_mead(objective, best_grid_params, max_iter=500, tol=1e-10)
            if f_opt < best_grid_value:
                best_grid_value = f_opt
                best_grid_params = x_opt
        except Exception:
            pass
    
    # Phase 3: Final polish with even tighter tolerance if time allows
    if time.time() < time_limit - 5.0:
        try:
            x_opt, f_opt = _nelder_mead(objective, best_grid_params, max_iter=200, tol=1e-12)
            if f_opt < best_grid_value:
                best_grid_value = f_opt
                best_grid_params = x_opt
        except Exception:
            pass
    
    # Ensure angles are in reasonable range
    best_grid_params = np.clip(best_grid_params, -np.pi, np.pi)
    
    # Format output
    result = np.zeros((p, 2))
    result[:, 0] = best_grid_params[:p]  # gammas
    result[:, 1] = best_grid_params[p:]  # betas
    
    return result


def angles(n, edges, p):
    # Time guard: must finish within 120s total
    start_time = time.time()
    time_budget = 110.0  # Leave 10s margin for safety
    
    try:
        result = _optimize_angles_for_graph(n, edges, p, time_budget=time_budget)
    except Exception:
        # Fallback to reasonable default
        result = np.tile([0.35, 0.25], (p, 1))
    
    # Final time check
    elapsed = time.time() - start_time
    if elapsed > 115.0:
        # Should not happen, but return what we have
        pass
    
    return result
# EVOLVE-BLOCK-END
