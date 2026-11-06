"""Combined program: fast state-vector simulation with multi-start Nelder-Mead + BFGS polish."""
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


def _precompute_flip_masks(n):
    """Precompute partner indices for X operations."""
    num_states = 1 << n
    states = np.arange(num_states, dtype=np.int64)
    masks = []
    for i in range(n):
        mask = 1 << i
        partner = states ^ mask
        masks.append(partner)
    return masks


def _expected_cut_value(n, gammas, betas, C_diag, flip_masks):
    """
    Compute the expected cut value for a given QAOA circuit.
    Uses exact state simulation with precomputed flip masks.
    """
    num_states = 1 << n
    psi = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    
    for k in range(len(gammas)):
        gamma = gammas[k]
        beta = betas[k]
        
        # Apply exp(-i*gamma*C): diagonal
        phase = np.exp(-1j * gamma * C_diag)
        psi = psi * phase
        
        # Apply exp(-i*beta*M) where M = sum_i X_i
        c = np.cos(beta)
        s = np.sin(beta)
        
        for i in range(n):
            partner = flip_masks[i]
            psi_partner = psi[partner].copy()
            psi = c * psi - 1j * s * psi_partner
    
    probs = np.abs(psi) ** 2
    expected_cut = np.sum(probs * C_diag)
    
    return expected_cut


def _exhaustive_maxcut(C_diag):
    """Compute the exact maximum cut."""
    return float(np.max(C_diag))


def _spectral_init(n, edges, p):
    """Generate initial angles using spectral heuristic for MaxCut."""
    # Build adjacency matrix
    A = np.zeros((n, n))
    for (u, v) in edges:
        A[u, v] = 1
        A[v, u] = 1
    
    # Compute leading eigenvector
    try:
        eigvals, eigvecs = np.linalg.eigh(A)
        # Take the eigenvector corresponding to the largest eigenvalue
        idx = np.argmax(eigvals)
        v = eigvecs[:, idx]
    except Exception:
        v = np.ones(n) / np.sqrt(n)
    
    # Use spectral gap to inform initial angles
    # For p=2, use heuristics based on graph density
    density = len(edges) / (n * (n - 1) / 2) if n > 1 else 0
    
    # Known good starting points for different densities
    if density < 0.3:
        g1, b1 = 0.3, 0.3
        g2, b2 = 0.4, 0.2
    elif density < 0.5:
        g1, b1 = 0.35, 0.25
        g2, b2 = 0.35, 0.25
    else:
        g1, b1 = 0.4, 0.2
        g2, b2 = 0.3, 0.3
    
    return np.array([g1, b1, g2, b2])


def _optimize_angles_for_graph(n, edges, p, time_budget=100.0):
    """Optimize QAOA angles for a specific graph using multi-start Nelder-Mead + BFGS."""
    start_time = time.time()
    
    C_diag = _build_cut_diag(n, edges)
    opt_cut = _exhaustive_maxcut(C_diag)
    if opt_cut == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    flip_masks = _precompute_flip_masks(n)
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        try:
            val = _expected_cut_value(n, gammas, betas, C_diag, flip_masks)
            return -val / opt_cut
        except Exception:
            return 1e10
    
    def elapsed():
        return time.time() - start_time
    
    # Multi-start optimization with diverse initial points
    # For p=2, we have 4 parameters: gamma1, beta1, gamma2, beta2
    
    # Define several starting points based on known good QAOA parameters
    start_points = [
        _spectral_init(n, edges, p),
        np.array([0.35, 0.25, 0.35, 0.25]),
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.2, 0.3, 0.4, 0.2]),
        np.array([0.6, 0.1, 0.3, 0.4]),
        np.array([0.1, 0.6, 0.5, 0.3]),
        np.array([0.8, 0.2, 0.2, 0.6]),
        np.array([0.4, 0.4, 0.6, 0.4]),
        np.array([0.3, 0.5, 0.2, 0.5]),
        np.array([0.25, 0.35, 0.45, 0.15]),
        np.array([0.45, 0.15, 0.25, 0.35]),
        np.array([0.15, 0.45, 0.35, 0.15]),
    ]
    
    # Add more random starts with better coverage
    rng = np.random.default_rng(42)
    for _ in range(12):
        g1 = rng.uniform(0.1, 1.2)
        b1 = rng.uniform(0.1, 1.0)
        g2 = rng.uniform(0.1, 1.2)
        b2 = rng.uniform(0.1, 1.0)
        start_points.append(np.array([g1, b1, g2, b2]))
    
    best_params = start_points[0].copy()
    best_val = -objective(best_params)
    
    time_limit = start_time + time_budget
    
    # First pass: quick Nelder-Mead from each start point
    for sp in start_points:
        if time.time() > time_limit - 15.0:
            break
        if elapsed() > time_budget * 0.55:
            break
            
        try:
            res = minimize(objective, sp, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7,
                                    'adaptive': True})
            if res.fun < objective(best_params):
                best_params = res.x.copy()
                best_val = -res.fun
        except Exception:
            pass
    
    # Second pass: refine the best solution with tighter tolerance
    if time.time() < time_limit - 10.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True})
            if res.fun < objective(best_params):
                best_params = res.x.copy()
                best_val = -res.fun
        except Exception:
            pass
    
    # Third pass: very tight polish if time permits
    if time.time() < time_limit - 5.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < objective(best_params):
                best_params = res.x.copy()
                best_val = -res.fun
        except Exception:
            pass
    
    # Final polish with BFGS if available and time permits
    if time.time() < time_limit - 2.0:
        try:
            res = minimize(objective, best_params, method='BFGS',
                           options={'maxiter': 300, 'gtol': 1e-12, 'disp': False})
            if res.fun < objective(best_params):
                best_params = res.x.copy()
                best_val = -res.fun
        except Exception:
            pass
    
    # One more ultra-tight Nelder-Mead polish
    if time.time() < time_limit - 1.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14,
                                    'adaptive': True})
            if res.fun < objective(best_params):
                best_params = res.x.copy()
                best_val = -res.fun
        except Exception:
            pass
    
    best_angles = np.clip(best_params, -np.pi, np.pi)
    
    result = np.zeros((p, 2))
    result[:, 0] = best_angles[:p]
    result[:, 1] = best_angles[p:]
    
    return result


def angles(n, edges, p):
    start_time = time.time()
    time_budget = 140.0  # Leave 10s margin for 150s limit
    
    try:
        result = _optimize_angles_for_graph(n, edges, p, time_budget=time_budget)
    except Exception:
        result = np.tile([0.35, 0.25], (p, 1))
    
    return result
# EVOLVE-BLOCK-END
