import numpy as np
import time
import scipy.optimize

def _remaining(deadline):
    return max(0.0, deadline - time.time())

def _precompute_cuts(n, edges):
    """Precompute cost values (number of cut edges) for each basis state."""
    dim = 1 << n
    costs = np.zeros(dim, dtype=float)
    for state in range(dim):
        cut_count = 0
        for u, v in edges:
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                cut_count += 1
        costs[state] = cut_count
    return costs

def _precompute_mixer_indices(n, dim):
    """Precompute indices for the X-mixer application."""
    idx = np.arange(dim)
    masks = []
    partners_list = []
    for i in range(n):
        bit = 1 << i
        mask = (idx & bit) == 0
        partners = idx[mask] | bit
        masks.append(mask)
        partners_list.append(partners)
    return idx, masks, partners_list

def _apply_beta_fast(psi, beta, idx, masks, partners_list):
    """Apply exp(-i beta * sum X_i) using precomputed indices."""
    c = np.cos(beta)
    s = np.sin(beta)
    for i in range(len(masks)):
        mask = masks[i]
        partners = partners_list[i]
        a0 = psi[idx[mask]]
        a1 = psi[partners]
        psi[idx[mask]] = c * a0 - 1j * s * a1
        psi[partners] = -1j * s * a0 + c * a1
    return psi

def _expected_cut_fast(params, n, p, costs, dim, idx, masks, partners_list):
    """Compute expected cut value for given QAOA parameters using fast simulation."""
    gammas = params[0:p]
    betas = params[p:2*p]
    
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for k in range(p):
        psi = psi * np.exp(-1j * gammas[k] * costs)
        psi = _apply_beta_fast(psi, betas[k], idx, masks, partners_list)
    
    probs = np.abs(psi)**2
    return np.sum(probs * costs)

def _optimize_4params(n, edges, p, costs, dim, deadline):
    """Optimize the 4 QAOA parameters using grid search and local polish."""
    idx, masks, partners_list = _precompute_mixer_indices(n, dim)
    
    # 1. Coarse Grid Search
    # We search in [0, pi] for all parameters
    # Use 12 points per dimension -> 12^4 = 20736 evaluations
    # This is fast enough for n<=10
    
    num_grid = 12
    grid_vals = np.linspace(0.0, np.pi, num_grid, endpoint=False)
    
    best_cut = -np.inf
    best_params = None
    
    # Generate all combinations
    g1, g2, b1, b2 = np.meshgrid(grid_vals, grid_vals, grid_vals, grid_vals, indexing='ij')
    g1_flat = g1.ravel()
    g2_flat = g2.ravel()
    b1_flat = b1.ravel()
    b2_flat = b2.ravel()
    
    total_points = len(g1_flat)
    
    for i in range(total_points):
        if _remaining(deadline) < 10.0:
            break
            
        params = np.array([g1_flat[i], g2_flat[i], b1_flat[i], b2_flat[i]])
        
        cut_val = _expected_cut_fast(params, n, p, costs, dim, idx, masks, partners_list)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_params = params
    
    # 2. Local Polish
    if best_params is not None and _remaining(deadline) > 10.0:
        def objective(params):
            return -_expected_cut_fast(params, n, p, costs, dim, idx, masks, partners_list)
        
        try:
            res = scipy.optimize.minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            polish_cut = _expected_cut_fast(res.x, n, p, costs, dim, idx, masks, partners_list)
            if polish_cut > best_cut:
                best_cut = polish_cut
                best_params = res.x
        except Exception:
            pass
        
        # 3. Local High-Resolution Grid around the best point
        if _remaining(deadline) > 20.0:
            box_size = np.pi / 10.0
            low = np.clip(best_params - box_size, 0.0, np.pi)
            high = np.clip(best_params + box_size, 0.0, np.pi)
            
            # 8 points per dim -> 8^4 = 4096 points
            local_grid = np.linspace(low, high, 8, endpoint=True)
            lg1, lg2, lb1, lb2 = np.meshgrid(local_grid, local_grid, local_grid, local_grid, indexing='ij')
            lg1_flat = lg1.ravel()
            lg2_flat = lg2.ravel()
            lb1_flat = lb1.ravel()
            lb2_flat = lb2.ravel()
            
            for i in range(len(lg1_flat)):
                if _remaining(deadline) < 5.0:
                    break
                params = np.array([lg1_flat[i], lg2_flat[i], lb1_flat[i], lb2_flat[i]])
                cut_val = _expected_cut_fast(params, n, p, costs, dim, idx, masks, partners_list)
                if cut_val > best_cut:
                    best_cut = cut_val
                    best_params = params
            
            # Final Polish
            if _remaining(deadline) > 5.0:
                try:
                    res = scipy.optimize.minimize(
                        objective, best_params, method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    polish_cut = _expected_cut_fast(res.x, n, p, costs, dim, idx, masks, partners_list)
                    if polish_cut > best_cut:
                        best_cut = polish_cut
                        best_params = res.x
                except Exception:
                    pass

    if best_params is None:
        return np.array([[0.5, 0.5], [0.5, 0.5]])
    
    gammas = best_params[0:p]
    betas = best_params[p:2*p]
    return np.array(list(zip(gammas, betas)))

def angles(n, edges, p):
    start_time = time.time()
    deadline = start_time + 140.0
    
    dim = 1 << n
    costs = _precompute_cuts(n, edges)
    
    result = _optimize_4params(n, edges, p, costs, dim, deadline)
    
    return result
