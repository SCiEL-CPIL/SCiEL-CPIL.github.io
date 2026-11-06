# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import product

def _exact_maxcut(n, edges):
    """Compute the exact maximum cut value for a graph with n nodes and given edges."""
    m = len(edges)
    if m == 0:
        return 0.0
    
    best_val = 0.0
    for mask in range(1 << n):
        val = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                val += 1
        if val > best_val:
            best_val = val
    return float(best_val)


def _build_cost_diag(n, edges):
    """Build the diagonal of the cost matrix C for the graph."""
    dim = 1 << n
    c_diag = np.zeros(dim, dtype=np.float64)
    
    for s in range(dim):
        val = 0
        for u, v in edges:
            if ((s >> u) & 1) != ((s >> v) & 1):
                val += 1
        c_diag[s] = val
        
    return c_diag


def _expected_cut_fast(n, c_diag, gammas, betas):
    """
    Compute expected cut given precomputed C diagonal.
    Uses efficient tensor operations for X-rotations.
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    p = len(gammas)
    
    for i in range(p):
        # Apply exp(-i gamma_i C)
        state = state * np.exp(-1j * gammas[i] * c_diag)
        
        # Apply exp(-i beta_i H_B)
        beta = betas[i]
        c_beta = np.cos(beta)
        s_beta = -1j * np.sin(beta)
        
        # Reshape state to tensor of shape (2, 2, ..., 2)
        shape = [2] * n
        tensor = state.reshape(shape)
        
        # Apply rotation on each axis
        for k in range(n):
            tensor = np.moveaxis(tensor, k, 0)
            
            comp0 = tensor[0].copy()
            comp1 = tensor[1].copy()
            
            tensor[0] = c_beta * comp0 + s_beta * comp1
            tensor[1] = s_beta * comp0 + c_beta * comp1
            
            tensor = np.moveaxis(tensor, 0, k)
        
        state = tensor.reshape(dim)
        
    # Compute expected cut
    probs = np.abs(state) ** 2
    expected_cut = np.dot(probs, c_diag)
    
    return float(np.real(expected_cut))


def angles(n, edges, p):
    # 1. Compute exact optimum
    opt_val = _exact_maxcut(n, edges)
    if opt_val == 0:
        return np.tile([0.0, 0.0], (p, 1))
    
    # 2. Build cost diagonal
    c_diag = _build_cost_diag(n, edges)
    
    # 3. Define objective function for minimization (we want to maximize ratio)
    def objective(params):
        gammas = params[0::2]
        betas = params[1::2]
        
        exp_cut = _expected_cut_fast(n, c_diag, gammas, betas)
        ratio = exp_cut / opt_val
        return -ratio
    
    # 4. Global grid search with two passes: coarse then fine
    # Coarse grid: 7 points in [0, pi/2] for both gamma and beta
    n_coarse = 7
    grid_coarse = np.linspace(0.0, np.pi/2, n_coarse)
    
    best_params = None
    best_score = -np.inf
    top_candidates = []
    
    K = 10  # Number of candidates to keep
    
    # Coarse grid search
    for g1 in grid_coarse:
        for b1 in grid_coarse:
            for g2 in grid_coarse:
                for b2 in grid_coarse:
                    params = np.array([g1, b1, g2, b2])
                    score = objective(params)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    top_candidates.append((score, params.copy()))
    
    # Sort and keep top K
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = top_candidates[:K]
    
    # Fine grid search around top candidates
    # For each top candidate, create a fine grid in a neighborhood
    fine_candidates = []
    neighborhood = np.pi / 20  # Small neighborhood radius
    
    for score, center in top_candidates[:5]:  # Only refine top 5
        # Create fine grid points around center
        offsets = np.linspace(-neighborhood, neighborhood, 5)
        fine_grid = center + np.tile(offsets, (4, 1)).T
        
        # Ensure parameters stay in [0, pi/2]
        fine_grid = np.clip(fine_grid, 0.0, np.pi/2)
        
        for params in fine_grid:
            s = objective(params)
            fine_candidates.append((s, params.copy()))
    
    # Combine and sort all candidates
    all_candidates = top_candidates + fine_candidates
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    all_candidates = all_candidates[:K]
    
    # 5. High-precision Nelder-Mead refinement from top K grid points
    global_best_params = best_params
    global_best_score = best_score
    
    for score, init_params in all_candidates:
        # Use Nelder-Mead with tight tolerances
        result = minimize(objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True, 'initial_simplex': None})
        
        # Clip parameters to valid range
        result.x = np.clip(result.x, 0.0, np.pi/2)
        
        final_score = objective(result.x)
        
        if final_score < global_best_score:
            global_best_score = final_score
            global_best_params = result.x
    
    # 6. Format output
    out = np.zeros((p, 2))
    for i in range(p):
        out[i, 0] = global_best_params[2*i]
        out[i, 1] = global_best_params[2*i + 1]
        
    return out

# EVOLVE-BLOCK-END
