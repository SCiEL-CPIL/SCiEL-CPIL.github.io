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
    
    # 4. Global grid search
    # Use a grid over [0, pi/2] for both gamma and beta.
    # This is a standard heuristic for QAOA p=2.
    # We use a slightly larger range to be safe: [0, pi] for gamma, [0, pi/2] for beta?
    # Let's stick to [0, pi/2] for both as it's the primary region.
    # To be safer, let's use [0, pi] for gamma and [0, pi/2] for beta.
    # Actually, symmetry suggests [0, pi/2] is sufficient for both in many cases.
    # Let's use a grid of 5 points in [0, pi/2] for each parameter.
    
    # To balance speed and coverage, we use a coarse grid first.
    n_grid = 5
    # Range: 0 to pi/2
    grid_points = np.linspace(0.0, np.pi/2, n_grid)
    
    best_params = None
    best_score = -np.inf
    top_candidates = [] # Store top K candidates
    
    K = 5 # Number of candidates to refine
    
    # Iterate over grid for p=2
    # 5^4 = 625 evaluations. This is fast.
    for g1 in grid_points:
        for b1 in grid_points:
            for g2 in grid_points:
                for b2 in grid_points:
                    params = np.array([g1, b1, g2, b2])
                    score = objective(params)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                    
                    # Keep top K
                    top_candidates.append((score, params.copy()))
    
    # Sort top candidates and take the best K
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = top_candidates[:K]
    
    # 5. High-precision Nelder-Mead refinement from top K grid points
    # This ensures we escape local minima traps better than a single start.
    global_best_params = best_params
    global_best_score = best_score
    
    for score, init_params in top_candidates:
        # Use Nelder-Mead
        # Set options for precision and iteration limits
        result = minimize(objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7,
                                   'adaptive': True})
        
        if result.fun < global_best_score:
            global_best_score = result.fun
            global_best_params = result.x
    
    # 6. Format output
    out = np.zeros((p, 2))
    for i in range(p):
        out[i, 0] = global_best_params[2*i]
        out[i, 1] = global_best_params[2*i + 1]
        
    return out

# EVOLVE-BLOCK-END
