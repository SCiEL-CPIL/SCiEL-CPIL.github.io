"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _build_cost_vector(n, edges):
    """
    Precompute the cut value for all 2^n states.
    Returns a vector E of length 2^n where E[s] is the number of cut edges for state s.
    """
    num_states = 2 ** n
    states = np.arange(num_states)
    E = np.zeros(num_states, dtype=np.int32)
    
    for u, v in edges:
        # Get bits for u and v for all states
        bu = (states >> u) & 1
        bv = (states >> v) & 1
        # If bits are different, it's a cut.
        E += (bu != bv)
        
    return E.astype(np.float64)

def _apply_mixing(state, beta, n, states):
    """
    Apply exp(-i beta sum X_i) to the state vector.
    """
    c = np.cos(beta)
    s = np.sin(beta)
    # We apply the operator for each bit i
    # U = prod_i (c I - i s X_i)
    # This can be done iteratively
    for i in range(n):
        # Flip bit i
        flipped = state[states ^ (1 << i)]
        state = c * state - 1j * s * flipped
    return state

def _evaluate_angles(params, E, n, states):
    """
    Compute the expected cut value for given angles (g1, b1, g2, b2).
    """
    g1, b1, g2, b2 = params
    num_states = 2 ** n
    c = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Layer 1
    c = c * np.exp(-1j * g1 * E)
    c = _apply_mixing(c, b1, n, states)
    
    # Layer 2
    c = c * np.exp(-1j * g2 * E)
    c = _apply_mixing(c, b2, n, states)
    
    # Expectation value
    probs = np.abs(c) ** 2
    val = np.sum(probs * E)
    return val

def angles(n, edges, p):
    # 1. Precompute
    E = _build_cost_vector(n, edges)
    num_states = 2 ** n
    states = np.arange(num_states)
    
    # 2. Coarse Grid Search
    # Angles are periodic. For p=2, optimal angles often lie in [0, pi] for gamma and [0, pi/2] for beta.
    # However, to be safe, we scan [0, pi] for both.
    # Grid resolution: 10 points per dimension -> 10^4 = 10,000 evaluations.
    # Each eval is ~O(n 2^n). For n=10, ~10k ops. Total ~10^8 ops.
    # This might be slow in Python if not vectorized well, but _evaluate_angles is vectorized over states.
    # Let's try 8 points per dimension to be safe on time. 8^4 = 4096.
    
    gamma_grid = np.linspace(0, np.pi, 8)
    beta_grid = np.linspace(0, np.pi, 8)
    
    best_val = -np.inf
    best_params = None
    
    # Iterate over grid
    for g1 in gamma_grid:
        for b1 in beta_grid:
            for g2 in gamma_grid:
                for b2 in beta_grid:
                    params = np.array([g1, b1, g2, b2])
                    val = _evaluate_angles(params, E, n, states)
                    if val > best_val:
                        best_val = val
                        best_params = params.copy()
                        
    # 3. Local Refinement with Nelder-Mead
    # Use the best grid point as initial guess
    x0 = best_params
    bounds = [(0, 2 * np.pi)] * 4
    
    def objective(params):
        return -_evaluate_angles(params, E, n, states)
        
    res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, 
                   options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 2000, 'maxfev': 5000})
    
    opt_angles = res.x
    
    # Reshape to (p, 2)
    out = np.array([
        [opt_angles[0], opt_angles[1]],
        [opt_angles[2], opt_angles[3]]
    ])
    
    return out

# EVOLVE-BLOCK-END
