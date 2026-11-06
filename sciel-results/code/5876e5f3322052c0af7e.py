"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def _build_cost_vector(n, edges):
    """
    Precompute the cut value for all 2^n states.
    Returns a vector E of length 2^n where E[s] is the number of cut edges for state s.
    """
    num_states = 2 ** n
    states = np.arange(num_states)
    E = np.zeros(num_states, dtype=np.float64)
    
    for u, v in edges:
        bu = (states >> u) & 1
        bv = (states >> v) & 1
        E += (bu != bv)
        
    return E

def _evaluate_angles(params, E, n, states):
    """
    Compute the expected cut value for given angles (g1, b1, g2, b2).
    Optimized state vector simulation.
    """
    g1, b1, g2, b2 = params
    num_states = 2 ** n
    
    # Start in |+>^n
    # c = [1/sqrt(2^n)] * 2^n
    # Instead of creating a new array, we can work with the vector directly.
    c = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
    
    # Helper to apply mixing in-place or with minimal allocation
    def apply_mixing(state, beta):
        c_val = np.cos(beta)
        s_val = np.sin(beta)
        # We need to update state in a way that doesn't use old values for the flip
        # state_new[i] = c * state[i] - i * s * state[i ^ (1<<k)]
        # This is a permutation + scaling.
        # We can do this bit by bit.
        for k in range(n):
            mask = 1 << k
            # Indices where bit k is 0: i % (2*mask) < mask
            # Indices where bit k is 1: i % (2*mask) >= mask
            # Actually, simpler:
            # Let A be indices with bit k = 0.
            # Let B be indices with bit k = 1.
            # B = A ^ mask.
            # new_state[A] = c * state[A] - i * s * state[B]
            # new_state[B] = c * state[B] - i * s * state[A]
            
            # Create boolean mask for bit k being 0
            # (states >> k) & 1 == 0
            mask_zero = ((states >> k) & 1) == 0
            idx_zero = np.where(mask_zero)[0]
            idx_one = idx_zero ^ mask
            
            # Save old values to avoid overwrite issues
            # Since we are modifying in place, we need temp arrays for the swapped parts
            # Or we can compute the new values for idx_zero and idx_one separately
            v_zero = state[idx_zero]
            v_one = state[idx_one]
            
            state[idx_zero] = c_val * v_zero - 1j * s_val * v_one
            state[idx_one] = c_val * v_one - 1j * s_val * v_zero
        return state

    # Layer 1
    c *= np.exp(-1j * g1 * E)
    c = apply_mixing(c, b1)
    
    # Layer 2
    c *= np.exp(-1j * g2 * E)
    c = apply_mixing(c, b2)
    
    # Expectation value
    probs = np.abs(c) ** 2
    val = np.sum(probs * E)
    return val

def angles(n, edges, p):
    E = _build_cost_vector(n, edges)
    num_states = 2 ** n
    states = np.arange(num_states)
    
    # Objective function for scipy
    def objective(params):
        return -_evaluate_angles(params, E, n, states)
        
    bounds = [(0, 2 * np.pi)] * 4
    
    # 1. Global Optimization with Differential Evolution
    # DE is robust to multi-modality and doesn't require a good initial guess.
    # For n<=10, the evaluation is fast, so we can afford a decent number of iterations.
    # popsize=15, maxiter=100 is usually sufficient for 4D problems.
    try:
        res_de = differential_evolution(objective, bounds, 
                                        maxiter=100, 
                                        popsize=15,
                                        tol=1e-6,
                                        seed=42,
                                        polish=False) # We will polish manually
        best_params = res_de.x
    except Exception:
        # Fallback to a random point if DE fails
        best_params = np.random.uniform(0, 2*np.pi, 4)
        
    # 2. Local Refinement with Nelder-Mead
    # Polish the result from DE to ensure high precision.
    res_nm = minimize(objective, best_params, method='Nelder-Mead', 
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 5000})
    
    opt_angles = res_nm.x
    
    # Reshape to (p, 2)
    out = np.array([
        [opt_angles[0], opt_angles[1]],
        [opt_angles[2], opt_angles[3]]
    ])
    
    return out

# EVOLVE-BLOCK-END
