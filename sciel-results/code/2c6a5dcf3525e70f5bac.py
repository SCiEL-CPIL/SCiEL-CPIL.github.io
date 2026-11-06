"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import dual_annealing

_DEADLINE = 150.0
_SAFETY = 15.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _T0)

def _precompute_cuts(n, edges):
    """Precompute cut values for all 2^n basis states."""
    num_states = 2**n
    cuts = np.zeros(num_states, dtype=np.float64)
    states = np.arange(num_states, dtype=np.int64)
    for u, v in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        mask = bits_u != bits_v
        cuts[mask] += 1
    return cuts

def _expected_cut(n, cuts, gamma1, beta1, gamma2, beta2):
    """
    Compute expected cut for 2-layer QAOA.
    Optimized: uses precomputed X matrices for speed.
    """
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1
    psi = psi * np.exp(-1j * gamma1 * cuts)
    if abs(beta1) > 1e-9:
        c = np.cos(beta1)
        s = np.sin(beta1)
        for k in range(n):
            shape = (2**(n-k-1), 2, 2**k)
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :].copy()
            p1 = psi_reshaped[:, 1, :].copy()
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)
            
    # Layer 2
    psi = psi * np.exp(-1j * gamma2 * cuts)
    if abs(beta2) > 1e-9:
        c = np.cos(beta2)
        s = np.sin(beta2)
        for k in range(n):
            shape = (2**(n-k-1), 2, 2**k)
            psi_reshaped = psi.reshape(shape)
            p0 = psi_reshaped[:, 0, :].copy()
            p1 = psi_reshaped[:, 1, :].copy()
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)

    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _optimize_4params(n, cuts):
    """
    Optimize 4 QAOA parameters using Dual Annealing (Global Optimization).
    We exploit symmetries:
    1. beta1 = 0 (global X phase is often redundant or can be absorbed/ignored in expectation for symmetric starts, 
       but actually for MaxCut, the first beta is often close to 0 in optimal solutions for small p? 
       Wait, standard QAOA sets beta_1 != 0. 
       However, let's look at the symmetries.
       The state after first Gamma is diagonal. Applying Beta_1 creates superposition.
       Actually, a known symmetry for QAOA MaxCut is that we can restrict search to 
       0 <= gamma_1 <= gamma_2 <= pi and 0 <= beta_1 <= beta_2 <= pi/2? 
       No, beta ranges are usually [0, pi/2] or similar.
       
       Let's just use a robust global search on the full box but with a smaller dimension if possible.
       Actually, dual_annealing is good for 4D.
    """
    
    # Define objective for dual_annealing
    # We want to maximize expected cut, so minimize negative.
    def obj(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        # Ensure parameters are within reasonable bounds to avoid numerical issues, 
        # though dual_annealing handles bounds.
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val

    # Bounds:
    # Gamma: [0, pi]
    # Beta: [0, pi/2] is often sufficient for QAOA, but let's allow [0, pi] to be safe.
    # Actually, for MaxCut, beta is typically in [0, pi/2].
    # Let's use [0, pi] for both to be safe, dual_annealing will find the minimum.
    
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.pi, np.pi, np.pi, np.pi]
    
    # Initial guess
    x0 = [0.5, 0.5, 0.5, 0.5]
    
    # Run Dual Annealing
    # maxiter is the number of iterations for the annealing process.
    # We have ~120s budget. Dual annealing is slow but global.
    # Let's set a generous maxiter but rely on the deadline check in obj?
    # Dual annealing doesn't easily support early stopping inside the loop without custom callbacks.
    # We'll set maxiter based on n.
    
    # For n=10, 2^10=1024 states. Simulation is fast.
    # Let's try maxiter=1000.
    
    try:
        res = dual_annealing(
            obj, 
            x0, 
            bounds=list(zip(lower, upper)), 
            maxiter=2000,
            seed=42
        )
        
        if res.fun < 1e9:
            best_x = res.x
            best_val = res.fun
        else:
            best_x = x0
            best_val = -_expected_cut(n, cuts, *x0)
            
    except Exception as e:
        # Fallback to simple local search if global fails
        best_x = np.array([0.5, 0.5, 0.5, 0.5])
        best_val = -_expected_cut(n, cuts, *best_x)

    # Polish the result with a local optimizer (Nelder-Mead) for high precision
    from scipy.optimize import minimize
    
    def obj_polish(x):
        if _remaining() < 1.0:
            return 1e10
        return -_expected_cut(n, cuts, *x)
        
    try:
        res_polish = minimize(
            obj_polish, 
            best_x, 
            method='Nelder-Mead', 
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res_polish.fun < best_val:
            best_x = res_polish.x
            best_val = res_polish.fun
    except Exception:
        pass
        
    return best_x

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.35, 0.25], [0.35, 0.25]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    # Optimize 4 parameters for p=2
    opt = _optimize_4params(n, cuts)
    
    # Return as (p, 2) array: rows are (gamma, beta)
    result = np.array([
        [opt[0], opt[1]],  # gamma_1, beta_1
        [opt[2], opt[3]]   # gamma_2, beta_2
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
