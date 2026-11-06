"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

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
    Uses state-vector simulation.
    """
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    # Layer 1: exp(-i gamma1 * C)
    psi = psi * np.exp(-1j * gamma1 * cuts)
    
    # Layer 1: exp(-i beta1 * sum X)
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
            
    # Layer 2: exp(-i gamma2 * C)
    psi = psi * np.exp(-1j * gamma2 * cuts)
    
    # Layer 2: exp(-i beta2 * sum X)
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
    Optimize 4 QAOA parameters using a dense grid search followed by 
    intensive multi-start local search with physics-informed initialization.
    """
    
    # Define objective for minimization (negative expected cut)
    def obj(x):
        if _remaining() < 2.0:
            return 1e10
        g1, b1, g2, b2 = x
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val

    # Bounds
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.pi, np.pi/2, np.pi, np.pi/2]
    
    best_x = None
    best_val = np.inf
    
    # Step 1: Dense grid search to find good starting points
    # Use a grid of 5x5x5x5 = 625 points
    if _remaining() > 20.0:
        grid_res = 5
        g1_vals = np.linspace(0.1, np.pi - 0.1, grid_res)
        b1_vals = np.linspace(0.1, np.pi/2 - 0.1, grid_res)
        g2_vals = np.linspace(0.1, np.pi - 0.1, grid_res)
        b2_vals = np.linspace(0.1, np.pi/2 - 0.1, grid_res)
        
        for g1 in g1_vals:
            for b1 in b1_vals:
                for g2 in g2_vals:
                    for b2 in b2_vals:
                        if _remaining() < 5.0:
                            break
                        x = [g1, b1, g2, b2]
                        val = obj(x)
                        if val < best_val:
                            best_val = val
                            best_x = x
        
        # If grid search found nothing, use heuristic
        if best_x is None:
            best_x = [0.5, 0.5, 0.5, 0.5]
            best_val = obj(best_x)
    
    # Step 2: Multi-start local search from diverse initial points
    # Combine grid best with physics-informed heuristics
    inits = [best_x] if best_x is not None else []
    
    # Add physics-informed initializations
    inits.extend([
        [0.5, 0.5, 0.5, 0.5],
        [0.8, 0.4, 0.6, 0.4],
        [1.0, 0.3, 0.5, 0.5],
        [0.3, 0.6, 0.7, 0.3],
        [0.5, 0.2, 0.8, 0.6],
        [np.pi/4, np.pi/4, np.pi/4, np.pi/4],
        [0.7, 0.35, 0.35, 0.17],
        [0.3, 0.15, 0.15, 0.07],
    ])
    
    # Remove duplicates
    unique_inits = []
    for init in inits:
        if not any(np.allclose(init, u, atol=1e-3) for u in unique_inits):
            unique_inits.append(init)
    
    for init in unique_inits:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(
                obj, 
                init, 
                method='Nelder-Mead', 
                bounds=list(zip(lower, upper)),
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            continue
            
    # Step 3: Final high-precision polish pass
    if _remaining() > 5.0:
        try:
            res_polish = minimize(
                obj, 
                best_x, 
                method='Nelder-Mead', 
                bounds=list(zip(lower, upper)),
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_polish.fun < best_val:
                best_val = res_polish.fun
                best_x = res_polish.x
        except Exception:
            pass
        
    # Fallback if nothing worked
    if best_x is None:
        best_x = np.array([0.5, 0.25, 0.25, 0.125])
        
    return np.array(best_x)

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    
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
