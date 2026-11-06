"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 145.0
_SAFETY = 5.0
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
    Optimized state-vector simulation with reduced memory copies.
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
            p0 = psi_reshaped[:, 0, :]
            p1 = psi_reshaped[:, 1, :]
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
            p0 = psi_reshaped[:, 0, :]
            p1 = psi_reshaped[:, 1, :]
            psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
            psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
            psi = psi_reshaped.reshape(-1)

    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _local_search(n, cuts, x0, maxiter=500, tol=1e-8):
    """Perform a local Nelder-Mead search starting from x0."""
    def obj(x):
        if _remaining() < 1.0:
            return 1e10
        g1, b1, g2, b2 = x
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        return -val
    
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.pi, np.pi/2, np.pi, np.pi/2]
    
    try:
        res = minimize(
            obj, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        best_x = np.clip(res.x, lower, upper)
        best_val = -res.fun
        return best_x, best_val
    except Exception:
        return x0, _expected_cut(n, cuts, *x0)

def angles(n, edges, p):
    if _remaining() < 1.0:
        return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    
    cuts = _precompute_cuts(n, edges)
    
    best_x = [0.5, 0.5, 0.5, 0.5]
    best_val = _expected_cut(n, cuts, *best_x)
    
    # Phase 1: Broad Random Search with Light Local Refinement
    # Generate many random candidates
    n_random = 300
    rng = np.random.default_rng(713) # Fixed seed for reproducibility of search
    
    candidates = []
    for i in range(n_random):
        if _remaining() < 5.0:
            break
        # Sample parameters
        # gamma in [0, pi], beta in [0, pi/2]
        g1 = rng.uniform(0, np.pi)
        b1 = rng.uniform(0, np.pi/2)
        g2 = rng.uniform(0, np.pi)
        b2 = rng.uniform(0, np.pi/2)
        
        # Quick evaluation
        val = _expected_cut(n, cuts, g1, b1, g2, b2)
        candidates.append((val, [g1, b1, g2, b2]))
        
    # Sort by value (descending)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Local Refinement on Top Candidates
    n_refine = 15
    for i in range(min(n_refine, len(candidates))):
        if _remaining() < 10.0:
            break
        
        val, x = candidates[i]
        # If this candidate is already better than current best, or if we haven't refined it
        # We should refine the top candidates to see if they are local maxima
        refined_x, refined_val = _local_search(n, cuts, x, maxiter=300, tol=1e-7)
        
        if refined_val > best_val:
            best_val = refined_val
            best_x = refined_x
            
    # Phase 3: Deep Polish on the Best Result
    if _remaining() > 15.0:
        final_x, final_val = _local_search(n, cuts, best_x, maxiter=2000, tol=1e-10)
        if final_val > best_val:
            best_val = final_val
            best_x = final_x

    result = np.array([
        [best_x[0], best_x[1]],
        [best_x[2], best_x[3]]
    ], dtype=np.float64)
    
    return result

# EVOLVE-BLOCK-END
