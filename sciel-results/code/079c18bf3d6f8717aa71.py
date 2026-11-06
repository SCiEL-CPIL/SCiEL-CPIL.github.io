"""Quantum information: fixed-depth QAOA parameter design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_SAFETY = 10.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _T0)

def _precompute_cuts(n, edges):
    num_states = 2**n
    cuts = np.zeros(num_states, dtype=np.float64)
    states = np.arange(num_states, dtype=np.int64)
    
    # Vectorized calculation of cut values for all basis states
    for u, v in edges:
        bits_u = (states >> u) & 1
        bits_v = (states >> v) & 1
        mask = bits_u != bits_v
        cuts[mask] += 1.0
        
    return cuts

def _apply_beta(psi, beta, n):
    """Apply exp(-i beta sum X) to the state vector."""
    c = np.cos(beta)
    s = np.sin(beta)
    
    # We apply the rotation bit by bit.
    # For bit k, the state index b has bit k at position k (LSB is bit 0).
    # We reshape the state vector to separate the bit k dimension.
    # Shape: (2^(n-k-1), 2, 2^k)
    # Axis 1 corresponds to bit k.
    
    for k in range(n):
        shape = (2**(n-k-1), 2, 2**k)
        psi_reshaped = psi.reshape(shape)
        
        p0 = psi_reshaped[:, 0, :].copy()
        p1 = psi_reshaped[:, 1, :].copy()
        
        psi_reshaped[:, 0, :] = c * p0 - 1j * s * p1
        psi_reshaped[:, 1, :] = -1j * s * p0 + c * p1
        
        psi = psi_reshaped.reshape(-1)
        
    return psi

def _expected_cut(n, cuts, params, p):
    """
    Calculate expected cut for given parameters.
    params: array of shape (p, 2), rows are (gamma, beta).
    """
    psi = np.ones(2**n, dtype=np.complex128) / np.sqrt(2**n)
    
    for i in range(p):
        gamma = params[i, 0]
        beta = params[i, 1]
        
        # Apply Gamma layer: exp(-i gamma C)
        # C is diagonal in Z basis with eigenvalues cuts[b]
        phase = np.exp(-1j * gamma * cuts)
        psi = psi * phase
        
        # Apply Beta layer: exp(-i beta sum X)
        psi = _apply_beta(psi, beta, n)
        
    probs = np.abs(psi)**2
    return np.sum(probs * cuts)

def _optimize_angles(n, edges, p, cuts):
    """
    Optimize QAOA parameters for a specific graph.
    """
    # Initial guess: uniform random or heuristic
    # Heuristic: small angles
    x0 = np.array([0.4, 0.2, 0.3, 0.1]) # gamma1, beta1, gamma2, beta2
    
    def obj(x):
        # Reshape x to (p, 2)
        params = x.reshape(p, 2)
        val = _expected_cut(n, cuts, params, p)
        return -val
        
    # Use Nelder-Mead for derivative-free optimization
    # Bounds are not strictly enforced by NM, but angles are periodic.
    # We can wrap them if needed, but NM usually stays in local region.
    
    try:
        res = minimize(obj, x0, method='Nelder-Mead', 
                       options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
        return res.x
    except Exception:
        return x0

def angles(n, edges, p):
    if _remaining() < 5.0:
        # Fallback to heuristic if time is short
        # Simple heuristic: gamma ~ 0.3, beta ~ 0.2
        return np.array([[0.3, 0.2], [0.2, 0.1]])

    cuts = _precompute_cuts(n, edges)
    
    # Try a few random restarts to avoid local minima
    best_params = None
    best_val = -np.inf
    
    # Number of restarts depends on time remaining
    # Each eval takes ~ O(2^n * n * p). For n=10, ~1024*10*2 = 20k ops.
    # 300 iterations * 20k = 6M ops. Fast in numpy.
    # Let's do 3 restarts if time allows.
    
    num_restarts = 3
    if _remaining() < 20.0:
        num_restarts = 1
        
    for r in range(num_restarts):
        if r == 0:
            x0 = np.array([0.4, 0.2, 0.3, 0.1])
        else:
            # Random perturbation around previous best or random
            if best_params is not None:
                x0 = best_params + np.random.normal(0, 0.2, 4)
            else:
                x0 = np.random.uniform(0, np.pi/2, 4)
                
        # Check time
        if _remaining() < 5.0:
            break
            
        params = _optimize_angles(n, edges, p, cuts)
        
        # Evaluate best found
        val = _expected_cut(n, cuts, params, p)
        
        if val > best_val:
            best_val = val
            best_params = params.copy()
            
    if best_params is None:
        # Should not happen, but fallback
        return np.array([[0.3, 0.2], [0.2, 0.1]])
        
    return best_params.reshape(p, 2)

# EVOLVE-BLOCK-END
