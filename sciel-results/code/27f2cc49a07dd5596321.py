"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

# Global deadline management
_T0 = time.time()
_DEADLINE = 145.0  # Safety margin from 150s
_ANGLE_CACHE = {}

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _build_cost_diag(n, edges):
    """
    Efficiently compute the diagonal of the MaxCut Hamiltonian C.
    C(b) = number of edges crossing the cut defined by bitstring b.
    We compute this for all 2^n bitstrings.
    """
    num_states = 2 ** n
    c_diag = np.zeros(num_states)
    
    # Precompute edge bit masks for fast checking
    # For edge (u, v), we check if bit u != bit v.
    # This is equivalent to checking if the XOR of the two bits is 1.
    # Or simply: ((mask >> u) ^ (mask >> v)) & 1
    
    # Optimization: Instead of looping through all edges for all masks,
    # we can use bit manipulation tricks or just a tight loop.
    # For n<=10, 2^n <= 1024. Edges <= 45. 1024*45 ~ 46k ops. Very fast.
    
    for mask in range(num_states):
        cut_val = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut_val += 1
        c_diag[mask] = cut_val
        
    return c_diag

def _apply_mixer(state, beta, n):
    """
    Apply exp(-i beta sum X) to state.
    sum X = sum_j X_j.
    exp(-i beta sum X) = prod_j (cos(beta) I_j - i sin(beta) X_j).
    This is a tensor product of identical single-qubit gates.
    Gate U = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    """
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    gate = np.array([[c, s], [s, c]], dtype=complex)
    
    state_tensor = state.reshape([2] * n)
    
    # Apply gate to each qubit
    # We need to be careful with axis ordering.
    # If state index is sum b_k 2^k, and we reshape to [2]*n,
    # axis 0 corresponds to b_0 (LSB), axis n-1 to b_{n-1} (MSB).
    # The operator sum X acts on all qubits independently.
    
    for j in range(n):
        # Move axis j to front
        state_tensor = np.moveaxis(state_tensor, j, 0)
        # Reshape to (-1, 2)
        orig_shape = state_tensor.shape
        state_tensor = state_tensor.reshape(-1, 2)
        # Apply gate: state_new = gate @ state
        state_tensor = gate @ state_tensor
        # Reshape back
        state_tensor = state_tensor.reshape(orig_shape)
        # Move axis back
        state_tensor = np.moveaxis(state_tensor, 0, j)
        
    return state_tensor.reshape(-1)

def _expected_cut_value(n, edges, gammas, betas, c_diag=None):
    """
    Compute expected cut for given parameters.
    """
    if c_diag is None:
        c_diag = _build_cost_diag(n, edges)
        
    # Initial state |+>^n
    state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    
    for k in range(len(gammas)):
        # Apply Cost: exp(-i gamma C)
        phase = np.exp(-1j * gammas[k] * c_diag)
        state = state * phase
        
        # Apply Mixer: exp(-i beta sum X)
        state = _apply_mixer(state, betas[k], n)
        
    # Compute expectation <psi| C |psi>
    prob = np.abs(state)**2
    return np.dot(prob, c_diag)

def _optimize_angles(n, edges, p):
    """
    Optimize angles for a specific graph.
    Uses Nelder-Mead with multiple starting points.
    """
    if _time_left() < 2.0:
        return np.tile([0.35, 0.25], (p, 1))
        
    c_diag = _build_cost_diag(n, edges)
    
    def objective(params):
        gammas = params[:p]
        betas = params[p:]
        val = _expected_cut_value(n, edges, gammas, betas, c_diag)
        return -val  # Minimize negative expected cut
        
    best_val = np.inf
    best_params = None
    
    # Define bounds (though NM doesn't strictly need them, they help initialization)
    bounds = [(0, np.pi)] * (2 * p)
    
    # Starting points:
    # 1. Seed values
    starts = [
        [0.35, 0.25] * p,
        [0.5, 0.2] * p,
        [1.0, 0.5] * p,
        [np.pi/4, np.pi/8] * p,
        [0.1, 0.1] * p,
        [1.5, 0.5] * p,
    ]
    
    # Add a few random starts if time allows
    if _time_left() > 5.0:
        rng = np.random.RandomState(42)
        for _ in range(3):
            starts.append(rng.uniform(0, np.pi, size=2*p))
            
    for start in starts:
        if _time_left() < 1.0:
            break
            
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue
            
    if best_params is None:
        return np.tile([0.35, 0.25], (p, 1))
        
    gammas = best_params[:p]
    betas = best_params[p:]
    return np.column_stack((gammas, betas))

def angles(n, edges, p):
    """
    Entry point.
    Returns (p, 2) array of angles.
    """
    # Normalize edges for caching
    # Ensure edges are tuples of sorted ints
    norm_edges = tuple(sorted((min(u, v), max(u, v)) for u, v in edges))
    key = (n, norm_edges, p)
    
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
        
    # If time is critical, return default
    if _time_left() < 1.0:
        res = np.tile([0.35, 0.25], (p, 1))
        return res
        
    # Optimize
    res = _optimize_angles(n, edges, p)
    
    # Cache result
    _ANGLE_CACHE[key] = res
    
    return res
# EVOLVE-BLOCK-END
