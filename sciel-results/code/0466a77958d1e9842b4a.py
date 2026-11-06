# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_cz_mask_cache = {}
_idx_cache = {}

def _build_H(n, coupling, field):
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    # Z terms
    for i in range(n - 1):
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> (i + 1)) & 1
            zz_val = 1.0 if bi == bj else -1.0
            H[state, state] += -coupling * zz_val
    # X terms
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += -field
    _H_cache[key] = H
    return H

def _exact_ground_energy(n, coupling, field):
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_H(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0

def _get_cz_masks(n):
    key = n
    if key in _cz_mask_cache:
        return _cz_mask_cache[key]
    dim = 1 << n
    arange = np.arange(dim)
    masks = []
    for i in range(n - 1):
        mask1 = 1 << i
        mask2 = 1 << (i + 1)
        both_set = ((arange & mask1) != 0) & ((arange & mask2) != 0)
        masks.append(both_set)
    _cz_mask_cache[key] = masks
    return masks

def _get_idx_pairs(n):
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 1 << n
    idx_0 = {}
    idx_1 = {}
    for q in range(n):
        mask = 1 << q
        idx_0[q] = np.arange(dim)[(np.arange(dim) & mask) == 0]
        idx_1[q] = idx_0[q] + mask
    _idx_cache[key] = (idx_0, idx_1)
    return (idx_0, idx_1)

def _circuit_state_fast(n, thetas, layers):
    """
    Vectorized application of the Ry/CZ circuit.
    thetas: (layers, n) array of rotation angles.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j
    
    idx_0_cache, idx_1_cache = _get_idx_pairs(n)
    cz_masks = _get_cz_masks(n)

    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0 = idx_0_cache[q]
            idx_1 = idx_1_cache[q]
            
            vals_0 = state[idx_0]
            vals_1 = state[idx_1]
            
            new_vals_0 = c * vals_0 - s * vals_1
            new_vals_1 = s * vals_0 + c * vals_1
            
            state[idx_0] = new_vals_0
            state[idx_1] = new_vals_1

        # Apply CZ between layers (except after the last layer)
        if layer < layers - 1:
            for i in range(n - 1):
                state[cz_masks[i]] *= -1.0

    return state

def _circuit_energy(n, coupling, field, thetas, layers):
    H = _build_H(n, coupling, field)
    psi = _circuit_state_fast(n, thetas, layers)
    return np.real(np.conj(psi) @ H @ psi)

def _objective(params, n, coupling, field, layers, e0):
    thetas = params.reshape((layers, n))
    e_circuit = _circuit_energy(n, coupling, field, thetas, layers)
    return (e_circuit - e0) / n

def _heuristic_init(n, coupling, field, layers):
    """
    Heuristic initialization based on physics intuition.
    """
    return None

def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    best_x = np.zeros(layers * n)
    best_val = _objective(best_x, n, coupling, field, layers, e0)
    
    # Generate diverse initializations
    inits = []
    
    # Init 1: All zeros
    inits.append(np.zeros(layers * n))
    
    # Init 2: All pi/2
    inits.append(np.full(layers * n, np.pi / 2))
    
    # Init 3: All pi
    inits.append(np.full(layers * n, np.pi))
    
    # Init 4: Based on ratio
    ratio = field / coupling if coupling != 0 else field
    # Smooth interpolation
    theta_est = (np.pi / 2) * (1 - np.exp(-ratio))
    inits.append(np.full(layers * n, theta_est))
    
    # Init 5: Alternating pattern for antiferromagnetic-like features (though this is ferro)
    # For transverse field, maybe small angles?
    inits.append(np.full(layers * n, 0.5))
    inits.append(np.full(layers * n, 1.0))
    inits.append(np.full(layers * n, 1.5))
    
    # Init 6-15: Random with different seeds
    for i in range(10):
        np.random.seed(100 + i)
        inits.append(np.random.uniform(-np.pi, np.pi, size=(layers * n)))
    
    # Init 16-20: Random with bias toward [0, pi]
    for i in range(5):
        np.random.seed(200 + i)
        inits.append(np.random.uniform(0, np.pi, size=(layers * n)))

    num_restarts = min(len(inits), 20 if n <= 4 else 16)
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        x_init = inits[i].copy()
        
        def obj(p):
            return _objective(p, n, coupling, field, layers, e0)
        
        try:
            # Use L-BFGS-B with tighter tolerance and more iterations
            res = minimize(obj, x_init, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # If time remains, do a final polish from the best solution
    remaining = deadline - time.time()
    if remaining > 3.0:
        try:
            res = minimize(lambda p: _objective(p, n, coupling, field, layers, e0), best_x, 
                          method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
