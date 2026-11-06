"""Combined: cached circuit ops + MPS-inspired init + multi-restart L-BFGS-B with polish."""
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


def _exact_ground_state(n, coupling, field):
    key = (n, coupling, field)
    if key not in _H_cache:
        _build_H(n, coupling, field)
    H = _H_cache[key]
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs[:, 0]


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
        idx_0[q] = np.array([i for i in range(dim) if (i & mask) == 0])
        idx_1[q] = idx_0[q] + mask
    _idx_cache[key] = (idx_0, idx_1)
    return _idx_cache[key]


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
            
            vals_0 = state[idx_0].copy()
            vals_1 = state[idx_1].copy()
            
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


def _mps_inspired_init(n, coupling, field, layers):
    """
    Initialize angles using physics-inspired heuristics based on the regime.
    For transverse-field Ising model:
    - Large field: ground state close to |----> (X-basis), so Ry(pi/2)
    - Small field: ground state close to |00..0> or |11..1> (Z-basis), so Ry(0) or Ry(pi)
    - Intermediate: some entanglement, angles between 0 and pi/2
    """
    ratio = field / coupling if coupling != 0 else field
    
    # Smooth interpolation: at ratio=0, angle~0; at ratio->inf, angle~pi/2
    base_angle = (np.pi / 2) * (1 - np.exp(-ratio))
    
    # Create a pattern that varies by layer and position
    inits = []
    
    # Init 1: All same angle
    inits.append(np.full((layers, n), base_angle))
    
    # Init 2: Layer-dependent (first layer more field-like, later layers adjust)
    layer_factors = np.linspace(1.0, 0.7, layers)
    for l in range(layers):
        for q in range(n):
            pass
    arr = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            arr[l, q] = base_angle * layer_factors[l]
    inits.append(arr.copy().ravel())
    
    # Init 3: Position-dependent (center qubits might need different angles)
    arr = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            # Center qubits slightly larger angle
            pos_factor = 1.0 - 0.1 * abs(q - (n - 1) / 2.0) / max(1, (n - 1) / 2.0)
            arr[l, q] = base_angle * pos_factor
    inits.append(arr.copy().ravel())
    
    # Init 4: Alternating small perturbation
    arr = np.zeros((layers, n))
    for l in range(layers):
        for q in range(n):
            pert = 0.05 * np.sin(q * np.pi / n)
            arr[l, q] = base_angle + pert
    inits.append(arr.copy().ravel())
    
    return inits


def _heuristic_init(n, coupling, field, layers):
    """
    Generate diverse initializations combining MPS-inspired and random starts.
    """
    inits = []
    
    # MPS-inspired inits
    mps_inits = _mps_inspired_init(n, coupling, field, layers)
    for init in mps_inits:
        inits.append(init.copy())
    
    # Init: All zeros
    inits.append(np.zeros(layers * n))
    
    # Init: All pi/2
    inits.append(np.full(layers * n, np.pi / 2))
    
    # Init: All pi
    inits.append(np.full(layers * n, np.pi))
    
    # Init: Based on ratio (smooth)
    ratio = field / coupling if coupling != 0 else field
    theta_est = (np.pi / 2) * (1 - np.exp(-ratio))
    inits.append(np.full(layers * n, theta_est))
    
    # Init: Fixed values
    for val in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        inits.append(np.full(layers * n, val))
    
    # Random inits with different seeds
    for i in range(12):
        np.random.seed(100 + i)
        inits.append(np.random.uniform(-np.pi, np.pi, size=(layers * n)))
    
    # Random with bias toward [0, pi]
    for i in range(8):
        np.random.seed(200 + i)
        inits.append(np.random.uniform(0, np.pi, size=(layers * n)))
    
    # Random with bias toward small angles
    for i in range(4):
        np.random.seed(300 + i)
        inits.append(np.random.uniform(-0.5, 0.5, size=(layers * n)))
    
    return inits


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    best_x = np.zeros(layers * n)
    best_val = _objective(best_x, n, coupling, field, layers, e0)
    
    # Generate diverse initializations
    inits = _heuristic_init(n, coupling, field, layers)
    
    num_restarts = min(len(inits), 24 if n <= 4 else 18)
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        x_init = inits[i].copy()
        
        def obj(p):
            return _objective(p, n, coupling, field, layers, e0)
        
        try:
            # Use L-BFGS-B with reasonable settings
            res = minimize(obj, x_init, method='L-BFGS-B',
                          options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
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
                          options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Second polish if time still remains
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            res = minimize(lambda p: _objective(p, n, coupling, field, layers, e0), best_x, 
                          method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
