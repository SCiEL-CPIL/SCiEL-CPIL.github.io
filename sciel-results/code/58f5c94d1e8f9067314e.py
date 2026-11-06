"""Quantum information: shallow-circuit Ising ground states - MPS-inspired optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_H_cache = {}
_e0_cache = {}
_evec_cache = {}
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
    eigvals, _ = np.linalg.eigh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0


def _exact_ground_state(n, coupling, field):
    key = (n, coupling, field)
    if key in _evec_cache:
        return _evec_cache[key]
    H = _build_H(n, coupling, field)
    _, eigvecs = np.linalg.eigh(H)
    psi = eigvecs[:, 0].copy()
    # Normalize and fix global phase
    psi = psi / np.linalg.norm(psi)
    _evec_cache[key] = psi
    return psi


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
    return idx_0, idx_1


def _circuit_state_fast(n, thetas, layers):
    """
    Vectorized application of the Ry/CZ circuit.
    thetas: (layers, n) array of rotation angles.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0 + 0j

    idx_0, idx_1 = _get_idx_pairs(n)
    cz_masks = _get_cz_masks(n)

    for layer in range(layers):
        # Apply Ry to all qubits
        for q in range(n):
            theta = thetas[layer, q]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)

            i0 = idx_0[q]
            i1 = idx_1[q]

            vals_0 = state[i0]
            vals_1 = state[i1]

            new_vals_0 = c * vals_0 - s * vals_1
            new_vals_1 = s * vals_0 + c * vals_1

            state[i0] = new_vals_0
            state[i1] = new_vals_1

        # Apply CZ between layers (except after the last layer)
        if layer < layers - 1:
            for mask in cz_masks:
                state[mask] *= -1.0

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
    Initialize angles using MPS-inspired approach:
    Fit the circuit's effective structure to approximate the ground state.
    
    For a 1D chain, the ground state can be approximated by an MPS.
    We use this to guide initialization of the Ry angles.
    
    Strategy:
    1. Compute the exact ground state
    2. For each layer, try to match the single-qubit reductions
    3. Use the phase structure to guide angle choices
    """
    psi = _exact_ground_state(n, coupling, field)
    dim = 1 << n
    
    # Compute single-qubit reduced density matrices
    # For each qubit, compute the local expectation values to guide angles
    
    thetas = np.zeros((layers, n), dtype=np.float64)
    
    # For each qubit, estimate the optimal Ry angle
    # based on the local field direction in the ground state
    for q in range(n):
        # Compute local X and Z expectations
        mask = 1 << q
        # <X_q> = Re(<psi|X_q|psi>)
        x_val = 0.0
        z_val = 0.0
        for state_idx in range(dim):
            if abs(psi[state_idx]) > 1e-15:
                # X expectation contribution
                flipped = state_idx ^ mask
                x_val += np.conj(psi[state_idx]) * psi[flipped] + np.conj(psi[flipped]) * psi[state_idx]
                x_val = np.real(x_val) / 2.0
                # Z expectation
                bit = (state_idx & mask) != 0
                z_val += (1.0 if bit else -1.0) * (abs(psi[state_idx])**2)
        
        # The optimal Ry angle to align with the local field
        # Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
        # We want to maximize overlap with the local state direction
        # If local state is mostly |0>, theta ~ 0; if mostly |1>, theta ~ pi
        # If equal mix, theta ~ pi/2
        
        # Use arctan2 to find the angle
        # tan(theta/2) = x_val / (1 - z_val) approximately
        # Better: theta = 2 * arctan(x_val / (1 - z_val))
        if abs(1.0 - z_val) > 1e-10:
            theta_est = 2.0 * np.arctan2(x_val, 1.0 - z_val)
        else:
            theta_est = np.pi if x_val > 0 else 0.0
        
        # Clamp to reasonable range
        theta_est = np.clip(theta_est, -np.pi, np.pi)
        
        # Use this as initialization for all layers (with some variation)
        thetas[0, q] = theta_est
        thetas[1, q] = theta_est * 0.7  # Slightly different for layer 2
        thetas[2, q] = theta_est * 0.5  # And for layer 3
    
    return thetas


def _heuristic_init(n, coupling, field, layers):
    """
    Heuristic initialization based on physics intuition.
    """
    ratio = field / coupling if coupling != 0 else field
    
    # For large ratio (field-dominated), we want ~pi/2
    # For small ratio (coupling-dominated), we want ~0
    theta_est = (np.pi / 2) * (1 - np.exp(-ratio))
    
    thetas = np.full((layers, n), theta_est)
    
    # Add some spatial variation
    for q in range(n):
        # Edge qubits may behave slightly differently
        edge_factor = 1.0 if q == 0 or q == n-1 else 0.95
        thetas[:, q] *= edge_factor
    
    return thetas


def angles(n, coupling, field, layers):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    e0 = _exact_ground_energy(n, coupling, field)
    
    best_x = np.zeros(layers * n)
    best_val = _objective(best_x, n, coupling, field, layers, e0)
    
    # Generate diverse initializations
    inits = []
    
    # Init 1: MPS-inspired
    try:
        theta_mps = _mps_inspired_init(n, coupling, field, layers)
        inits.append(theta_mps.flatten())
    except Exception:
        pass
    
    # Init 2: Heuristic
    try:
        theta_heur = _heuristic_init(n, coupling, field, layers)
        inits.append(theta_heur.flatten())
    except Exception:
        pass
    
    # Init 3: All zeros
    inits.append(np.zeros(layers * n))
    
    # Init 4: All pi/2
    inits.append(np.full(layers * n, np.pi / 2))
    
    # Init 5: All pi
    inits.append(np.full(layers * n, np.pi))
    
    # Init 6: Based on ratio
    ratio = field / coupling if coupling != 0 else field
    theta_est = (np.pi / 2) * (1 - np.exp(-ratio))
    inits.append(np.full(layers * n, theta_est))
    
    # Init 7-12: Random with different seeds
    for i in range(6):
        np.random.seed(100 + i)
        inits.append(np.random.uniform(-np.pi, np.pi, size=(layers * n)))
    
    # Init 13-16: Random with bias toward [0, pi]
    for i in range(4):
        np.random.seed(200 + i)
        inits.append(np.random.uniform(0, np.pi, size=(layers * n)))
    
    # Sort inits by initial objective value (best first)
    def _init_score(x):
        try:
            return _objective(x, n, coupling, field, layers, e0)
        except Exception:
            return float('inf')
    
    scored_inits = [(x, _init_score(x)) for x in inits]
    scored_inits.sort(key=lambda t: t[1])
    
    # Track the best from initializations
    for x_init, val in scored_inits:
        if val < best_val:
            best_val = val
            best_x = x_init.copy()
    
    # Determine number of restarts based on time budget
    num_restarts = min(len(scored_inits), 20 if n <= 4 else 16)
    
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        x_init = scored_inits[i][0].copy()
        
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
                          options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # One more polish if time allows
    remaining = deadline - time.time()
    if remaining > 2.0:
        try:
            res = minimize(lambda p: _objective(p, n, coupling, field, layers, e0), best_x, 
                          method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape((layers, n))
# EVOLVE-BLOCK-END
