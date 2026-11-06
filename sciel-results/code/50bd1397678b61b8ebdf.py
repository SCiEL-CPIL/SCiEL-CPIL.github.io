"""Optimize 3-layer Ry/CZ circuit for open transverse-field Ising chains."""
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

# Caches
_H_cache = {}
_e0_cache = {}
_idx_cache = {}

def _build_hamiltonian(n, coupling, field):
    """Build the Hamiltonian matrix for the open chain Ising model."""
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # Z_i Z_{i+1} terms
    for i in range(n - 1):
        for idx in range(dim):
            bi = (idx >> i) & 1
            bj = (idx >> (i + 1)) & 1
            z_i = 1 if bi == 0 else -1
            z_j = 1 if bj == 0 else -1
            H[idx, idx] += -coupling * z_i * z_j
    
    # X_i terms
    for i in range(n):
        for idx in range(dim):
            bi = (idx >> i) & 1
            flip_idx = idx ^ (1 << i)
            H[flip_idx, idx] += -field
    
    _H_cache[key] = H
    return H

def _exact_ground_state(n, coupling, field):
    """Compute exact ground state and energy."""
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    H = _build_hamiltonian(n, coupling, field)
    eigvals, eigvecs = np.linalg.eigh(H)
    ground_energy = eigvals[0].real
    ground_state = eigvecs[:, 0]
    _e0_cache[key] = (ground_energy, ground_state)
    return _e0_cache[key]

def _get_indices(n):
    """Pre-compute index arrays for each qubit's 0 and 1 subspaces, and CZ masks."""
    key = n
    if key in _idx_cache:
        return _idx_cache[key]
    dim = 2**n
    qubit_indices = []
    for q in range(n):
        mask = 1 << q
        idx_0 = np.array([i for i in range(dim) if (i & mask) == 0], dtype=np.int64)
        idx_1 = idx_0 + mask
        qubit_indices.append((idx_0, idx_1))
    # Pre-compute CZ masks for adjacent pairs
    cz_masks = []
    all_idx = np.arange(dim)
    for i in range(n - 1):
        m1 = 1 << i
        m2 = 1 << (i + 1)
        both_set = ((all_idx & m1) != 0) & ((all_idx & m2) != 0)
        cz_masks.append(both_set)
    _idx_cache[n] = (qubit_indices, cz_masks)
    return _idx_cache[n]

def _simulate_circuit(angles, n):
    """
    Simulate the circuit: start |0>^n, apply layers of Ry then CZ.
    angles: (layers, n) array
    Returns: statevector (2^n,) as float64 (real)
    """
    layers = angles.shape[0]
    qubit_indices, cz_masks = _get_indices(n)
    dim = 2**n
    
    # Start state |0>^n
    psi = np.zeros(dim, dtype=np.float64)
    psi[0] = 1.0

    for layer in range(layers):
        # Apply Ry to each qubit
        for i in range(n):
            theta = angles[layer, i]
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            
            idx_0, idx_1 = qubit_indices[i]
            vals_0 = psi[idx_0]
            vals_1 = psi[idx_1]
            
            new_vals_0 = c * vals_0 - s * vals_1
            new_vals_1 = s * vals_0 + c * vals_1
            
            psi[idx_0] = new_vals_0
            psi[idx_1] = new_vals_1
        
        # Apply CZ to adjacent pairs (except after final layer)
        if layer < layers - 1:
            for i in range(n - 1):
                psi[cz_masks[i]] *= -1.0

    return psi

def _circuit_energy(psi, n, coupling, field):
    """Compute energy <psi|H|psi>."""
    H = _build_hamiltonian(n, coupling, field)
    return np.dot(psi, H @ psi)

def _energy_gap_objective(params, n, coupling, field, ground_energy):
    """Objective: mean energy gap per spin."""
    angles = params.reshape(3, n)
    psi = _simulate_circuit(angles, n)
    energy = _circuit_energy(psi, n, coupling, field)
    return (energy - ground_energy) / n

def _mean_field_init(n, coupling, field):
    """
    Variational mean-field / product state initialization.
    """
    angles = np.zeros((3, n))
    
    z_vals = np.zeros(n)
    for sweep in range(10):
        for i in range(n):
            z_left = z_vals[i-1] if i > 0 else 0.0
            z_right = z_vals[i+1] if i < n-1 else 0.0
            eff_z = coupling * (z_left + z_right) / 2.0
            if abs(eff_z) < 1e-10:
                theta_i = np.pi / 2
            else:
                theta_i = np.arctan2(field, eff_z)
                if theta_i < 0:
                    theta_i = np.pi + theta_i
                if theta_i > np.pi:
                    theta_i = 2*np.pi - theta_i
            z_vals[i] = np.cos(theta_i)
            angles[0, i] = theta_i
    
    # For layers 1 and 2, start with small corrections
    angles[1] = 0.1 * angles[0]
    angles[2] = 0.05 * angles[0]
    
    return angles

def _optimize_for_case(n, coupling, field, time_budget):
    """Optimize angles for a specific case."""
    ground_energy, _ = _exact_ground_state(n, coupling, field)
    
    # Initial guess
    angles0 = _mean_field_init(n, coupling, field)
    params0 = angles0.flatten()
    
    best_params = params0.copy()
    best_energy = _energy_gap_objective(params0, n, coupling, field, ground_energy)
    
    # Try multiple optimization methods and initializations
    inits = [params0]
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    for _ in range(3):
        perturbed = params0 + rng.normal(0, 0.5, size=params0.shape)
        inits.append(perturbed)
    
    # Also try all zeros and all pi/2
    inits.append(np.zeros(3 * n))
    inits.append(np.full(3 * n, np.pi / 2))
    
    for init in inits:
        if _remaining() < 15:
            break
        
        try:
            result = minimize(
                _energy_gap_objective,
                init,
                args=(n, coupling, field, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Polish with tighter tolerance if time allows
    if _remaining() > 20:
        try:
            result = minimize(
                _energy_gap_objective,
                best_params,
                args=(n, coupling, field, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params.reshape(3, n)

def angles(n, coupling, field, layers):
    """
    Main entry point. Returns angles (layers, n) in radians.
    """
    # Cache ground state computations
    ground_energy, _ = _exact_ground_state(n, coupling, field)
    
    # Optimize
    result = _optimize_for_case(n, coupling, field, _remaining())
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=np.pi, neginf=-np.pi)
    
    return result
