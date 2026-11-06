"""Optimize 3-layer Ry/CZ circuit parameters for transverse-field Ising chains."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Caches for Hamiltonian, ground energy, and CZ masks
_H_cache = {}
_e0_cache = {}
_cz_mask_cache = {}


def _build_hamiltonian(n, coupling, field):
    """Build the Ising Hamiltonian matrix for open chain with caching."""
    key = (n, coupling, field)
    if key in _H_cache:
        return _H_cache[key]
    
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    
    # ZZ terms: -coupling * sum Z_i Z_{i+1}
    for x in range(dim):
        zz_val = 0.0
        for i in range(n - 1):
            xi = (x >> i) & 1
            xj = (x >> (i + 1)) & 1
            if xi == xj:
                zz_val += 1.0
            else:
                zz_val -= 1.0
        H[x, x] -= coupling * zz_val
    
    # X terms: -field * sum X_i
    for i in range(n):
        for x in range(dim):
            xp = x ^ (1 << i)
            H[xp, x] -= field
    
    _H_cache[key] = H
    return H


def _exact_ground_energy(n, coupling, field):
    """Compute exact ground energy using eigendecomposition with caching."""
    key = (n, coupling, field)
    if key in _e0_cache:
        return _e0_cache[key]
    
    H = _build_hamiltonian(n, coupling, field)
    eigvals = np.linalg.eigvalsh(H)
    e0 = eigvals[0]
    _e0_cache[key] = e0
    return e0


def _get_cz_masks(n):
    """Pre-compute bitmasks for CZ operations."""
    key = n
    if key in _cz_mask_cache:
        return _cz_mask_cache[key]
    
    dim = 1 << n
    masks = []
    for i in range(n - 1):
        mask = np.zeros(dim, dtype=np.float64)
        for x in range(dim):
            bi = (x >> i) & 1
            bj = (x >> (i + 1)) & 1
            if bi == 1 and bj == 1:
                mask[x] = -1.0
            else:
                mask[x] = 1.0
        masks.append(mask)
    
    _cz_mask_cache[key] = masks
    return masks


def _apply_circuit_state(theta, n, layers):
    """
    Apply the circuit to |0>^n and return the state vector.
    Uses vectorized operations for speed.
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0
    
    cz_masks = _get_cz_masks(n)
    
    for l in range(layers):
        # Apply Ry to every qubit
        for i in range(n):
            theta_i = theta[l, i]
            c = np.cos(theta_i / 2)
            s = np.sin(theta_i / 2)
            
            # Vectorized Ry application
            # Split states into those with bit i = 0 and bit i = 1
            bit_i_mask = np.array([(x >> i) & 1 for x in range(dim)], dtype=np.int32)
            mask_0 = (bit_i_mask == 0)
            mask_1 = (bit_i_mask == 1)
            
            new_state = np.zeros_like(state)
            
            # For bit i = 0: |0> -> c|0> + s|1>
            idx_0 = np.where(mask_0)[0]
            idx_1 = idx_0 | (1 << i)
            
            new_state[idx_0] += c * state[idx_0]
            new_state[idx_1] += s * state[idx_0]
            
            # For bit i = 1: |1> -> -s|0> + c|1>
            idx_1_orig = np.where(mask_1)[0]
            idx_0_from_1 = idx_1_orig & ~(1 << i)
            
            new_state[idx_1_orig] += c * state[idx_1_orig]
            new_state[idx_0_from_1] -= s * state[idx_1_orig]
            
            state = new_state
        
        # Apply CZ on adjacent pairs if not final layer
        if l < layers - 1:
            for i in range(n - 1):
                state *= cz_masks[i]
    
    return state


def _circuit_energy(theta, n, coupling, field, H):
    """Compute <psi|H|psi> where psi is the circuit state."""
    psi = _apply_circuit_state(theta, n, layers=3)
    energy = np.vdot(psi, H @ psi).real
    return energy


def _objective_flat(theta_flat, n, coupling, field, H, ground_energy):
    """Objective function: (circuit energy - ground energy) / n"""
    theta = theta_flat.reshape(3, n)
    energy = _circuit_energy(theta, n, coupling, field, H)
    return (energy - ground_energy) / n


def angles(n, coupling, field, layers):
    """
    Optimize parameters of a fixed three-layer Ry/CZ circuit to approximate 
    ground states of open transverse-field Ising chains.
    
    Args:
        n: number of qubits (4 or 6)
        coupling: Ising coupling (1.0)
        field: transverse field (0.45, 1.0, 1.65)
        layers: number of layers (3)
    
    Returns:
        (layers, n) array of angles in radians
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Build Hamiltonian and compute exact ground energy
    H = _build_hamiltonian(n, coupling, field)
    ground_energy = _exact_ground_energy(n, coupling, field)
    
    dim = 1 << n
    total_params = layers * n
    
    best_theta = None
    best_obj = np.inf
    
    # Generate multiple initial guesses
    initial_guesses = []
    
    # Guess 1: zeros
    initial_guesses.append(np.zeros(total_params))
    
    # Guess 2: small random
    rng = np.random.RandomState(42)
    initial_guesses.append(rng.uniform(-np.pi/4, np.pi/4, total_params))
    
    # Guess 3: symmetric (same angle for all qubits in each layer)
    for l in range(layers):
        initial_guesses.append(np.array([0.1 * (l + 1)] * n).flatten())
    
    # Guess 4: Heuristic based on field strength
    # For small field, angles should be small; for large field, larger angles
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        guess = np.zeros(total_params)
        for l in range(layers):
            for i in range(n):
                # Symmetric initialization with field-dependent scale
                guess[l * n + i] = alpha * field * (1.0 / (l + 1))
        initial_guesses.append(guess)
    
    # Try L-BFGS-B from each initial guess
    for init in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _objective_flat,
                init,
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass
    
    # If we have time, do differential evolution for global search
    if time.time() < deadline - 40:
        # Differential evolution with bounded search
        bounds = [(-np.pi, np.pi)] * total_params
        
        try:
            de_result = differential_evolution(
                _objective_flat,
                bounds,
                args=(n, coupling, field, H, ground_energy),
                maxiter=80,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False
            )
            if de_result.fun < best_obj:
                best_obj = de_result.fun
                best_theta = de_result.x.reshape(layers, n)
        except Exception:
            pass
    
    # Final polish with L-BFGS-B if we found a good solution
    if best_theta is not None and time.time() < deadline - 15:
        try:
            res = minimize(
                _objective_flat,
                best_theta.flatten(),
                args=(n, coupling, field, H, ground_energy),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_theta = res.x.reshape(layers, n)
        except Exception:
            pass
    
    # If no good solution found, return a reasonable default
    if best_theta is None:
        best_theta = np.zeros((layers, n))
    
    return best_theta
